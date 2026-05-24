#!/usr/bin/env python3
"""Profile Qwen2.5-1.5B forward/backward speed with Grove-FSDP or torch FSDP2.

Run with torchrun, for example:

    torchrun --nproc_per_node=8 scripts/profile_qwen25_1p5b_fsdp.py --backend grove
    torchrun --nproc_per_node=8 scripts/profile_qwen25_1p5b_fsdp.py --backend torch-fsdp2

By default this constructs the Qwen2.5-1.5B architecture from an in-script
config and random weights. Pass --from-pretrained to load model weights instead.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


@dataclass(frozen=True)
class Result:
    backend: str
    world_size: int
    dtype: str
    batch_size: int
    seq_len: int
    warmup_steps: int
    steps: int
    elapsed_s: float
    ms_per_iter: float
    tokens_per_s: float
    peak_allocated_gib: float
    peak_reserved_gib: float


@dataclass(frozen=True)
class GroveLayoutStats:
    groups: int
    raw_numel: int
    weight_bucket_numel: int
    grad_bucket_numel: int
    main_weight_bucket_numel: int
    padded_numel: int
    weight_workspace_allocations: int
    weight_workspace_peak_live: int
    transpose_workspace_allocations: int
    transpose_workspace_peak_live: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=("grove", "torch-fsdp2", "both"),
        default="both",
        help="Which FSDP implementation to profile.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-1.5B",
        help="Hugging Face model id or local path used with --from-pretrained/--config-from-pretrained.",
    )
    parser.add_argument(
        "--from-pretrained",
        action="store_true",
        help="Load pretrained weights instead of constructing random weights from config.",
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        help="Pass low_cpu_mem_usage=True when --from-pretrained is set.",
    )
    parser.add_argument(
        "--config-from-pretrained",
        action="store_true",
        help="Load config from --model-name but still initialize random weights.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Microbatch size per rank.")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Untimed warmup iterations.")
    parser.add_argument("--steps", type=int, default=20, help="Timed iterations.")
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Model and activation dtype.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on the Hugging Face model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used for model init and synthetic batches.",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Local rank. torchrun sets LOCAL_RANK automatically.",
    )
    parser.add_argument(
        "--profile-dir",
        default=None,
        help="Optional directory for torch.profiler Chrome traces.",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=3,
        help="Number of active timed steps to capture when --profile-dir is set.",
    )
    parser.add_argument(
        "--grove-overlap",
        action="store_true",
        help="Enable Grove overlap_grad_reduce and overlap_param_gather.",
    )
    parser.add_argument(
        "--disable-bucketing",
        action="store_true",
        help="Pass disable_bucketing=True to Grove-FSDP.",
    )
    parser.add_argument(
        "--print-grove-layout",
        action="store_true",
        help="Print Grove DBuffer bucket/padding stats after wrapping.",
    )
    parser.add_argument(
        "--grove-dbuffer-workspace-size",
        type=int,
        default=0,
        help="Opt-in Grove DBuffer workspace pool size. 0 uses storage-resize allocation.",
    )
    return parser.parse_args()


def dtype_from_arg(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def rank_print(*values: object) -> None:
    if is_rank0():
        print(*values, flush=True)


def init_distributed(local_rank: int) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return dist.get_rank(), dist.get_world_size(), device


def get_qwen_classes() -> tuple[type[Any], type[Any], type[Any]]:
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    except ImportError as exc:
        raise RuntimeError(
            "This benchmark requires transformers. Install it in the target "
            "environment or run from an environment with the Qwen2 classes available."
        ) from exc
    return Qwen2Config, Qwen2ForCausalLM, Qwen2DecoderLayer


def local_qwen25_1p5b_config(dtype: torch.dtype) -> Any:
    Qwen2Config, _, _ = get_qwen_classes()
    return Qwen2Config(
        vocab_size=151936,
        hidden_size=1536,
        intermediate_size=8960,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        tie_word_embeddings=True,
        rope_theta=1_000_000.0,
        torch_dtype=dtype,
    )


def build_model(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> Any:
    Qwen2Config, Qwen2ForCausalLM, _ = get_qwen_classes()
    torch.manual_seed(args.seed)

    if args.from_pretrained:
        model = Qwen2ForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    else:
        config = (
            Qwen2Config.from_pretrained(args.model_name)
            if args.config_from_pretrained
            else local_qwen25_1p5b_config(dtype)
        )
        config.use_cache = False
        if hasattr(config, "torch_dtype"):
            config.torch_dtype = dtype
        model = Qwen2ForCausalLM(config)

    model.config.use_cache = False
    model.to(device=device, dtype=dtype)
    model.train()
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def apply_torch_fsdp2(model: torch.nn.Module, dtype: torch.dtype, world_size: int) -> torch.nn.Module:
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from torch.distributed.fsdp import fully_shard as torch_fully_shard

    _, _, Qwen2DecoderLayer = get_qwen_classes()
    mesh = init_device_mesh("cuda", (world_size,))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=dtype,
        output_dtype=dtype,
    )
    for layer in model.modules():
        if isinstance(layer, Qwen2DecoderLayer):
            torch_fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    torch_fully_shard(model, mesh=mesh, mp_policy=mp_policy)
    return model


def apply_grove_fsdp(
    model: torch.nn.Module,
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
    world_size: int,
) -> torch.nn.Module:
    from grove_fsdp import MixedPrecisionPolicy, fully_shard_model

    _, _, Qwen2DecoderLayer = get_qwen_classes()
    extra_kwargs = {}
    if args.grove_dbuffer_workspace_size != 0:
        extra_kwargs["grove_fsdp_dbuffer_workspace_size"] = args.grove_dbuffer_workspace_size
    mesh = init_device_mesh(
        "cuda",
        (world_size, 1),
        mesh_dim_names=("fsdp", "tp"),
    )
    mp_policy = MixedPrecisionPolicy(
        main_params_dtype=None,
        main_grads_dtype=dtype,
        grad_comm_dtype=dtype,
    )
    return fully_shard_model(
        module=model,
        device_mesh=mesh,
        dp_shard_dim="fsdp",
        tp_dim="tp",
        zero_dp_strategy="optim_grads_params",
        fsdp_unit_modules=[Qwen2DecoderLayer],
        device=device,
        mixed_precision_policy=mp_policy,
        overlap_grad_reduce=args.grove_overlap,
        overlap_param_gather=args.grove_overlap,
        sync_model_each_microbatch=True,
        disable_bucketing=args.disable_bucketing,
        preproc_state_dict_for_dcp_ckpt=False,
        **extra_kwargs,
    )


def grove_layout_stats(model: torch.nn.Module) -> GroveLayoutStats | None:
    buffer = getattr(model, "param_and_grad_buffer", None)
    if buffer is None:
        return None

    raw_numel = 0
    weight_bucket_numel = 0
    grad_bucket_numel = 0
    main_weight_bucket_numel = 0
    padded_numel = 0
    for group in buffer.parameter_groups:
        group_numel = sum(param.numel() for param in group.params)
        raw_numel += group_numel
        for attr in ("model_weight_buffer", "main_grad_buffer", "main_weight_buffer"):
            dbuf = getattr(group, attr, None)
            if dbuf is None:
                continue
            if attr == "model_weight_buffer":
                weight_bucket_numel += dbuf.bucket_index.size
            elif attr == "main_grad_buffer":
                grad_bucket_numel += dbuf.bucket_index.size
            elif attr == "main_weight_buffer":
                main_weight_bucket_numel += dbuf.bucket_index.size
            padded_numel += dbuf.bucket_index.size - group_numel

    return GroveLayoutStats(
        groups=len(buffer.parameter_groups),
        raw_numel=raw_numel,
        weight_bucket_numel=weight_bucket_numel,
        grad_bucket_numel=grad_bucket_numel,
        main_weight_bucket_numel=main_weight_bucket_numel,
        padded_numel=padded_numel,
        weight_workspace_allocations=getattr(buffer.weight_alloc, "num_workspace_allocations", -1),
        weight_workspace_peak_live=getattr(buffer.weight_alloc, "peak_live_workspaces", -1),
        transpose_workspace_allocations=getattr(
            buffer.transpose_weight_alloc, "num_workspace_allocations", -1
        ),
        transpose_workspace_peak_live=getattr(
            buffer.transpose_weight_alloc, "peak_live_workspaces", -1
        ),
    )


def synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }


def forward_backward(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    return loss.detach()


def maybe_profile(profile_dir: str | None, backend: str, rank: int, active_steps: int):
    if profile_dir is None:
        return nullcontext()

    os.makedirs(profile_dir, exist_ok=True)
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=active_steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            os.path.join(profile_dir, f"{backend}_rank{rank}")
        ),
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    )


def max_across_ranks(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def synchronize_backend(model: torch.nn.Module | None, device: torch.device) -> None:
    if model is not None and hasattr(model, "finish_grad_sync"):
        try:
            model.finish_grad_sync()
        except Exception as exc:  # noqa: BLE001
            rank_print(f"[cleanup] finish_grad_sync skipped: {type(exc).__name__}: {exc}")
    if model is not None and hasattr(model, "synchronize_param_gather"):
        try:
            model.synchronize_param_gather()
        except Exception as exc:  # noqa: BLE001
            rank_print(f"[cleanup] synchronize_param_gather skipped: {type(exc).__name__}: {exc}")
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def cleanup_after_backend(device: torch.device) -> None:
    clear_global_memory_buffer()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()


def clear_global_memory_buffer() -> None:
    try:
        from grove_fsdp.utils import get_global_memory_buffer

        get_global_memory_buffer().clear()
    except Exception as exc:  # noqa: BLE001
        rank_print(f"[cleanup] global memory buffer clear skipped: {type(exc).__name__}: {exc}")


def profile_backend(
    backend: str,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> Result:
    dtype = dtype_from_arg(args.dtype)
    model = None
    batch = None
    clear_global_memory_buffer()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()

    rank_print(f"\n[{backend}] building Qwen2.5-1.5B ({args.dtype})")
    model = build_model(args, dtype, device)
    vocab_size = model.config.vocab_size
    if backend == "grove":
        model = apply_grove_fsdp(model, args, dtype, device, world_size)
        if args.print_grove_layout:
            stats = grove_layout_stats(model)
            if stats is not None:
                rank_print("[grove] layout", json.dumps(asdict(stats), sort_keys=True))
    elif backend == "torch-fsdp2":
        model = apply_torch_fsdp2(model, dtype, world_size)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    batch = synthetic_batch(
        args.batch_size,
        args.seq_len,
        vocab_size,
        device,
    )
    dist.barrier()
    torch.cuda.synchronize(device)

    for _ in range(args.warmup_steps):
        forward_backward(model, batch)
    torch.cuda.synchronize(device)
    dist.barrier()

    rank_print(f"[{backend}] timing {args.steps} forward+backward iterations")
    torch.cuda.reset_peak_memory_stats(device)
    with maybe_profile(args.profile_dir, backend, rank, args.profile_steps) as prof:
        start = time.perf_counter()
        for _ in range(args.steps):
            forward_backward(model, batch)
            if prof is not None and hasattr(prof, "step"):
                prof.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    elapsed = max_across_ranks(elapsed, device)
    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3
    peak_allocated_gib = max_across_ranks(peak_allocated_gib, device)
    peak_reserved_gib = max_across_ranks(peak_reserved_gib, device)
    tokens_per_iter = args.batch_size * args.seq_len * world_size
    result = Result(
        backend=backend,
        world_size=world_size,
        dtype=args.dtype,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        elapsed_s=elapsed,
        ms_per_iter=elapsed * 1000.0 / args.steps,
        tokens_per_s=tokens_per_iter * args.steps / elapsed,
        peak_allocated_gib=peak_allocated_gib,
        peak_reserved_gib=peak_reserved_gib,
    )

    synchronize_backend(model, device)
    del model
    del batch
    model = None
    batch = None
    cleanup_after_backend(device)
    return result


def print_results(results: Iterable[Result]) -> None:
    rows = list(results)
    if not is_rank0():
        return

    print("\nSummary", flush=True)
    for result in rows:
        print(json.dumps(asdict(result), sort_keys=True), flush=True)

    if len(rows) == 2:
        baseline, candidate = rows
        candidate_speedup = baseline.ms_per_iter / candidate.ms_per_iter
        allocated_delta = candidate.peak_allocated_gib - baseline.peak_allocated_gib
        reserved_delta = candidate.peak_reserved_gib - baseline.peak_reserved_gib
        print(
            f"\n{candidate.backend} vs {baseline.backend}: "
            f"{candidate_speedup:.3f}x by ms/iter, "
            f"allocated_delta={allocated_delta:+.3f} GiB, "
            f"reserved_delta={reserved_delta:+.3f} GiB",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    rank, world_size, device = init_distributed(args.local_rank)
    try:
        backends = (
            ("grove", "torch-fsdp2")
            if args.backend == "both"
            else (args.backend,)
        )
        results = [
            profile_backend(backend, args, rank, world_size, device)
            for backend in backends
        ]
        print_results(results)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
