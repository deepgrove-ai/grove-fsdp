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
import glob
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
    step_cuda_ms: list[float]
    step_enqueue_ms: list[float]


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
        "--wandb",
        action="store_true",
        help="Log results, step times, and profiler traces to Weights & Biases from rank 0.",
    )
    parser.add_argument(
        "--wandb-entity",
        default="placenta",
        help="Weights & Biases entity for --wandb.",
    )
    parser.add_argument(
        "--wandb-project",
        default="vescale",
        help="Weights & Biases project for --wandb.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb-mode",
        default=None,
        choices=("online", "offline", "disabled"),
        help="Optional W&B mode override.",
    )
    parser.add_argument(
        "--wandb-upload-all-ranks",
        action="store_true",
        help="Upload profiler traces from all ranks. By default only rank 0 traces are uploaded.",
    )
    parser.add_argument(
        "--grove-overlap",
        action="store_true",
        help="Deprecated compatibility flag. Grove overlap is enabled by default.",
    )
    parser.add_argument(
        "--no-grove-overlap",
        action="store_true",
        help="Disable Grove overlap_grad_reduce and overlap_param_gather.",
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
        default=2,
        help=(
            "Initial Grove DBuffer workspace pool size for reusable full-bucket "
            "communication buffers. 0 uses storage-resize allocation."
        ),
    )
    parser.add_argument(
        "--no-grove-release-non-fsdp-unit-params",
        action="store_true",
        help=(
            "Disable releasing Grove-owned shallow parameters on non-FSDP-unit modules."
        ),
    )
    parser.add_argument(
        "--grove-release-non-fsdp-unit-params",
        action="store_true",
        help=(
            "Deprecated compatibility flag. This optimization is enabled by default."
        ),
    )
    parser.add_argument(
        "--no-grove-coalesce-all-gather",
        action="store_true",
        help=(
            "Disable torch.distributed coalesced all-gather for Grove parameter buckets."
        ),
    )
    parser.add_argument(
        "--no-grove-inplace-reduce-scatter",
        action="store_true",
        help=(
            "Disable Grove's DBuffer in-place reduce-scatter path. The profiler "
            "enables it by default because it zeros Grove grad buffers every step."
        ),
    )
    parser.add_argument(
        "--grove-nccl-registered-memory",
        action="store_true",
        help=(
            "Enable NCCL registered user-buffer allocation for Grove without "
            "requesting symmetric-memory registration."
        ),
    )
    parser.add_argument(
        "--grove-nccl-symmetric-memory",
        action="store_true",
        help=(
            "Enable NCCL registered symmetric-memory buffers for Grove via the "
            "Megatron-FSDP user-buffer path."
        ),
    )
    args = parser.parse_args()
    if args.grove_nccl_registered_memory and args.grove_nccl_symmetric_memory:
        parser.error(
            "--grove-nccl-registered-memory and --grove-nccl-symmetric-memory "
            "are mutually exclusive."
        )
    return args


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
        hidden_size=2048,
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
    use_nccl_registered_memory = (
        args.grove_nccl_registered_memory or args.grove_nccl_symmetric_memory
    )
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
    mp_policy = MixedPrecisionPolicy(
        main_params_dtype=None,
        main_grads_dtype=dtype,
        grad_comm_dtype=dtype,
    )
    return fully_shard_model(
        module=model,
        device_mesh=mesh,
        dp_shard_dim="fsdp",
        tp_dim=None,
        zero_dp_strategy="optim_grads_params",
        fsdp_unit_modules=[Qwen2DecoderLayer],
        device=device,
        mixed_precision_policy=mp_policy,
        overlap_grad_reduce=args.grove_overlap or not args.no_grove_overlap,
        overlap_param_gather=args.grove_overlap or not args.no_grove_overlap,
        sync_model_each_microbatch=True,
        disable_bucketing=args.disable_bucketing,
        preproc_state_dict_for_dcp_ckpt=False,
        nccl_ub=use_nccl_registered_memory,
        disable_symmetric_registration=not args.grove_nccl_symmetric_memory,
        grove_fsdp_release_non_fsdp_unit_params=not args.no_grove_release_non_fsdp_unit_params,
        grove_fsdp_coalesce_all_gather=not args.no_grove_coalesce_all_gather,
        grove_fsdp_inplace_reduce_scatter=not args.no_grove_inplace_reduce_scatter,
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
    if hasattr(model, "zero_grad_buffer"):
        model.zero_grad_buffer()
    if hasattr(model, "start_param_sync"):
        model.start_param_sync()
    loss = model(**batch).loss
    loss.backward()
    return loss.detach()


def backend_trace_dir(profile_dir: str, backend: str, rank: int) -> str:
    return os.path.join(profile_dir, backend, f"rank{rank}")


def maybe_profile(args: argparse.Namespace, backend: str, rank: int):
    profile_dir = args.profile_dir
    if profile_dir is None:
        return nullcontext()

    trace_dir = backend_trace_dir(profile_dir, backend, rank)
    os.makedirs(trace_dir, exist_ok=True)
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=args.profile_steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    )


def max_across_ranks(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def max_list_across_ranks(values: list[float], device: torch.device) -> list[float]:
    if not values:
        return []
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return [float(value) for value in tensor.cpu().tolist()]


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


def ensure_profile_dir(args: argparse.Namespace, rank: int) -> None:
    if not args.wandb or args.profile_dir is not None:
        return

    profile_dir = (
        os.path.join("profiles", f"qwen25_1p5b_{int(time.time())}") if rank == 0 else None
    )
    payload = [profile_dir]
    dist.broadcast_object_list(payload, src=0)
    args.profile_dir = payload[0]


def validate_wandb_available(args: argparse.Namespace, device: torch.device) -> None:
    if not args.wandb:
        return

    try:
        import wandb
        wandb_available = True
    except ImportError:
        wandb_available = False

    available_tensor = torch.tensor([int(wandb_available)], dtype=torch.int32, device=device)
    dist.all_reduce(available_tensor, op=dist.ReduceOp.MIN)
    if int(available_tensor.item()) == 0:
        raise RuntimeError(
            "W&B logging requested with --wandb, but wandb is not installed."
        )


def init_wandb(args: argparse.Namespace, world_size: int):
    if not args.wandb or not is_rank0():
        return None

    import wandb

    settings = {}
    if args.wandb_mode is not None:
        settings["mode"] = args.wandb_mode

    config = vars(args).copy()
    config["world_size"] = world_size
    config["torch_version"] = torch.__version__
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config,
        **settings,
    )


def trace_files_for_upload(profile_dir: str, upload_all_ranks: bool) -> list[str]:
    patterns = (
        [os.path.join(profile_dir, "*", "rank*", "*")]
        if upload_all_ranks
        else [os.path.join(profile_dir, "*", "rank0", "*")]
    )
    files = []
    for pattern in patterns:
        files.extend(path for path in glob.glob(pattern) if os.path.isfile(path))
    return sorted(files)


def log_wandb_results(run: Any, args: argparse.Namespace, results: list[Result]) -> None:
    if run is None or not is_rank0():
        return

    import wandb

    rows = []
    wandb_step = 0
    for result in results:
        result_dict = asdict(result)
        step_cuda_ms = result_dict.pop("step_cuda_ms")
        step_enqueue_ms = result_dict.pop("step_enqueue_ms")
        run.log({f"{result.backend}/summary/{key}": value for key, value in result_dict.items()})
        for step_idx, (cuda_ms, enqueue_ms) in enumerate(zip(step_cuda_ms, step_enqueue_ms)):
            run.log(
                {
                    "backend": result.backend,
                    f"{result.backend}/step_cuda_ms": cuda_ms,
                    f"{result.backend}/step_enqueue_ms": enqueue_ms,
                    f"{result.backend}/step": step_idx,
                },
                step=wandb_step,
            )
            wandb_step += 1
            rows.append([result.backend, step_idx, cuda_ms, enqueue_ms])

    if rows:
        table = wandb.Table(
            columns=["backend", "step", "cuda_ms", "enqueue_ms"],
            data=rows,
        )
        run.log({"step_times": table})

    if args.profile_dir is not None:
        trace_files = trace_files_for_upload(
            args.profile_dir,
            upload_all_ranks=args.wandb_upload_all_ranks,
        )
        if trace_files:
            artifact = wandb.Artifact(
                name=f"qwen25-1p5b-{run.id}-torch-profiler-traces",
                type="torch-profiler-trace",
                metadata={
                    "profile_dir": args.profile_dir,
                    "upload_all_ranks": args.wandb_upload_all_ranks,
                    "profile_steps": args.profile_steps,
                },
            )
            for trace_file in trace_files:
                artifact.add_file(
                    trace_file,
                    name=os.path.relpath(trace_file, args.profile_dir),
                )
            run.log_artifact(artifact)
            run.log({"trace_file_count": len(trace_files)})
        else:
            rank_print(f"[wandb] no profiler trace files found under {args.profile_dir}")


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
    step_enqueue_ms = []
    step_start_events = []
    step_end_events = []
    with maybe_profile(args, backend, rank) as prof:
        start = time.perf_counter()
        for step_idx in range(args.steps):
            step_start_event = torch.cuda.Event(enable_timing=True)
            step_end_event = torch.cuda.Event(enable_timing=True)
            step_start_events.append(step_start_event)
            step_end_events.append(step_end_event)
            step_enqueue_start = time.perf_counter()
            step_start_event.record()
            with torch.profiler.record_function(f"{backend}_forward_backward_step_{step_idx}"):
                forward_backward(model, batch)
            step_end_event.record()
            step_enqueue_ms.append((time.perf_counter() - step_enqueue_start) * 1000.0)
            if prof is not None and hasattr(prof, "step"):
                prof.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    step_cuda_ms = [
        start_event.elapsed_time(end_event)
        for start_event, end_event in zip(step_start_events, step_end_events)
    ]
    elapsed = max_across_ranks(elapsed, device)
    step_cuda_ms = max_list_across_ranks(step_cuda_ms, device)
    step_enqueue_ms = max_list_across_ranks(step_enqueue_ms, device)
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
        step_cuda_ms=step_cuda_ms,
        step_enqueue_ms=step_enqueue_ms,
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
        result_dict = asdict(result)
        result_dict["step_cuda_ms_mean"] = (
            sum(result.step_cuda_ms) / len(result.step_cuda_ms) if result.step_cuda_ms else 0.0
        )
        result_dict["step_enqueue_ms_mean"] = (
            sum(result.step_enqueue_ms) / len(result.step_enqueue_ms)
            if result.step_enqueue_ms
            else 0.0
        )
        result_dict.pop("step_cuda_ms")
        result_dict.pop("step_enqueue_ms")
        print(json.dumps(result_dict, sort_keys=True), flush=True)

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
    ensure_profile_dir(args, rank)
    wandb_run = None
    try:
        validate_wandb_available(args, device)
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
        dist.barrier()
        wandb_run = init_wandb(args, world_size)
        log_wandb_results(wandb_run, args, results)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
