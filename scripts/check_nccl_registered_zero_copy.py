#!/usr/bin/env python3
"""Check Grove-FSDP NCCL registered-memory all-gather/reduce-scatter zero-copy paths."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "veScale"):
    sys.path.insert(0, str(path))

from grove_fsdp import fully_shard_model  # noqa: E402
from grove_fsdp.dbuffer import DistributedBuffer  # noqa: E402
from grove_fsdp.param_and_grad_buffer import Bucket, DataParallelBuffer  # noqa: E402


def ptr(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    return (
        hex(tensor.untyped_storage().data_ptr()),
        hex(tensor.data_ptr()),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor._base is not None,
    )


def _same_storage_and_data_ptr(base: torch.Tensor, view: torch.Tensor) -> bool:
    expected_data_ptr = base.data_ptr() + view.storage_offset() * view.element_size()
    return (
        base.untyped_storage().data_ptr() == view.untyped_storage().data_ptr()
        and view.data_ptr() == expected_data_ptr
    )


def _assert_zero_copy(
    name: str,
    base: torch.Tensor,
    view: torch.Tensor,
    require_torch_view: bool = True,
) -> None:
    if not _same_storage_and_data_ptr(base, view):
        raise AssertionError(f"{name} is not a zero-copy view: base={ptr(base)} view={ptr(view)}")
    if require_torch_view and view._base is None:
        raise AssertionError(f"{name} is not recorded by PyTorch as a view: view={ptr(view)}")


class TinyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(8, 4))
        torch.nn.init.uniform_(self.weight, -0.1, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t()


def _install_zero_copy_checks(rank: int):
    state = {
        "buffer": None,
        "all_gather_checks": 0,
        "reduce_scatter_checks": 0,
    }
    original_fetch_bucket = DataParallelBuffer.fetch_bucket
    original_all_gather_into = DistributedBuffer.all_gather_into
    original_reduce_scatter_from = DistributedBuffer.reduce_scatter_from

    def print_ptrs(prefix: str, **tensors: Optional[torch.Tensor]) -> None:
        print(f"[rank {rank}] {prefix}", flush=True)
        for name, tensor in tensors.items():
            print(f"[rank {rank}]   {name} {ptr(tensor)}", flush=True)

    def checked_fetch_bucket(self, dtype=None, set_param_data: bool = False):
        bucket = original_fetch_bucket(self, dtype=dtype, set_param_data=set_param_data)
        if set_param_data:
            for param in self.params:
                item_id = self.param_idx[param]
                local_param = param.to_local() if hasattr(param, "to_local") else param
                full_bucket_slice = self.get_item_from_bucket(bucket, item_id).view(
                    local_param.shape
                )
                _assert_zero_copy(
                    "param.data -> full_bucket_slice",
                    bucket.data,
                    local_param.data,
                    require_torch_view=False,
                )
                _assert_zero_copy(
                    "full_bucket_slice -> all-gather bucket",
                    bucket.data,
                    full_bucket_slice,
                )
                if local_param.data.data_ptr() != full_bucket_slice.data_ptr():
                    raise AssertionError(
                        "param.data does not point at the exact full-bucket item slice: "
                        f"param={ptr(local_param.data)} full_bucket_slice={ptr(full_bucket_slice)}"
                    )
                print_ptrs(
                    "all_gather param binding",
                    param=local_param.data,
                    full_bucket_slice=full_bucket_slice,
                    local_shard=self.data,
                )
        return bucket

    def checked_all_gather_into(self, full_bucket, group=None, async_op: bool = False):
        local_shard = self.local_tensor
        full_bucket_shard = self.shard_bucket_view(full_bucket)
        _assert_zero_copy("all-gather output shard -> full bucket", full_bucket, full_bucket_shard)
        print_ptrs(
            "all_gather collective tensors",
            full_bucket=full_bucket,
            full_bucket_slice=full_bucket_shard,
            local_shard=local_shard,
        )
        state["all_gather_checks"] += 1
        return original_all_gather_into(self, full_bucket, group=group, async_op=async_op)

    def checked_reduce_scatter_from(
        self,
        full_bucket,
        group=None,
        op=dist.ReduceOp.SUM,
        async_op: bool = False,
    ):
        local_shard = self.local_tensor
        reduce_scatter_input_slice = self.shard_bucket_view(full_bucket)
        _assert_zero_copy(
            "reduce-scatter input shard -> full bucket",
            full_bucket,
            reduce_scatter_input_slice,
        )

        module_weight = None
        full_bucket_slice = None
        grad = None
        param_buffer = state["buffer"]
        if param_buffer is not None:
            for group_state in param_buffer.parameter_groups:
                gbuf = group_state.hfsdp_helper_gbuf or group_state.main_grad_buffer
                if gbuf is None or gbuf.dbuffer is not self:
                    continue
                bucket = Bucket(data=full_bucket)
                for param in group_state.params:
                    local_param = param.to_local() if hasattr(param, "to_local") else param
                    item_id = gbuf.param_idx[param]
                    full_bucket_slice = gbuf.get_item_from_bucket(bucket, item_id).view(
                        local_param.shape
                    )
                    grad = getattr(local_param, "main_grad", None)
                    if grad is not None:
                        _assert_zero_copy(
                            "main_grad -> reduce-scatter bucket",
                            full_bucket,
                            grad,
                            require_torch_view=False,
                        )
                        if grad.data_ptr() != full_bucket_slice.data_ptr():
                            raise AssertionError(
                                "main_grad does not point at the exact reduce-scatter input item "
                                f"slice: grad={ptr(grad)} full_bucket_slice={ptr(full_bucket_slice)}"
                            )
                    module_weight = local_param.data
                    break
                break

        print_ptrs(
            "reduce_scatter collective tensors",
            param=module_weight,
            full_bucket_slice=full_bucket_slice,
            local_shard=local_shard,
            rs_input_slice=reduce_scatter_input_slice,
            grad=grad,
        )
        state["reduce_scatter_checks"] += 1
        return original_reduce_scatter_from(
            self,
            full_bucket,
            group=group,
            op=op,
            async_op=async_op,
        )

    DataParallelBuffer.fetch_bucket = checked_fetch_bucket
    DistributedBuffer.all_gather_into = checked_all_gather_into
    DistributedBuffer.reduce_scatter_from = checked_reduce_scatter_from
    return state


def _worker(rank: int, world_size: int, init_file: str, symmetric: bool) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        checks = _install_zero_copy_checks(rank)
        torch.manual_seed(1234)
        module = TinyModule().to(device)
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        fsdp_module = fully_shard_model(
            module,
            device_mesh=mesh,
            dp_shard_dim="fsdp",
            tp_dim=None,
            zero_dp_strategy="optim_grads_params",
            fsdp_unit_modules=[TinyModule],
            device=device,
            nccl_ub=True,
            disable_symmetric_registration=not symmetric,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            sync_model_each_microbatch=True,
            disable_bucketing=True,
            preproc_state_dict_for_dcp_ckpt=False,
        )
        checks["buffer"] = fsdp_module.param_and_grad_buffer
        for bucket_id in range(fsdp_module.all_gather_pipeline.num_buckets):
            fsdp_module.all_gather_pipeline.release_bucket(bucket_id, bwd=False)
        original_params = [
            param
            for group in fsdp_module.param_and_grad_buffer.parameter_groups
            for param in group.params
        ]
        fsdp_module.all_gather_and_wait_parameters_ready(
            original_params,
            prefetch=False,
            wait_bucket_ready=True,
            bwd=False,
        )
        torch.cuda.synchronize(device)

        x = torch.randn(4, 4, device=device)
        loss = fsdp_module(x).square().mean()
        loss.backward()
        fsdp_module.finish_grad_sync()
        torch.cuda.synchronize(device)

        if checks["all_gather_checks"] == 0:
            raise AssertionError("No all-gather path was checked")
        if checks["reduce_scatter_checks"] == 0:
            raise AssertionError("No reduce-scatter path was checked")
        print(
            f"[rank {rank}] OK: checked {checks['all_gather_checks']} all-gather and "
            f"{checks['reduce_scatter_checks']} reduce-scatter NCCL zero-copy paths.",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Use symmetric NCCL memory registration instead of conventional registration.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
        print(
            "SKIP: NCCL registered-memory zero-copy check requires "
            f"{args.world_size} CUDA devices.",
            file=sys.stderr,
        )
        return 0
    if not dist.is_nccl_available():
        print("SKIP: PyTorch was not built with NCCL.", file=sys.stderr)
        return 0

    fd, init_file = tempfile.mkstemp(prefix="grove_nccl_zero_copy_")
    os.close(fd)
    try:
        mp.spawn(
            _worker,
            args=(args.world_size, init_file, args.symmetric),
            nprocs=args.world_size,
            join=True,
        )
    finally:
        try:
            os.remove(init_file)
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
