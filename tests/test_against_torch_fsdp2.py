import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard as torch_fully_shard
from torch.distributed.tensor import Shard

from grove_fsdp import fully_shard_model
from grove_fsdp.dbuffer import plan_dbuffer_layout


class TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 8, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(8, 3, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _assert_ragged_plan_preserves_blocks(plan, world_size: int) -> None:
    shard_size = plan.bucket_size // world_size
    for item in plan.items:
        start = item.global_data_index
        end = start + item.size
        for boundary in range(shard_size, plan.bucket_size, shard_size):
            if start < boundary < end:
                assert (boundary - start) % item.block_size == 0


def test_ragged_planner_preserves_blocks_that_fsdp2_even_shard_splits() -> None:
    tensor = torch.arange(24).reshape(6, 4)
    world_size = 3
    block_size = 6

    fsdp2_chunks, _ = Shard(0)._split_tensor(tensor, world_size)
    fsdp2_boundaries = []
    offset = 0
    for chunk in fsdp2_chunks[:-1]:
        offset += chunk.numel()
        fsdp2_boundaries.append(offset)

    assert any(boundary % block_size != 0 for boundary in fsdp2_boundaries)

    plan = plan_dbuffer_layout(
        elements=[tensor.shape],
        data_parallel_rank=0,
        data_parallel_world_size=world_size,
        is_data_distributed=True,
        bucket_id=0,
        chunk_size_factor=1,
        pad_bucket=True,
        item_block_size_fn=lambda _shape: block_size,
    )

    _assert_ragged_plan_preserves_blocks(plan, world_size)
    assert plan.shard.size == 12
    assert plan.padding == 12


def _run_grove_vs_torch_fsdp2(rank: int, world_size: int, init_file: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cuda", rank)
        torch.manual_seed(1234)
        torch_model = TinyMLP().to(device)
        torch_mesh = init_device_mesh("cuda", (world_size,))
        torch_fully_shard(torch_model, mesh=torch_mesh)

        torch.manual_seed(1234)
        grove_model = TinyMLP().to(device)
        grove_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        grove_model = fully_shard_model(
            grove_model,
            device_mesh=grove_mesh,
            dp_shard_dim="fsdp",
            tp_dim=None,
            zero_dp_strategy="optim_grads_params",
            fsdp_unit_modules=[TinyMLP],
            device=device,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            sync_model_each_microbatch=True,
            disable_bucketing=True,
            preproc_state_dict_for_dcp_ckpt=False,
        )

        torch.manual_seed(2026)
        x = torch.randn(5, 4, device=device)
        torch_x = x.detach().clone().requires_grad_(True)
        grove_x = x.detach().clone().requires_grad_(True)

        torch_loss = torch_model(torch_x).square().mean()
        grove_loss = grove_model(grove_x).square().mean()
        torch.testing.assert_close(grove_loss.detach(), torch_loss.detach(), rtol=1e-5, atol=1e-6)

        torch_loss.backward()
        grove_loss.backward()
        torch.testing.assert_close(grove_x.grad, torch_x.grad, rtol=1e-5, atol=1e-6)
        torch.cuda.synchronize(device)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or not dist.is_nccl_available(),
    reason="Grove-vs-FSDP2 integration comparison requires at least 2 NCCL CUDA devices",
)
def test_grove_forward_backward_matches_torch_fsdp2_cuda() -> None:
    world_size = 2
    fd, init_file = tempfile.mkstemp(prefix="grove_fsdp2_")
    os.close(fd)
    try:
        mp.spawn(
            _run_grove_vs_torch_fsdp2,
            args=(world_size, init_file),
            nprocs=world_size,
            join=True,
        )
    finally:
        try:
            os.remove(init_file)
        except FileNotFoundError:
            pass
