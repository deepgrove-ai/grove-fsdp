import math

import torch

from grove_fsdp.dbuffer import _check_valid_shard_size, plan_dbuffer_layout
from grove_fsdp.distributed_data_parallel_config import DistributedDataParallelConfig
from grove_fsdp.param_and_grad_buffer import (
    DBufferWorkspaceAllocator,
    _get_ragged_shard_block_size_fn,
    _get_ragged_shard_param_block_sizes,
)


def _assert_ragged_constraints(plan, world_size: int) -> None:
    shard_size = plan.bucket_size // world_size
    assert plan.bucket_size == shard_size * world_size

    intervals = []
    for item in plan.items:
        start = item.global_data_index
        end = start + item.size
        intervals.append((start, end))
        for boundary in range(shard_size, plan.bucket_size, shard_size):
            if start < boundary < end:
                assert (boundary - start) % item.block_size == 0

    for idx, (start, end) in enumerate(intervals):
        for other_start, other_end in intervals[idx + 1 :]:
            assert end <= other_start or other_end <= start


def _brute_force_min_shard_size(items, world_size: int, collective_unit_size: int) -> int:
    total_size = sum(shape.numel() for _, shape, _ in items)
    shard_size = math.ceil(total_size / world_size)
    while True:
        if shard_size % collective_unit_size == 0:
            if _check_valid_shard_size(items, shard_size, world_size) is not None:
                return shard_size
        shard_size += 1


def test_algorithm1_planner_finds_minimal_fixed_order_layout() -> None:
    shapes = [torch.Size([5, 4]), torch.Size([3, 2]), torch.Size([7])]
    world_size = 3
    collective_unit_size = 1
    items = [(idx, shape, 4 if idx == 0 else 1) for idx, shape in enumerate(shapes)]

    plan = plan_dbuffer_layout(
        elements=shapes,
        data_parallel_rank=1,
        data_parallel_world_size=world_size,
        is_data_distributed=True,
        bucket_id=0,
        chunk_size_factor=collective_unit_size,
        pad_bucket=True,
        item_block_size_fn=lambda shape: 4 if shape == torch.Size([5, 4]) else 1,
    )

    _assert_ragged_constraints(plan, world_size)
    assert plan.shard.size == _brute_force_min_shard_size(
        items, world_size, collective_unit_size
    )
    assert plan.shard.global_data_index == plan.shard.size


def test_algorithm1_planner_respects_requested_block_size() -> None:
    plan = plan_dbuffer_layout(
        elements=[torch.Size([12]), torch.Size([6]), torch.Size([6])],
        data_parallel_rank=0,
        data_parallel_world_size=2,
        is_data_distributed=True,
        bucket_id=3,
        chunk_size_factor=1,
        pad_bucket=True,
        item_block_size_fn=lambda _shape: 3,
    )

    _assert_ragged_constraints(plan, world_size=2)
    assert [item.block_size for item in plan.items] == [3, 3, 3]
    assert plan.bucket_id == 3


def test_default_block_size_matches_rowwise_granularity() -> None:
    plan = plan_dbuffer_layout(
        elements=[torch.Size([3, 512]), torch.Size([4096])],
        data_parallel_rank=0,
        data_parallel_world_size=8,
        is_data_distributed=True,
        bucket_id=0,
        chunk_size_factor=2048,
        pad_bucket=True,
    )

    _assert_ragged_constraints(plan, world_size=8)
    assert [item.block_size for item in plan.items] == [512, 1]
    assert plan.shard.size == 1024


def test_chunk_size_alignment_is_opt_in() -> None:
    plan = plan_dbuffer_layout(
        elements=[torch.Size([3, 512]), torch.Size([4096])],
        data_parallel_rank=0,
        data_parallel_world_size=8,
        is_data_distributed=True,
        bucket_id=0,
        chunk_size_factor=2048,
        pad_bucket=True,
        align_bucket_to_chunk_size=True,
    )

    _assert_ragged_constraints(plan, world_size=8)
    assert [item.block_size for item in plan.items] == [512, 1]
    assert plan.shard.size % 2048 == 0


def test_qwen_like_rowwise_layout_avoids_chunk_alignment_padding() -> None:
    shapes = [
        torch.Size([1536, 1536]),
        torch.Size([256, 1536]),
        torch.Size([256, 1536]),
        torch.Size([1536, 1536]),
        torch.Size([8960, 1536]),
        torch.Size([8960, 1536]),
        torch.Size([1536, 8960]),
        torch.Size([1536]),
        torch.Size([1536]),
    ]

    plan = plan_dbuffer_layout(
        elements=shapes,
        data_parallel_rank=0,
        data_parallel_world_size=8,
        is_data_distributed=True,
        bucket_id=0,
        chunk_size_factor=2048,
        pad_bucket=True,
    )
    chunk_aligned_plan = plan_dbuffer_layout(
        elements=shapes,
        data_parallel_rank=0,
        data_parallel_world_size=8,
        is_data_distributed=True,
        bucket_id=0,
        chunk_size_factor=2048,
        pad_bucket=True,
        align_bucket_to_chunk_size=True,
    )

    _assert_ragged_constraints(plan, world_size=8)
    assert plan.padding == 82_944
    assert chunk_aligned_plan.padding == 1_373_184
    assert plan.padding < chunk_aligned_plan.padding


def test_config_supports_shape_aware_block_size_function() -> None:
    def block_size_for_shape(shape: torch.Size) -> int:
        return 16 if len(shape) > 1 else 1

    config = DistributedDataParallelConfig(
        grove_fsdp_ragged_shard_block_size=8,
        grove_fsdp_ragged_shard_block_size_fn=block_size_for_shape,
    )
    block_size_fn = _get_ragged_shard_block_size_fn(config)

    plan = plan_dbuffer_layout(
        elements=[torch.Size([2, 32]), torch.Size([16])],
        data_parallel_rank=0,
        data_parallel_world_size=2,
        is_data_distributed=True,
        bucket_id=0,
        chunk_size_factor=1,
        pad_bucket=True,
        item_block_size_fn=block_size_fn,
    )

    _assert_ragged_constraints(plan, world_size=2)
    assert [item.block_size for item in plan.items] == [16, 1]


def test_config_supports_per_parameter_block_sizes() -> None:
    params = [
        torch.nn.Parameter(torch.empty(4, 8)),
        torch.nn.Parameter(torch.empty(4, 8)),
    ]

    def block_size_for_param(param: torch.nn.Parameter) -> int:
        return 16 if param is params[0] else 4

    config = DistributedDataParallelConfig(
        grove_fsdp_ragged_shard_block_size=8,
        grove_fsdp_ragged_shard_block_size_fn=lambda _shape: 8,
        grove_fsdp_ragged_shard_param_block_size_fn=block_size_for_param,
    )
    item_block_sizes = _get_ragged_shard_param_block_sizes(config, params)

    plan = plan_dbuffer_layout(
        elements=[param.shape for param in params],
        data_parallel_rank=0,
        data_parallel_world_size=2,
        is_data_distributed=True,
        bucket_id=0,
        chunk_size_factor=1,
        pad_bucket=True,
        item_block_sizes=item_block_sizes,
    )

    _assert_ragged_constraints(plan, world_size=2)
    assert [item.block_size for item in plan.items] == [16, 4]


def test_requested_block_size_must_divide_item_size() -> None:
    try:
        plan_dbuffer_layout(
            elements=[torch.Size([10])],
            data_parallel_rank=0,
            data_parallel_world_size=2,
            is_data_distributed=True,
            bucket_id=0,
            chunk_size_factor=1,
            pad_bucket=True,
            item_block_size_fn=lambda _shape: 4,
        )
    except ValueError as exc:
        assert "not divisible by block size" in str(exc)
    else:
        raise AssertionError("Expected invalid RaggedShard block size to raise ValueError")


def test_dbuffer_workspace_allocator_reuses_bounded_workspaces() -> None:
    allocator = DBufferWorkspaceAllocator("test_workspace", max_live_workspaces=1)
    allocator.set_planned_size(16)

    first = allocator.allocate(
        bucket_id=0,
        size=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    allocator.free(0)
    second = allocator.allocate(
        bucket_id=1,
        size=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    allocator.free(1)
    third = allocator.allocate(
        bucket_id=2,
        size=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert first.data.untyped_storage().data_ptr() == second.data.untyped_storage().data_ptr()
    assert third.data.numel() == 16
    assert allocator.num_workspace_allocations == 1
    assert list(allocator.workspace_sizes.values()) == [16]


def test_dbuffer_workspace_allocator_grows_with_live_concurrency() -> None:
    allocator = DBufferWorkspaceAllocator("test_workspace_limit", max_live_workspaces=1)
    allocator.set_planned_size(8)
    allocator.allocate(
        bucket_id=0,
        size=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    allocator.allocate(
        bucket_id=1,
        size=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert allocator.num_workspace_allocations == 2
    assert allocator.peak_live_workspaces == 2


def test_dbuffer_workspace_allocator_rejects_oversized_request() -> None:
    allocator = DBufferWorkspaceAllocator("test_workspace_oversized", max_live_workspaces=1)
    allocator.set_planned_size(8)

    try:
        allocator.allocate(
            bucket_id=0,
            size=16,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    except RuntimeError as exc:
        assert "planned size" in str(exc)
    else:
        raise AssertionError("Expected oversized DBuffer workspace request to fail")
