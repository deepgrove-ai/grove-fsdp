import math

import torch

from grove_fsdp.dbuffer import _check_valid_shard_size, plan_dbuffer_layout


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
