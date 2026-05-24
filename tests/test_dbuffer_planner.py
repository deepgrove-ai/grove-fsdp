import math

import torch

from grove_fsdp.dbuffer import (
    DBUFFER_REPLICATE,
    DBUFFER_SHARD,
    DBufferDeviceTopology,
    DBufferShardingSpec,
    DistributedBuffer,
    _check_valid_shard_size,
    plan_dbuffer_layout,
)
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
        assert sum(item.local_units) * item.block_size == item.size
        assert len(item.local_units) == world_size
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


def test_default_block_size_is_elementwise() -> None:
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
    assert [item.block_size for item in plan.items] == [1, 1]
    assert plan.shard.size == 704


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
    assert [item.block_size for item in plan.items] == [1, 1]
    assert plan.shard.size % 2048 == 0


def test_planner_exposes_tensor_order_heuristics() -> None:
    shapes = [torch.Size([12]), torch.Size([16]), torch.Size([8])]
    plan = plan_dbuffer_layout(
        elements=shapes,
        data_parallel_rank=0,
        data_parallel_world_size=2,
        is_data_distributed=True,
        bucket_id=0,
        chunk_size_factor=1,
        pad_bucket=True,
        item_block_sizes=[3, 8, 4],
        tensor_order="block_size",
    )

    _assert_ragged_constraints(plan, world_size=2)
    assert plan.tensor_order == (0, 2, 1)


def test_planner_materializes_per_item_ragged_shard_units() -> None:
    plan = plan_dbuffer_layout(
        elements=[torch.Size([6, 4])],
        data_parallel_rank=0,
        data_parallel_world_size=3,
        is_data_distributed=True,
        bucket_id=0,
        chunk_size_factor=1,
        pad_bucket=True,
        item_block_sizes=[4],
    )

    item = plan.items[0]
    assert item.local_units == (2, 2, 2)
    assert item.ragged_dims == (0,)
    assert plan.item_ragged_shard(0).local_units == (2, 2, 2)


def test_dbuffer_records_n_dimensional_topology_and_sharding_spec() -> None:
    plan = plan_dbuffer_layout(
        elements=[torch.Size([8]), torch.Size([8])],
        data_parallel_rank=0,
        data_parallel_world_size=4,
        is_data_distributed=True,
        bucket_id=0,
        chunk_size_factor=1,
        pad_bucket=True,
        item_block_sizes=[2, 2],
        topology=DBufferDeviceTopology(mesh_shape=(2, 2), coordinate=(1, 0)),
        sharding_spec=DBufferShardingSpec(placements=(DBUFFER_REPLICATE, DBUFFER_SHARD)),
    )

    assert plan.topology.mesh_shape == (2, 2)
    assert plan.sharding_spec.placements == (DBUFFER_REPLICATE, DBUFFER_SHARD)
    assert plan.shard_world_size == 2
    assert plan.shard_coordinate == 0
    assert plan.shard.global_data_index == 0
    assert plan.bucket_size == plan.shard.size * plan.shard_world_size


def test_dbuffer_uses_sharded_topology_size_for_replicated_mesh_dims() -> None:
    plan = plan_dbuffer_layout(
        elements=[torch.Size([8]), torch.Size([8])],
        data_parallel_rank=0,
        data_parallel_world_size=4,
        is_data_distributed=True,
        bucket_id=0,
        chunk_size_factor=1,
        pad_bucket=True,
        item_block_sizes=[2, 2],
        topology=DBufferDeviceTopology(mesh_shape=(2, 2), coordinate=(0, 1)),
        sharding_spec=DBufferShardingSpec(placements=(DBUFFER_REPLICATE, DBUFFER_SHARD)),
    )

    assert plan.shard_world_size == 2
    assert plan.shard_coordinate == 1
    assert plan.bucket_size == 16
    assert plan.shard.size == 8
    assert plan.shard.global_data_index == 8


def test_ragged_dims_preserve_shape_for_multi_row_blocks() -> None:
    plan = plan_dbuffer_layout(
        elements=[torch.Size([8, 4])],
        data_parallel_rank=0,
        data_parallel_world_size=2,
        is_data_distributed=True,
        bucket_id=0,
        chunk_size_factor=1,
        pad_bucket=True,
        item_block_sizes=[16],
    )

    item = plan.items[0]
    assert item.block_size == 16
    assert item.ragged_dims == (0,)
    reconstructed = item.ragged_shard.reconstruct_tensor_from_flat(
        torch.empty(16),
        tuple(item.shape),
    )
    assert reconstructed.shape == torch.Size([4, 4])


def test_distributed_buffer_copies_local_item_intersections() -> None:
    plan = plan_dbuffer_layout(
        elements=[torch.Size([12]), torch.Size([12])],
        data_parallel_rank=1,
        data_parallel_world_size=2,
        is_data_distributed=True,
        bucket_id=0,
        chunk_size_factor=1,
        pad_bucket=True,
        item_block_sizes=[3, 3],
    )
    dbuffer = DistributedBuffer(plan, is_data_distributed=True)
    dbuffer.init_data(torch.empty(plan.shard.size))

    tensors = [torch.arange(12, dtype=torch.float32), torch.arange(100, 112, dtype=torch.float32)]
    dbuffer.copy_local_items_from(tensors)

    full_bucket = torch.empty(plan.bucket_size)
    full_bucket.fill_(-1)
    dbuffer.copy_local_to_bucket_(full_bucket)
    for item in plan.items:
        local_and_source = dbuffer.item_local_and_source_intervals(item.item_id)
        if local_and_source is None:
            continue
        _, (src_start, src_end) = local_and_source
        expected = tensors[item.item_id].flatten()[src_start:src_end]
        actual = dbuffer.item_local_view(item.item_id)
        torch.testing.assert_close(actual, expected)


def test_distributed_buffer_grouped_ops() -> None:
    plans = [
        plan_dbuffer_layout(
            elements=[torch.Size([4])],
            data_parallel_rank=0,
            data_parallel_world_size=1,
            is_data_distributed=False,
            bucket_id=bucket_id,
            chunk_size_factor=1,
            pad_bucket=True,
        )
        for bucket_id in range(2)
    ]
    buffers = [DistributedBuffer(plan, is_data_distributed=False) for plan in plans]
    for buffer in buffers:
        buffer.init_data(torch.ones(buffer.local_numel))

    DistributedBuffer.grouped_scale_(buffers, 3.0)
    for buffer in buffers:
        torch.testing.assert_close(buffer.local_tensor, torch.full_like(buffer.local_tensor, 3.0))

    DistributedBuffer.grouped_add_(buffers, 2.0)
    for buffer in buffers:
        torch.testing.assert_close(buffer.local_tensor, torch.full_like(buffer.local_tensor, 5.0))

    copies = [torch.empty_like(buffer.local_tensor) for buffer in buffers]
    DistributedBuffer.grouped_copy_(
        [(copy, buffer.local_tensor) for copy, buffer in zip(copies, buffers)]
    )
    for copy in copies:
        torch.testing.assert_close(copy, torch.full_like(copy, 5.0))

    DistributedBuffer.grouped_add_tensors_(
        [(buffer.local_tensor, torch.ones_like(buffer.local_tensor)) for buffer in buffers]
    )
    for buffer in buffers:
        torch.testing.assert_close(buffer.local_tensor, torch.full_like(buffer.local_tensor, 6.0))

    DistributedBuffer.grouped_zero_(buffers)
    for buffer in buffers:
        torch.testing.assert_close(buffer.local_tensor, torch.zeros_like(buffer.local_tensor))


def test_distributed_buffer_exposes_zero_copy_bucket_item_views() -> None:
    plan = plan_dbuffer_layout(
        elements=[torch.Size([2, 2]), torch.Size([4])],
        data_parallel_rank=0,
        data_parallel_world_size=1,
        is_data_distributed=False,
        bucket_id=0,
        chunk_size_factor=1,
        pad_bucket=True,
    )
    dbuffer = DistributedBuffer(plan, is_data_distributed=False)
    bucket = torch.arange(plan.bucket_size, dtype=torch.float32)
    views = dbuffer.bucket_item_views(bucket)

    assert views[0]._base is not None
    assert views[1]._base is not None
    views[0].fill_(9)
    torch.testing.assert_close(bucket[:4], torch.full((4,), 9.0))


def test_qwen_like_elementwise_default_minimizes_padding() -> None:
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
    assert [item.block_size for item in plan.items] == [1] * len(shapes)
    assert plan.padding == 0
    assert chunk_aligned_plan.padding == 13_312
    assert plan.padding < chunk_aligned_plan.padding


def test_qwen_like_rowwise_layout_still_available_when_requested() -> None:
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
    rowwise_block_size = lambda shape: max(1, shape[1:].numel()) if len(shape) > 1 else 1

    plan = plan_dbuffer_layout(
        elements=shapes,
        data_parallel_rank=0,
        data_parallel_world_size=8,
        is_data_distributed=True,
        bucket_id=0,
        chunk_size_factor=2048,
        pad_bucket=True,
        item_block_size_fn=rowwise_block_size,
    )
    chunk_aligned_plan = plan_dbuffer_layout(
        elements=shapes,
        data_parallel_rank=0,
        data_parallel_world_size=8,
        is_data_distributed=True,
        bucket_id=0,
        chunk_size_factor=2048,
        pad_bucket=True,
        item_block_size_fn=rowwise_block_size,
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


def test_config_defaults_to_dbuffer_workspace_pool() -> None:
    assert DistributedDataParallelConfig().grove_fsdp_dbuffer_workspace_size == 2


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
