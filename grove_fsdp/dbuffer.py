# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""veScale RaggedShard DBuffer planning and storage helpers for Grove-FSDP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from vescale.dtensor.placement_types import RaggedShard


DBUFFER_SHARD = "shard"
DBUFFER_REPLICATE = "replicate"
DBUFFER_PARTIAL = "partial"


@dataclass(frozen=True)
class DBufferDeviceTopology:
    mesh_shape: Tuple[int, ...]
    coordinate: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.mesh_shape:
            raise ValueError("DBuffer topology must have at least one dimension")
        if len(self.mesh_shape) != len(self.coordinate):
            raise ValueError("DBuffer topology shape and coordinate must have the same rank")
        for dim, (size, coord) in enumerate(zip(self.mesh_shape, self.coordinate)):
            if size <= 0:
                raise ValueError(f"mesh dimension {dim} must be positive")
            if coord < 0 or coord >= size:
                raise ValueError(f"coordinate {coord} is out of range for mesh dimension {dim}")

    @property
    def ndim(self) -> int:
        return len(self.mesh_shape)


@dataclass(frozen=True)
class DBufferShardingSpec:
    placements: Tuple[str, ...]

    def __post_init__(self) -> None:
        valid = {DBUFFER_SHARD, DBUFFER_REPLICATE, DBUFFER_PARTIAL}
        invalid = [placement for placement in self.placements if placement not in valid]
        if invalid:
            raise ValueError(f"Unsupported DBuffer placements: {invalid}")

    def shard_dims(self) -> Tuple[int, ...]:
        return tuple(
            dim for dim, placement in enumerate(self.placements) if placement == DBUFFER_SHARD
        )


@dataclass(frozen=True)
class DBufferItem:
    item_id: int
    global_data_index: int
    size: int
    shape: torch.Size
    block_size: int
    local_units: Tuple[int, ...]
    ragged_dims: Tuple[int, ...]

    @property
    def end(self) -> int:
        return self.global_data_index + self.size

    @property
    def ragged_shard(self) -> RaggedShard:
        return RaggedShard(dims=self.ragged_dims, local_units=self.local_units)


@dataclass(frozen=True)
class DBufferShard:
    bucket_id: int
    global_data_index: int
    local_data_index: int
    bucket_data_index: int
    size: int

    @property
    def end(self) -> int:
        return self.global_data_index + self.size


@dataclass(frozen=True)
class DBufferPlan:
    bucket_id: int
    bucket_size: int
    topology: DBufferDeviceTopology
    sharding_spec: DBufferShardingSpec
    items: Tuple[DBufferItem, ...]
    shard: DBufferShard
    ragged_shard: RaggedShard
    tensor_order: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.topology.ndim != len(self.sharding_spec.placements):
            raise ValueError("DBuffer topology and sharding spec ranks must match")

    @property
    def padding(self) -> int:
        return self.bucket_size - sum(item.size for item in self.items)

    @property
    def shard_world_size(self) -> int:
        shard_dims = self.sharding_spec.shard_dims()
        if not shard_dims:
            return 1
        return math.prod(self.topology.mesh_shape[dim] for dim in shard_dims)

    @property
    def shard_coordinate(self) -> int:
        shard_dims = self.sharding_spec.shard_dims()
        coordinate = 0
        stride = 1
        for dim in reversed(shard_dims):
            coordinate += self.topology.coordinate[dim] * stride
            stride *= self.topology.mesh_shape[dim]
        return coordinate

    def item_ragged_shard(self, item_id: int) -> RaggedShard:
        return self.items[item_id].ragged_shard


class DistributedBuffer:
    def __init__(self, plan: DBufferPlan, is_data_distributed: bool) -> None:
        self.plan = plan
        self.is_data_distributed = is_data_distributed
        self.data: Optional[torch.Tensor] = None
        self._items_by_id = {item.item_id: item for item in plan.items}

    @property
    def local_numel(self) -> int:
        return self.plan.shard.size if self.is_data_distributed else self.plan.bucket_size

    @property
    def local_tensor(self) -> torch.Tensor:
        if self.data is None:
            raise RuntimeError("DBuffer storage has not been initialized")
        return self.data

    def init_data(self, data: torch.Tensor) -> None:
        if data.numel() != self.local_numel:
            raise ValueError(
                f"DBuffer backing tensor has {data.numel()} elements, expected {self.local_numel}"
            )
        self.data = data

    def local_interval(self, global_start: int, global_end: int) -> Optional[Tuple[int, int]]:
        if global_end < global_start:
            raise ValueError("global_end must be >= global_start")
        if not self.is_data_distributed:
            return global_start, global_end
        start = max(global_start, self.plan.shard.global_data_index)
        end = min(global_end, self.plan.shard.end)
        if start >= end:
            return None
        return start - self.plan.shard.global_data_index, end - self.plan.shard.global_data_index

    def item_local_and_source_intervals(
        self, item_id: int
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        item = self._items_by_id[item_id]
        local_interval = self.local_interval(item.global_data_index, item.end)
        if local_interval is None:
            return None
        local_start, local_end = local_interval
        global_start = (
            self.plan.shard.global_data_index + local_start
            if self.is_data_distributed
            else local_start
        )
        item_start = global_start - item.global_data_index
        return local_interval, (item_start, item_start + local_end - local_start)

    def item_local_view(self, item_id: int) -> torch.Tensor:
        item = self._items_by_id[item_id]
        interval = self.local_interval(item.global_data_index, item.end)
        if interval is None:
            return self.local_tensor.new_empty(0)
        start, end = interval
        return self.local_tensor[start:end]

    def item_bucket_view(self, bucket: torch.Tensor, item_id: int) -> torch.Tensor:
        self._check_full_bucket(bucket)
        item = self._items_by_id[item_id]
        return bucket[item.global_data_index : item.end]

    def shard_bucket_view(self, bucket: torch.Tensor) -> torch.Tensor:
        self._check_full_bucket(bucket)
        shard = self.plan.shard
        return bucket[shard.bucket_data_index : shard.bucket_data_index + shard.size]

    def copy_local_items_from(self, tensors: Sequence[torch.Tensor]) -> None:
        self._check_item_tensor_count(tensors)
        for item in self.plan.items:
            flat_src = tensors[item.item_id].detach().reshape(-1)
            intervals = self.item_local_and_source_intervals(item.item_id)
            if intervals is None:
                continue
            (local_start, local_end), (src_start, src_end) = intervals
            self.local_tensor[local_start:local_end].copy_(flat_src[src_start:src_end])

    def copy_local_items_to(self, tensors: Sequence[torch.Tensor]) -> None:
        self._check_item_tensor_count(tensors)
        for item in self.plan.items:
            flat_dst = tensors[item.item_id].detach().reshape(-1)
            intervals = self.item_local_and_source_intervals(item.item_id)
            if intervals is None:
                continue
            (local_start, local_end), (dst_start, dst_end) = intervals
            flat_dst[dst_start:dst_end].copy_(self.local_tensor[local_start:local_end])

    def copy_local_to_bucket_(self, full_bucket: torch.Tensor) -> None:
        self._check_full_bucket(full_bucket)
        if self.is_data_distributed:
            self.shard_bucket_view(full_bucket).copy_(self.local_tensor)
        else:
            full_bucket.copy_(self.local_tensor)

    def bucket_item_views(self, bucket: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(
            self.item_bucket_view(bucket, item.item_id).view(item.shape) for item in self.plan.items
        )

    def all_gather_into(self, full_bucket: torch.Tensor, group=None, async_op: bool = False):
        self._check_full_bucket(full_bucket)
        if not self.is_data_distributed:
            full_bucket.copy_(self.local_tensor)
            return None
        return torch.distributed.all_gather_into_tensor(
            full_bucket, self.local_tensor, group=group, async_op=async_op
        )

    def reduce_scatter_from(
        self,
        full_bucket: torch.Tensor,
        group=None,
        op=torch.distributed.ReduceOp.SUM,
        async_op: bool = False,
    ):
        self._check_full_bucket(full_bucket)
        if not self.is_data_distributed:
            self.local_tensor.copy_(full_bucket)
            return None
        return torch.distributed.reduce_scatter_tensor(
            self.local_tensor, full_bucket, op=op, group=group, async_op=async_op
        )

    @staticmethod
    def grouped_zero_(buffers: Sequence["DistributedBuffer"]) -> None:
        _foreach_by_type([buffer.local_tensor for buffer in buffers], "zero", 0.0)

    @staticmethod
    def grouped_scale_(buffers: Sequence["DistributedBuffer"], value: float) -> None:
        _foreach_by_type([buffer.local_tensor for buffer in buffers], "mul", value)

    @staticmethod
    def grouped_add_(buffers: Sequence["DistributedBuffer"], value: float) -> None:
        _foreach_by_type([buffer.local_tensor for buffer in buffers], "add", value)

    @staticmethod
    def grouped_copy_(dst_src_pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        for dst, src in dst_src_pairs:
            if dst.numel() > 0:
                dst.copy_(src)

    @staticmethod
    def grouped_add_tensors_(dst_src_pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        for dst, src in dst_src_pairs:
            if dst.numel() > 0:
                dst.add_(src)

    def _check_full_bucket(self, bucket: torch.Tensor) -> None:
        if bucket.numel() != self.plan.bucket_size:
            raise ValueError(
                f"full bucket has {bucket.numel()} elements, expected {self.plan.bucket_size}"
            )

    def _check_item_tensor_count(self, tensors: Sequence[torch.Tensor]) -> None:
        if len(tensors) != len(self.plan.items):
            raise ValueError(f"got {len(tensors)} tensors for {len(self.plan.items)} items")


def _foreach_by_type(tensors: Sequence[torch.Tensor], op: str, value: float) -> None:
    for tensor in tensors:
        if tensor.numel() == 0:
            continue
        if op == "zero":
            tensor.zero_()
        elif op == "mul":
            tensor.mul_(value)
        elif op == "add":
            tensor.add_(value)
        else:
            raise ValueError(f"unsupported foreach op {op}")


def _infer_ragged_dims(shape: torch.Size, block_size: int) -> Tuple[int, ...]:
    suffix_numel = shape.numel()
    for prefix_len in range(0, len(shape) + 1):
        if suffix_numel > 0 and block_size % suffix_numel == 0:
            return tuple(range(prefix_len))
        if prefix_len < len(shape):
            dim_size = shape[prefix_len]
            if dim_size == 0:
                break
            suffix_numel //= dim_size
    return tuple(range(len(shape)))


def _item_local_units(
    start: int,
    size: int,
    block_size: int,
    shard_size: int,
    num_devices: int,
) -> Tuple[int, ...]:
    end = start + size
    units = []
    for rank in range(num_devices):
        shard_start = rank * shard_size
        shard_end = shard_start + shard_size
        local_start = max(start, shard_start)
        local_end = min(end, shard_end)
        local_size = max(0, local_end - local_start)
        if local_size % block_size != 0:
            raise ValueError("planned local interval is not divisible by block size")
        units.append(local_size // block_size)
    return tuple(units)


def _validate_planner_items(items: Sequence[Tuple[int, torch.Size, int]]) -> None:
    for item_id, shape, block_size in items:
        if block_size <= 0:
            raise ValueError(f"DBuffer block size must be positive for item {item_id}")
        if shape.numel() % block_size != 0:
            raise ValueError(
                f"DBuffer item {item_id} has {shape.numel()} elements, which is not divisible "
                f"by block size {block_size}"
            )


def _tensor_boundary_is_valid(start: int, size: int, block_size: int, shard_size: int) -> bool:
    end = start + size
    boundary = ((start // shard_size) + 1) * shard_size
    while boundary < end:
        if (boundary - start) % block_size != 0:
            return False
        boundary += shard_size
    return True


def _next_valid_tensor_start(
    cursor: int,
    size: int,
    block_size: int,
    shard_size: int,
    num_devices: int,
) -> Optional[int]:
    total_size = shard_size * num_devices
    if cursor + size > total_size:
        return None
    for shard_idx in range(cursor // shard_size, num_devices):
        shard_start = shard_idx * shard_size
        shard_end = shard_start + shard_size
        base = max(cursor, shard_start)
        if base + size <= shard_end:
            return base
        lower = max(base, shard_end - size + 1)
        upper = shard_end - 1
        candidate = lower + ((shard_end - lower) % block_size)
        if candidate <= upper and _tensor_boundary_is_valid(
            candidate, size, block_size, shard_size
        ):
            return candidate
    return None


def _check_valid_shard_size(
    ordered_items: Sequence[Tuple[int, torch.Size, int]],
    shard_size: int,
    num_devices: int,
) -> Optional[Dict[int, int]]:
    cursor = 0
    offsets: Dict[int, int] = {}
    for item_id, shape, block_size in ordered_items:
        start = _next_valid_tensor_start(
            cursor, shape.numel(), block_size, shard_size, num_devices
        )
        if start is None:
            return None
        offsets[item_id] = start
        cursor = start + shape.numel()
    return offsets


def _minimal_feasible_shard_layout(
    ordered_items: Sequence[Tuple[int, torch.Size, int]],
    num_devices: int,
    alignment: int,
) -> Tuple[int, Dict[int, int]]:
    total_size = sum(shape.numel() for _, shape, _ in ordered_items)
    low_k = max(1, math.ceil(math.ceil(total_size / num_devices) / alignment))
    high_k = low_k
    offsets = None
    while offsets is None:
        offsets = _check_valid_shard_size(ordered_items, high_k * alignment, num_devices)
        if offsets is None:
            high_k *= 2
    best_k = high_k
    best_offsets = offsets
    while low_k <= high_k:
        mid_k = (low_k + high_k) // 2
        offsets = _check_valid_shard_size(ordered_items, mid_k * alignment, num_devices)
        if offsets is None:
            low_k = mid_k + 1
        else:
            best_k = mid_k
            best_offsets = offsets
            high_k = mid_k - 1
    return best_k * alignment, best_offsets


def _algorithm1_plan_offsets(
    ordered_items: Sequence[Tuple[int, torch.Size, int]],
    num_devices: int,
    collective_unit_size: int,
) -> Tuple[int, Dict[int, int]]:
    if not ordered_items:
        return 0, {}
    _validate_planner_items(ordered_items)
    alignment = max(1, collective_unit_size)
    best_shard_size = math.inf
    best_offsets: Dict[int, int] = {}
    for block_size in sorted({block_size for _, _, block_size in ordered_items}):
        alignment = math.lcm(alignment, block_size)
        shard_size, offsets = _minimal_feasible_shard_layout(
            ordered_items, num_devices, alignment
        )
        if shard_size < best_shard_size:
            best_shard_size = shard_size
            best_offsets = offsets
    return int(best_shard_size), best_offsets


def _ordered_item_variants(
    items: Sequence[Tuple[int, torch.Size, int]],
    tensor_order: str,
) -> List[Tuple[str, List[Tuple[int, torch.Size, int]]]]:
    items = list(items)
    if tensor_order == "default":
        return [("default", items)]
    if tensor_order == "block_size":
        return [("block_size", sorted(items, key=lambda item: (item[2], item[0])))]
    if tensor_order == "shape":
        return [("shape", sorted(items, key=lambda item: (tuple(item[1]), item[0])))]
    if tensor_order == "size":
        return [("size", sorted(items, key=lambda item: (item[1].numel(), item[0])))]
    if tensor_order == "best":
        return [
            ("default", items),
            ("block_size", sorted(items, key=lambda item: (item[2], item[0]))),
            ("shape", sorted(items, key=lambda item: (tuple(item[1]), item[0]))),
        ]
    raise ValueError(
        "tensor_order must be one of 'default', 'block_size', 'shape', 'size', or 'best'"
    )


def plan_dbuffer_layout(
    elements: Iterable[torch.Size],
    data_parallel_rank: int,
    data_parallel_world_size: int,
    is_data_distributed: bool,
    bucket_id: int,
    chunk_size_factor: int,
    pad_bucket: bool,
    item_block_size_fn: Optional[Callable[[torch.Size], int]] = None,
    item_block_sizes: Optional[Iterable[int]] = None,
    align_bucket_to_chunk_size: bool = False,
    tensor_order: str = "default",
    topology: Optional[DBufferDeviceTopology] = None,
    sharding_spec: Optional[DBufferShardingSpec] = None,
) -> DBufferPlan:
    if data_parallel_world_size <= 0:
        raise ValueError("data_parallel_world_size must be positive")
    if not 0 <= data_parallel_rank < data_parallel_world_size:
        raise ValueError("data_parallel_rank is out of range")
    if item_block_size_fn is not None and item_block_sizes is not None:
        raise ValueError("Only one of item_block_size_fn and item_block_sizes may be set")

    if topology is None:
        topology = DBufferDeviceTopology(
            mesh_shape=(data_parallel_world_size,), coordinate=(data_parallel_rank,)
        )
    if sharding_spec is None:
        sharding_spec = DBufferShardingSpec(placements=(DBUFFER_SHARD,))
    if topology.ndim != len(sharding_spec.placements):
        raise ValueError("DBuffer topology and sharding spec dimensions must match")

    shard_dims = sharding_spec.shard_dims()
    shard_world_size = (
        math.prod(topology.mesh_shape[dim] for dim in shard_dims) if shard_dims else 1
    )
    shard_coordinate = 0
    stride = 1
    for dim in reversed(shard_dims):
        shard_coordinate += topology.coordinate[dim] * stride
        stride *= topology.mesh_shape[dim]

    shapes = [torch.Size(shape) for shape in elements]
    if item_block_sizes is None:
        block_fn = item_block_size_fn or (lambda _shape: 1)
        block_sizes = [int(block_fn(shape)) for shape in shapes]
    else:
        block_sizes = [int(block_size) for block_size in item_block_sizes]
        if len(block_sizes) != len(shapes):
            raise ValueError("item_block_sizes length must match elements length")

    items = [
        (item_id, shape, block_size)
        for item_id, (shape, block_size) in enumerate(zip(shapes, block_sizes))
    ]
    collective_unit_size = max(1, chunk_size_factor) if align_bucket_to_chunk_size else 1

    best_score = None
    best_result = (tuple(), 0, {})
    for _, ordered_items in _ordered_item_variants(items, tensor_order):
        shard_size, offsets = _algorithm1_plan_offsets(
            ordered_items, shard_world_size, collective_unit_size
        )
        bucket_size = shard_size * shard_world_size
        padding = bucket_size - sum(shape.numel() for _, shape, _ in items)
        order = tuple(item_id for item_id, _, _ in ordered_items)
        score = (bucket_size, padding, order)
        if best_score is None or score < best_score:
            best_score = score
            best_result = (order, shard_size, offsets)

    selected_order, shard_size, offsets = best_result
    bucket_size = shard_size * shard_world_size
    if not pad_bucket and items:
        bucket_size = max(offsets[item_id] + shape.numel() for item_id, shape, _ in items)
        shard_size = math.ceil(bucket_size / shard_world_size)
        bucket_size = shard_size * shard_world_size

    planned_items = tuple(
        DBufferItem(
            item_id=item_id,
            global_data_index=offsets[item_id],
            size=shape.numel(),
            shape=shape,
            block_size=block_size,
            local_units=_item_local_units(
                offsets[item_id], shape.numel(), block_size, shard_size, shard_world_size
            ),
            ragged_dims=_infer_ragged_dims(shape, block_size),
        )
        for item_id, shape, block_size in sorted(items, key=lambda item: item[0])
    )

    bucket_data_index = shard_size * shard_coordinate
    return DBufferPlan(
        bucket_id=bucket_id,
        bucket_size=bucket_size,
        topology=topology,
        sharding_spec=sharding_spec,
        items=planned_items,
        shard=DBufferShard(
            bucket_id=bucket_id,
            global_data_index=bucket_data_index,
            local_data_index=0 if is_data_distributed else bucket_data_index,
            bucket_data_index=bucket_data_index,
            size=shard_size,
        ),
        ragged_shard=RaggedShard(
            dims=(0,), local_units=tuple(1 for _ in range(shard_world_size))
        ),
        tensor_order=selected_order,
    )
