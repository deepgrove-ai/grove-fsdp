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

"""RaggedShard layout planning and DBuffer metadata for Grove-FSDP.

Grove-FSDP uses the vendored veScale RaggedShard placement implementation.
This module owns only the Grove DBuffer planning metadata that consumes that
placement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from vescale.dtensor.placement_types import (
    RaggedShard,
    _StridedRaggedShard,
    is_ragged_shard,
)
from vescale.dtensor.vescale_utils.ragged_shard_utils import (
    flatten_index,
    get_ragged_shard,
    substitute_ragged_with_replicate,
    unravel_index,
)


@dataclass(frozen=True)
class DBufferItem:
    """One tensor's planned location in a distributed buffer."""

    item_id: int
    global_data_index: int
    size: int
    shape: torch.Size
    block_size: int


@dataclass(frozen=True)
class DBufferShard:
    """The local shard view of a global DBuffer bucket."""

    bucket_id: int
    global_data_index: int
    local_data_index: int
    bucket_data_index: int
    size: int


@dataclass(frozen=True)
class DBufferPlan:
    """Planned logical layout for one FSDP communication bucket."""

    bucket_id: int
    bucket_size: int
    items: Tuple[DBufferItem, ...]
    shard: DBufferShard
    ragged_shard: RaggedShard

    @property
    def padding(self) -> int:
        """Return planned padding elements in the bucket."""
        return self.bucket_size - sum(item.size for item in self.items)


class DistributedBuffer:
    """Metadata wrapper for a planned distributed buffer.

    The wrapper owns the logical DBuffer plan and optionally a backing tensor.
    Existing Grove-FSDP paths still operate on ``DataParallelBuffer.data``;
    this class centralizes the plan/backing relationship without forcing a
    broad rewrite of communication code.
    """

    def __init__(self, plan: DBufferPlan, is_data_distributed: bool) -> None:
        self.plan = plan
        self.is_data_distributed = is_data_distributed
        self.data: Optional[torch.Tensor] = None

    def init_data(self, data: torch.Tensor) -> None:
        """Attach backing tensor storage to this DBuffer."""
        expected_size = self.plan.shard.size if self.is_data_distributed else self.plan.bucket_size
        if data.numel() != expected_size:
            raise ValueError(
                f"DBuffer backing tensor has {data.numel()} elements, expected {expected_size}"
            )
        self.data = data

    @property
    def local_tensor(self) -> torch.Tensor:
        """Return the local backing tensor."""
        if self.data is None:
            raise RuntimeError("DBuffer storage has not been initialized")
        return self.data


def _pad(number_to_be_padded: int, divisor: int) -> int:
    return int(math.ceil(number_to_be_padded / divisor) * divisor)


def _default_item_block_size(shape: torch.Size, requested_block_size: Optional[int]) -> int:
    if requested_block_size is not None:
        return requested_block_size
    if len(shape) <= 1:
        return 1
    return max(1, shape[1:].numel())


def _compact_in_original_order(
    items: List[Tuple[int, torch.Size, int]],
    shard_quantum: int,
) -> Dict[int, int]:
    """Fallback planner preserving module parameter order."""

    planned: Dict[int, int] = {}
    offset = 0
    for item_id, shape, block_size in items:
        alignment = math.gcd(block_size, shard_quantum)
        offset = _pad(offset, alignment)
        planned[item_id] = offset
        offset += shape.numel()
    return planned


def _validate_planner_items(items: List[Tuple[int, torch.Size, int]]) -> None:
    for item_id, shape, block_size in items:
        size = shape.numel()
        if block_size <= 0:
            raise ValueError(f"RaggedShard block size must be positive for item {item_id}")
        if size % block_size != 0:
            raise ValueError(
                f"RaggedShard item {item_id} has {size} elements, which is not divisible "
                f"by block size {block_size}"
            )


def _tensor_boundary_is_valid(start: int, size: int, block_size: int, shard_size: int) -> bool:
    """Return true if shard boundaries crossing a tensor preserve whole blocks."""
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
    """Find the earliest start >= cursor satisfying Algorithm 1's constraints."""
    total_size = shard_size * num_devices
    if cursor + size > total_size:
        return None

    first_shard = cursor // shard_size
    for shard_idx in range(first_shard, num_devices):
        shard_start = shard_idx * shard_size
        shard_end = shard_start + shard_size
        base = max(cursor, shard_start)

        # Case 1 from the paper: the tensor lies entirely within a local shard.
        if base + size <= shard_end:
            return base

        if shard_end >= total_size:
            break

        # Cases 2/3: the tensor crosses this shard boundary. The boundary must
        # land on a tensor block boundary, i.e. start == shard_end (mod block).
        lower = max(base, shard_end - size + 1)
        upper = shard_end - 1
        start = lower + ((shard_end - lower) % block_size)
        if start <= upper and _tensor_boundary_is_valid(
            start, size, block_size, shard_size
        ):
            return start

    return None


def _check_valid_shard_size(
    ordered_items: List[Tuple[int, torch.Size, int]],
    shard_size: int,
    num_devices: int,
) -> Optional[Dict[int, int]]:
    """Check a candidate per-device buffer size and return offsets if feasible."""
    cursor = 0
    offsets: Dict[int, int] = {}
    for item_id, shape, block_size in ordered_items:
        start = _next_valid_tensor_start(
            cursor,
            shape.numel(),
            block_size,
            shard_size,
            num_devices,
        )
        if start is None:
            return None
        offsets[item_id] = start
        cursor = start + shape.numel()
    return offsets


def _minimal_feasible_shard_layout(
    ordered_items: List[Tuple[int, torch.Size, int]],
    num_devices: int,
    alignment: int,
) -> Tuple[int, Dict[int, int]]:
    """Binary-search the smallest feasible shard size for a fixed alignment."""
    total_size = sum(shape.numel() for _, shape, _ in ordered_items)
    lower_k = max(1, math.ceil(math.ceil(total_size / num_devices) / alignment))
    high_k = lower_k
    offsets = None

    while offsets is None:
        offsets = _check_valid_shard_size(ordered_items, high_k * alignment, num_devices)
        if offsets is None:
            high_k *= 2

    low_k = lower_k
    best_k = high_k
    best_offsets = offsets
    while low_k <= high_k:
        mid_k = (low_k + high_k) // 2
        offsets = _check_valid_shard_size(ordered_items, mid_k * alignment, num_devices)
        if offsets is not None:
            best_k = mid_k
            best_offsets = offsets
            high_k = mid_k - 1
        else:
            low_k = mid_k + 1

    return best_k * alignment, best_offsets


def _algorithm1_plan_offsets(
    ordered_items: List[Tuple[int, torch.Size, int]],
    num_devices: int,
    collective_unit_size: int,
) -> Tuple[int, Dict[int, int]]:
    """Plan offsets using veScale-FSDP Algorithm 1 for the given tensor order.

    The paper fixes an ordered tensor list and searches over least-common-multiple
    alignments of the collective unit and per-tensor RaggedShard block sizes. For
    each alignment, ``CheckValidShard`` determines whether all tensors can fit in
    ``num_devices`` uniform local shards while preserving non-shardable blocks.
    """
    if not ordered_items:
        return 0, {}

    _validate_planner_items(ordered_items)
    alignment = max(1, collective_unit_size)
    best_shard_size = math.inf
    best_offsets: Dict[int, int] = {}

    for block_size in sorted({block_size for _, _, block_size in ordered_items}):
        alignment = math.lcm(alignment, block_size)
        shard_size, offsets = _minimal_feasible_shard_layout(
            ordered_items,
            num_devices,
            alignment,
        )
        if shard_size < best_shard_size:
            best_shard_size = shard_size
            best_offsets = offsets

    return int(best_shard_size), best_offsets


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
    preserve_item_order: bool = False,
) -> DBufferPlan:
    """Plan a RaggedShard-compatible DBuffer layout.

    Args:
        elements: Tensor shapes in this bucket.
        data_parallel_rank: Current rank in the DP/FSDP group.
        data_parallel_world_size: World size of the DP/FSDP group.
        is_data_distributed: Whether the persistent buffer is actually sharded.
        bucket_id: Bucket identifier.
        chunk_size_factor: Existing Grove-FSDP communication segmentation unit.
        pad_bucket: Whether the bucket must be padded to shard evenly.
        item_block_size_fn: Optional per-shape block-size override.
        item_block_sizes: Optional per-item block sizes. Mutually exclusive with
            item_block_size_fn.
        align_bucket_to_chunk_size: If true, require each per-rank shard size
            to be a multiple of chunk_size_factor. The default keeps
            communication chunking separate from DBuffer layout planning, which
            reduces padding for mixed row sizes.
        preserve_item_order: If true, keeps the legacy compact-in-order layout
            instead of running the RaggedShard planner.

    Returns:
        A DBufferPlan that can be translated into the existing bucket index types.
    """

    collective_unit_size = max(1, chunk_size_factor) if align_bucket_to_chunk_size else 1
    shard_quantum = data_parallel_world_size * collective_unit_size
    if item_block_size_fn is not None and item_block_sizes is not None:
        raise ValueError("Only one of item_block_size_fn and item_block_sizes may be set")
    item_block_size_fn = item_block_size_fn or (
        lambda shape: _default_item_block_size(shape, None)
    )
    shapes = [torch.Size(shape) for shape in elements]
    if item_block_sizes is None:
        block_sizes = [item_block_size_fn(shape) for shape in shapes]
    else:
        block_sizes = list(item_block_sizes)
        if len(block_sizes) != len(shapes):
            raise ValueError(
                f"item_block_sizes has {len(block_sizes)} entries, expected {len(shapes)}"
            )
    items = [
        (item_id, shape, max(1, int(block_size)))
        for item_id, (shape, block_size) in enumerate(zip(shapes, block_sizes))
    ]

    if preserve_item_order:
        offsets = _compact_in_original_order(items, shard_quantum)
        end_offset = max(
            (offsets[item_id] + shape.numel() for item_id, shape, _ in items),
            default=0,
        )
        bucket_size = _pad(end_offset, shard_quantum) if pad_bucket else end_offset
    else:
        shard_size, offsets = _algorithm1_plan_offsets(
            items,
            num_devices=data_parallel_world_size,
            collective_unit_size=collective_unit_size,
        )
        bucket_size = shard_size * data_parallel_world_size
        if not pad_bucket:
            bucket_size = max(
                (offsets[item_id] + shape.numel() for item_id, shape, _ in items),
                default=0,
            )

    planned_items = tuple(
        DBufferItem(
            item_id=item_id,
            global_data_index=offsets[item_id],
            size=shape.numel(),
            shape=shape,
            block_size=block_size,
        )
        for item_id, shape, block_size in sorted(items, key=lambda item: item[0])
    )

    shard_size = bucket_size // data_parallel_world_size
    bucket_data_index = shard_size * data_parallel_rank
    global_data_index = bucket_data_index
    local_data_index = 0 if is_data_distributed else global_data_index
    shard = DBufferShard(
        bucket_id=bucket_id,
        global_data_index=global_data_index,
        local_data_index=local_data_index,
        bucket_data_index=bucket_data_index,
        size=shard_size,
    )
    return DBufferPlan(
        bucket_id=bucket_id,
        bucket_size=bucket_size,
        items=planned_items,
        shard=shard,
        ragged_shard=RaggedShard(
            dims=(0,),
            local_units=tuple(1 for _ in range(data_parallel_world_size)),
        ),
    )
