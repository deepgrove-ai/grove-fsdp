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

The ``RaggedShard`` placement is adapted from veScale's open-source
implementation. Grove-FSDP still communicates flat DBuffer buckets today,
but the placement itself can split, scatter, gather, and reshard uneven flat
tensors for callers that need a real DTensor-like placement object.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import ProcessGroup, get_global_rank
from torch.distributed.tensor.placement_types import Placement, Replicate


def is_ragged_shard(placement: Placement) -> bool:
    """Return true if ``placement`` is a Grove-FSDP RaggedShard."""
    return isinstance(placement, RaggedShard)


if not hasattr(Placement, "is_ragged_shard"):
    Placement.is_ragged_shard = is_ragged_shard  # type: ignore[attr-defined]


def mesh_scatter_ragged(
    output: torch.Tensor,
    scatter_list: List[torch.Tensor],
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    *,
    group_src: int = 0,
) -> None:
    """Scatter uneven tensor shards across a DeviceMesh dimension.

    PyTorch's public scatter utilities expect even output sizes. This mirrors
    veScale's send/recv based implementation for ragged tensors and only relies
    on APIs available in PyTorch 2.9.1.
    """
    if output.is_meta:
        return

    dim_group = mesh.get_group(mesh_dim)
    if not isinstance(dim_group, ProcessGroup):
        raise TypeError(f"Expected ProcessGroup for mesh dim {mesh_dim}, got {type(dim_group)}")

    group_rank = torch.distributed.get_rank(dim_group)
    if group_src == group_rank:
        for rank, shard in enumerate(scatter_list):
            if rank == group_src:
                continue
            torch.distributed.send(shard.contiguous(), dst=get_global_rank(dim_group, rank))
        output.copy_(scatter_list[group_src])
    else:
        torch.distributed.recv(output, src=get_global_rank(dim_group, group_src))


def unravel_index(index: int, shape: Tuple[int, ...]) -> List[int]:
    """Convert a flat row-major index into coordinates."""
    coords = [0] * len(shape)
    for dim in range(len(shape) - 1, -1, -1):
        coords[dim] = index % shape[dim]
        index //= shape[dim]
    return coords


def flatten_index(index: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
    """Convert row-major coordinates into a flat index."""
    if len(shape) != len(index):
        raise ValueError(f"Shape length {len(shape)} and index length {len(index)} must match")

    flat_index = 0
    stride = 1
    for size, coord in zip(reversed(shape), reversed(index)):
        if not 0 <= coord < size:
            raise IndexError(f"Index {coord} out of bounds for dimension size {size}")
        flat_index += coord * stride
        stride *= size
    return flat_index


def get_ragged_shard(placements: Iterable[Placement]) -> Tuple[int, RaggedShard]:
    """Return the single RaggedShard placement and its mesh-dimension index."""
    ragged_placement = None
    ragged_placement_idx = None
    n_other_placements = 0
    placements_tuple = tuple(placements)
    for idx, placement in enumerate(placements_tuple):
        if isinstance(placement, RaggedShard):
            if ragged_placement is not None:
                raise RuntimeError("Only one RaggedShard placement is supported")
            if n_other_placements != 0:
                raise RuntimeError(f"RaggedShard must appear before non-replicate placements: {placements_tuple}")
            ragged_placement = placement
            ragged_placement_idx = idx
            continue
        if isinstance(placement, Replicate):
            continue
        n_other_placements += 1

    if ragged_placement is None or ragged_placement_idx is None:
        raise RuntimeError(f"No RaggedShard placement found in {placements_tuple}")
    return ragged_placement_idx, ragged_placement


def substitute_ragged_with_replicate(placements: Iterable[Placement]) -> Tuple[Placement, ...]:
    """Replace the RaggedShard placement with Replicate."""
    placements_tuple = tuple(placements)
    idx, _ = get_ragged_shard(placements_tuple)
    return (*placements_tuple[:idx], Replicate(), *placements_tuple[idx + 1 :])


@dataclasses.dataclass(frozen=True)
class RaggedShard(Placement):
    """A ragged DTensor placement over contiguous flattened tensor storage.

    Args:
        dims: Prefix tensor dimensions represented by the flattened ragged shard.
            For ``shape=(n, m, k)`` and ``dims=(0,)``, each local flat tensor is
            reconstructable as ``(-1, m, k)``.
        local_units: Relative per-rank allocation. The tuple length must equal
            ``mesh.size(mesh_dim)`` when used with a device mesh.

    Notes:
        This is intentionally scoped to the veScale RaggedShard core that is
        compatible with PyTorch 2.9.1 placement APIs. It does not install
        veScale's broader DTensor dispatch/redistribute monkey patches.
    """

    dims: Tuple[int, ...]
    local_units: Tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.dims) == 0:
            raise ValueError("RaggedShard dims must be non-empty")
        if self.dims != tuple(range(len(self.dims))):
            raise ValueError(f"RaggedShard only supports prefix dims, got {self.dims}")
        if len(self.local_units) == 0:
            raise ValueError("RaggedShard local_units must be non-empty")
        if any(unit < 0 for unit in self.local_units):
            raise ValueError(f"RaggedShard local_units must be non-negative, got {self.local_units}")
        if sum(self.local_units) <= 0:
            raise ValueError(f"RaggedShard local_units must sum to a positive value, got {self.local_units}")

    def is_ragged_shard(self) -> bool:
        """Return true for compatibility with veScale placement checks."""
        return True

    def is_replicate(self) -> bool:
        """Override PyTorch's Placement method for pybind-subclass safety."""
        return False

    def is_shard(self, dim: Optional[int] = None) -> bool:
        """Override PyTorch's Placement method for pybind-subclass safety."""
        return False

    def is_partial(self) -> bool:
        """Override PyTorch's Placement method for pybind-subclass safety."""
        return False

    def _split_tensor(self, tensor: torch.Tensor, num_chunks: int) -> List[torch.Tensor]:
        """Split a contiguous tensor into uneven flat shards."""
        if not tensor.is_contiguous():
            raise ValueError("RaggedShard expects a contiguous tensor")
        if num_chunks != len(self.local_units):
            raise ValueError(
                f"num_chunks ({num_chunks}) must equal len(local_units) ({len(self.local_units)})"
            )

        total_units = sum(self.local_units)
        total_numel = tensor.numel()
        if total_numel % total_units != 0:
            raise ValueError(
                f"tensor.numel() ({total_numel}) must be divisible by sum(local_units) ({total_units})"
            )

        unit_numel = total_numel // total_units
        flat_tensor = tensor.view(-1)
        start_idx = 0
        shard_list = []
        for local_unit in self.local_units:
            shard_len = local_unit * unit_numel
            shard_list.append(flat_tensor.narrow(0, start_idx, shard_len))
            start_idx += shard_len
        return shard_list

    def _ragged_shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> torch.Tensor:
        """Shard and scatter a tensor over ``mesh_dim`` using ragged sizes."""
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        num_chunks = mesh.size(mesh_dim=mesh_dim)
        if len(self.local_units) != num_chunks:
            raise ValueError(
                f"len(local_units) ({len(self.local_units)}) must equal mesh dim size ({num_chunks})"
            )

        mesh_dim_local_rank = my_coordinate[mesh_dim]
        scatter_list = self._split_tensor(tensor, num_chunks)
        if src_data_rank is None:
            return scatter_list[mesh_dim_local_rank]

        output = torch.empty_like(scatter_list[mesh_dim_local_rank])
        mesh_scatter_ragged(
            output,
            scatter_list,
            mesh,
            mesh_dim=mesh_dim,
            group_src=src_data_rank,
        )
        return output

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: Iterable[int],
    ) -> torch.Tensor:
        """Gather ragged local shards into a replicated flat tensor."""
        logical_numel = math.prod(tuple(current_logical_shape))
        total_units = sum(self.local_units)
        if logical_numel % total_units != 0:
            raise ValueError(
                f"logical numel ({logical_numel}) must be divisible by sum(local_units) ({total_units})"
            )

        unit_numel = logical_numel // total_units
        tensor_list = [
            torch.empty(
                unit_numel * self.local_units[rank],
                dtype=local_tensor.dtype,
                device=local_tensor.device,
            )
            for rank in range(mesh.size(mesh_dim))
        ]
        torch.distributed.all_gather(tensor_list, local_tensor.contiguous(), group=mesh.get_group(mesh_dim))
        return torch.cat(tensor_list)

    def _to_new_ragged_shard(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: Iterable[int],
        new_local_units: Tuple[int, ...],
    ) -> torch.Tensor:
        """Redistribute from this ragged layout to another local-unit layout."""
        numel = math.prod(tuple(current_logical_shape))
        src_total_units = sum(self.local_units)
        dst_total_units = sum(new_local_units)
        if numel % src_total_units != 0 or numel % dst_total_units != 0:
            raise ValueError(
                "current_logical_shape numel must be divisible by both old and new local units"
            )

        src_slices = tuple(unit * (numel // src_total_units) for unit in self.local_units)
        dst_slices = tuple(unit * (numel // dst_total_units) for unit in new_local_units)
        coord = mesh.get_coordinate()
        if coord is None:
            return local_tensor.new_empty(0)
        rank = coord[mesh_dim]

        input_tensor_list = []
        src_left = sum(src_slices[:rank])
        src_right = src_left + src_slices[rank]
        for dst_rank in range(len(new_local_units)):
            dst_left = sum(dst_slices[:dst_rank])
            dst_right = dst_left + dst_slices[dst_rank]
            if dst_right <= src_left or dst_left >= src_right:
                input_tensor_list.append(torch.empty(0, dtype=local_tensor.dtype, device=local_tensor.device))
            else:
                start = max(dst_left, src_left)
                end = min(dst_right, src_right)
                input_tensor_list.append(local_tensor.narrow(0, start - src_left, end - start))

        output_tensor_list = []
        dst_left = sum(dst_slices[:rank])
        dst_right = dst_left + dst_slices[rank]
        for src_rank in range(len(self.local_units)):
            src_left = sum(src_slices[:src_rank])
            src_right = src_left + src_slices[src_rank]
            length = max(0, min(src_right, dst_right) - max(src_left, dst_left))
            output_tensor_list.append(torch.empty(length, dtype=local_tensor.dtype, device=local_tensor.device))

        torch.distributed.all_to_all(
            output_tensor_list,
            input_tensor_list,
            group=mesh.get_group(mesh_dim),
        )
        return torch.cat(output_tensor_list)

    def reconstruct_tensor_from_flat(self, flat_tensor: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """Recover a local unflattened tensor view from a flat ragged shard."""
        if flat_tensor.ndim != 1:
            raise ValueError("flat_tensor must be 1-dimensional")
        ndim = len(self.dims)
        suffix_numel = math.prod(shape[ndim:]) if ndim < len(shape) else 1
        if flat_tensor.numel() % suffix_numel != 0:
            raise ValueError(
                f"flat_tensor.numel() ({flat_tensor.numel()}) must be divisible by suffix numel ({suffix_numel})"
            )
        return flat_tensor.view(-1, *shape[ndim:])

    def __repr__(self) -> str:
        return f"RaggedShard(dims={self.dims}, local_units={self.local_units})"

    def __str__(self) -> str:
        return repr(self)


@dataclasses.dataclass(frozen=True)
class _StridedRaggedShard(RaggedShard):
    """RaggedShard analogue of PyTorch's private _StridedShard metadata."""

    split_factor: int


@dataclasses.dataclass(frozen=True)
class DBufferItem:
    """One tensor's planned location in a distributed buffer."""

    item_id: int
    global_data_index: int
    size: int
    shape: torch.Size
    block_size: int


@dataclasses.dataclass(frozen=True)
class DBufferShard:
    """The local shard view of a global DBuffer bucket."""

    bucket_id: int
    global_data_index: int
    local_data_index: int
    bucket_data_index: int
    size: int


@dataclasses.dataclass(frozen=True)
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
        preserve_item_order: If true, keeps the legacy compact-in-order layout
            instead of running the RaggedShard planner.

    Returns:
        A DBufferPlan that can be translated into the existing bucket index types.
    """

    collective_unit_size = max(1, chunk_size_factor)
    shard_quantum = data_parallel_world_size * collective_unit_size
    item_block_size_fn = item_block_size_fn or (
        lambda shape: _default_item_block_size(shape, chunk_size_factor)
    )
    items = [
        (item_id, torch.Size(shape), max(1, item_block_size_fn(torch.Size(shape))))
        for item_id, shape in enumerate(elements)
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
