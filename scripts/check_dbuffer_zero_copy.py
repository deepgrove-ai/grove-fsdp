#!/usr/bin/env python3
"""Check DBuffer zero-copy view behavior with data pointers.

This script verifies that DBuffer view helpers return tensor views backed by
the same allocation as their source tensors. For sliced views, ``data_ptr()``
is expected to equal the source pointer plus the view's byte offset.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "veScale"):
    sys.path.insert(0, str(path))

from grove_fsdp.dbuffer import DistributedBuffer, plan_dbuffer_layout  # noqa: E402


def storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def expected_data_ptr(base: torch.Tensor, view: torch.Tensor) -> int:
    return base.data_ptr() + view.storage_offset() * view.element_size()


def assert_zero_copy_view(name: str, base: torch.Tensor, view: torch.Tensor) -> None:
    same_storage = storage_ptr(base) == storage_ptr(view)
    expected_ptr = expected_data_ptr(base, view)
    same_data_ptr = view.data_ptr() == expected_ptr

    print(
        f"{name}: "
        f"base_storage=0x{storage_ptr(base):x} view_storage=0x{storage_ptr(view):x} "
        f"base_data=0x{base.data_ptr():x} view_data=0x{view.data_ptr():x} "
        f"offset={view.storage_offset()} expected_view_data=0x{expected_ptr:x}"
    )

    if not same_storage:
        raise AssertionError(f"{name} does not share backing storage")
    if not same_data_ptr:
        raise AssertionError(f"{name} data_ptr does not match the expected offset")
    if view._base is None:
        raise AssertionError(f"{name} is not recorded by PyTorch as a view")


def check_bucket_item_views() -> None:
    plan = plan_dbuffer_layout(
        elements=[torch.Size([2, 2]), torch.Size([3]), torch.Size([1, 2])],
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

    for idx, view in enumerate(views):
        assert_zero_copy_view(f"bucket_item_views[{idx}]", bucket, view)

    views[1].fill_(77)
    torch.testing.assert_close(bucket[4:7], torch.full((3,), 77.0))


def check_sharded_views() -> None:
    plan = plan_dbuffer_layout(
        elements=[torch.Size([4]), torch.Size([4]), torch.Size([4])],
        data_parallel_rank=1,
        data_parallel_world_size=2,
        is_data_distributed=True,
        bucket_id=1,
        chunk_size_factor=1,
        pad_bucket=True,
    )
    dbuffer = DistributedBuffer(plan, is_data_distributed=True)
    local = torch.arange(plan.shard.size, dtype=torch.float32)
    bucket = torch.empty(plan.bucket_size, dtype=torch.float32)
    dbuffer.init_data(local)

    shard_view = dbuffer.shard_bucket_view(bucket)
    assert_zero_copy_view("shard_bucket_view", bucket, shard_view)

    item_views: Iterable[tuple[int, torch.Tensor]] = (
        (item.item_id, dbuffer.item_local_view(item.item_id)) for item in plan.items
    )
    for item_id, view in item_views:
        if view.numel() == 0:
            continue
        assert_zero_copy_view(f"item_local_view[{item_id}]", local, view)

    non_empty_views = [
        dbuffer.item_local_view(item.item_id)
        for item in plan.items
        if dbuffer.item_local_view(item.item_id).numel()
    ]
    first_non_empty = non_empty_views[0]
    first_non_empty.fill_(123)
    expected_start = first_non_empty.storage_offset()
    expected_end = expected_start + first_non_empty.numel()
    torch.testing.assert_close(
        local[expected_start:expected_end],
        torch.full((first_non_empty.numel(),), 123.0),
    )


def main() -> int:
    check_bucket_item_views()
    check_sharded_views()
    print("OK: DBuffer views are zero-copy by storage and data pointer checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
