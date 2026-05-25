from grove_fsdp.utils import FSDPDistributedIndex


class _FakeGroup:
    def __init__(self, name: str, size: int = 1, rank: int = 0) -> None:
        self.name = name
        self.group_desc = name
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


class _FakeMesh:
    def __init__(self, mesh_dim_names, groups=None, group=None) -> None:
        self.mesh_dim_names = tuple(mesh_dim_names)
        self._groups = groups or {}
        self._group = group
        self._flatten_mapping = {}

    def _get_root_mesh(self):
        return self

    def __getitem__(self, key):
        if isinstance(key, str):
            return _FakeMesh((key,), self._groups, self._groups[key])
        group = self._groups[key[0]] if len(key) == 1 else self._group
        return _FakeMesh(tuple(key), self._groups, group)

    def get_group(self):
        return self._group


def test_expert_groups_fall_back_to_regular_fsdp_without_expert_mesh() -> None:
    fsdp_group = _FakeGroup("fsdp")
    device_mesh = _FakeMesh(("dp_shard",), {"dp_shard": fsdp_group})

    dist_index = FSDPDistributedIndex(
        device_mesh=device_mesh,
        dp_shard_dim="dp_shard",
        expt_device_mesh=None,
    )

    assert dist_index.get_fsdp_group(is_expert_parallel=True) is fsdp_group
    assert dist_index.get_dp_group(is_expert_parallel=True) is fsdp_group
    assert dist_index.get_root_mesh(is_expert_parallel=True) is device_mesh
    assert dist_index.get_submesh("dp_shard", is_expert_parallel=True).get_group() is fsdp_group


def test_expert_groups_fall_back_to_regular_hsdp_without_expert_mesh() -> None:
    outer_group = _FakeGroup("outer")
    shard_group = _FakeGroup("shard")
    hybrid_group = _FakeGroup("hybrid")
    device_mesh = _FakeMesh(
        ("dp_replicate", "dp_shard"),
        {
            "dp_replicate": outer_group,
            "dp_shard": shard_group,
        },
        hybrid_group,
    )

    dist_index = FSDPDistributedIndex(
        device_mesh=device_mesh,
        dp_outer_dim="dp_replicate",
        dp_shard_dim="dp_shard",
        hybrid_fsdp_group=hybrid_group,
        expt_device_mesh=None,
    )

    assert dist_index.get_fsdp_group(is_expert_parallel=True) is shard_group
    assert dist_index.get_outer_fsdp_group(is_expert_parallel=True) is outer_group
    assert dist_index.get_dp_group(is_expert_parallel=True) is hybrid_group
    assert dist_index.get_root_mesh(is_expert_parallel=True) is device_mesh
