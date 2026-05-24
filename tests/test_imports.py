from grove_fsdp import GroveFSDP, fully_shard_model
from grove_fsdp.dbuffer import RaggedShard
from grove_fsdp.distributed_data_parallel_config import DistributedDataParallelConfig


def test_public_imports() -> None:
    assert GroveFSDP.__name__ == "GroveFSDP"
    assert fully_shard_model.__name__ == "fully_shard_model"
    assert DistributedDataParallelConfig().overlap_grad_reduce is False


def test_ragged_shard_placement_flags() -> None:
    placement = RaggedShard((0,), (1, 2))
    assert placement.is_ragged_shard()
    assert not placement.is_replicate()
    assert not placement.is_shard()
    assert not placement.is_partial()

