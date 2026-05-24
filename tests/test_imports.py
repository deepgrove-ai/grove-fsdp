import torch
from vescale.dtensor.placement_types import RaggedShard

from grove_fsdp import GroveFSDP, fully_shard_model
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


def test_grove_fsdp_delegates_unknown_attributes_to_wrapped_module() -> None:
    class ModuleWithMetadata(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.config = {"hidden_size": 2}

    wrapped = ModuleWithMetadata()
    fsdp = GroveFSDP.__new__(GroveFSDP)
    torch.nn.Module.__init__(fsdp)
    fsdp.module = wrapped

    assert fsdp.config == {"hidden_size": 2}
    assert fsdp.linear is wrapped.linear
    assert fsdp.module is wrapped
