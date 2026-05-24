# Grove-FSDP

Grove-FSDP is a standalone PyTorch package for fully sharded data parallel
training. It is derived from Megatron-FSDP, but is packaged for direct use as
`grove_fsdp` outside the Megatron-LM source tree.

The package keeps the existing high-performance FSDP buffer path and adds
RaggedShard-aware DBuffer layout planning. The RaggedShard placement core is
adapted from veScale and is scoped to PyTorch placement APIs available in
PyTorch 2.9.1.

## Install

From a local checkout:

```bash
python -m pip install .
```

Build artifacts:

```bash
python -m build
```

## Quick Start

```python
import torch

from grove_fsdp import fully_shard_model, fully_shard_optimizer

torch.distributed.init_process_group()
torch.cuda.set_device(torch.distributed.get_rank())

model = torch.nn.Transformer().cuda()
fsdp_model = fully_shard_model(
    module=model,
    fsdp_unit_modules=[torch.nn.TransformerEncoder, torch.nn.TransformerDecoder],
)

optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-3)
optimizer = fully_shard_optimizer(optimizer)
```

## Main APIs

- `GroveFSDP`
- `fully_shard_model`
- `fully_shard_optimizer`
- `fully_shard`
- `MixedPrecisionPolicy`
- `DistributedDataParallelConfig`
- `RaggedShard`
- `DistributedBuffer`
- `DBufferPlan`

## Notes

Megatron-Core and TransformerEngine integrations are optional. When they are not
installed, Grove-FSDP uses local fallback utilities where possible.
