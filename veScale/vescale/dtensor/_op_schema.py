################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################
from torch.distributed.tensor._op_schema import (
    OpInfo,
    OpSpec,
    OpSchema,
    OpStrategy,
    OutputSharding,
    OutputSpecType,
    TupleStrategy,
    PlacementList,
    RuntimeSchemaInfo,
    StrategyType,
)

PlacementStrategy = OpSpec

try:
    from torch.distributed.tensor._op_schema import _is_inplace_op, _is_out_variant_op
except ImportError:

    def _is_inplace_op(op):
        # PyTorch 2.9+ exposes this check as OpSchema.is_inplace_op() instead
        # of a module-level helper. Keep the old veScale helper API intact.
        return op._schema.name[-1] == "_"

    def _is_out_variant_op(op):
        return "out" in op._schema.overload_name

__all__ = [
    "OpInfo",
    "_is_inplace_op",
    "_is_out_variant_op",
    "OpSchema",
    "OpStrategy",
    "OutputSharding",
    "OutputSpecType",
    "PlacementStrategy",
    "TupleStrategy",
    "PlacementList",
    "RuntimeSchemaInfo",
    "StrategyType",
]
