#!/usr/bin/env python3
"""Profile Qwen3-4B forward/backward speed with Grove-FSDP or torch FSDP2.

Run with torchrun, for example:

    torchrun --nproc_per_node=8 scripts/profile_qwen3_4b_fsdp.py --backend grove
    torchrun --nproc_per_node=8 scripts/profile_qwen3_4b_fsdp.py --backend both --print-grove-layout --wandb

By default this constructs the Qwen3-4B architecture from an in-script config
and random weights. Pass --from-pretrained to load model weights instead.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import scripts.profile_qwen25_1p5b_fsdp as profiler  # noqa: E402

_base_parse_args = profiler.parse_args
_base_rank_print = profiler.rank_print


def get_qwen_classes() -> tuple[type[Any], type[Any], type[Any]]:
    try:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    except ImportError as exc:
        raise RuntimeError(
            "This benchmark requires a transformers version with Qwen3 support. "
            "Install/update transformers or run with --from-pretrained in an "
            "environment that provides Qwen3 classes."
        ) from exc
    return Qwen3Config, Qwen3ForCausalLM, Qwen3DecoderLayer


def local_qwen3_4b_config(dtype: torch.dtype) -> Any:
    Qwen3Config, _, _ = get_qwen_classes()
    return Qwen3Config(
        vocab_size=151936,
        hidden_size=2560,
        intermediate_size=9728,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=40960,
        max_window_layers=36,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        tie_word_embeddings=True,
        rope_theta=1_000_000.0,
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
        sliding_window=None,
        torch_dtype=dtype,
    )


def parse_args():
    args = _base_parse_args()
    argv = sys.argv[1:]
    if not any(arg == "--model-name" or arg.startswith("--model-name=") for arg in argv):
        args.model_name = "Qwen/Qwen3-4B"
    if not any(arg == "--seq-len" or arg.startswith("--seq-len=") for arg in argv):
        args.seq_len = 2048
    return args


def ensure_profile_dir(args, rank: int) -> None:
    if not args.wandb or args.profile_dir is not None:
        return
    profile_dir = os.path.join("profiles", f"qwen3_4b_{int(time.time())}") if rank == 0 else None
    payload = [profile_dir]
    profiler.dist.broadcast_object_list(payload, src=0)
    args.profile_dir = payload[0]


def rank_print(*values: object) -> None:
    patched_values = tuple(
        value.replace("Qwen2.5-1.5B", "Qwen3-4B") if isinstance(value, str) else value
        for value in values
    )
    _base_rank_print(*patched_values)


profiler.__doc__ = __doc__
profiler.get_qwen_classes = get_qwen_classes
profiler.local_qwen25_1p5b_config = local_qwen3_4b_config
profiler.parse_args = parse_args
profiler.ensure_profile_dir = ensure_profile_dir
profiler.rank_print = rank_print


if __name__ == "__main__":
    profiler.main()
