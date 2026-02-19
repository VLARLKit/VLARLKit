import functools
import logging
from typing import Any

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import (
    _module_wrap_policy,
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers.trainer_pt_utils import get_module_class_from_name

from vlarlkit.models.base import BaseModel

logger = logging.getLogger(__name__)


def get_fsdp_wrap_policy(model: torch.nn.Module):
    """Build a combined FSDP auto-wrap policy from model attributes.

    Combines three kinds of sub-policies with an OR logic:
    1. transformer_auto_wrap_policy  – wraps modules whose *class* is listed
       in ``model._no_split_modules`` (e.g. ``GemmaDecoderLayer``).
    2. _module_wrap_policy           – wraps ``ValueHead`` (if present) so its
       float32 params stay in their own FSDP unit.
    3. lambda_auto_wrap_policy       – wraps modules whose ``_fsdp_wrap_name``
       is listed in ``model._no_split_names`` (e.g. ``action_in_proj``).

    All built-in PyTorch wrap policies correctly handle the ``recurse`` flag:
    they return True for ``recurse=True`` (always recurse into children) and
    only decide whether to *wrap* when ``recurse=False``.
    """
    policies: list = []

    # --- 1) Transformer layer classes from _no_split_modules ---
    transformer_cls_names = getattr(model, "_no_split_modules", None)
    if transformer_cls_names:
        transformer_cls_to_wrap: set[type] = set()
        for cls_name in transformer_cls_names:
            cls = get_module_class_from_name(model, cls_name)
            if cls is not None:
                transformer_cls_to_wrap.add(cls)
            else:
                logger.warning(
                    "Could not resolve transformer layer class %r in model", cls_name
                )
        if transformer_cls_to_wrap:
            policies.append(
                functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=transformer_cls_to_wrap,
                )
            )

    # --- 2) ValueHead as its own FSDP unit (float32, avoids dtype mixing) ---
    if hasattr(model, "value_head"):
        from vlarlkit.models.modules.value_head import ValueHead

        policies.append(
            functools.partial(_module_wrap_policy, module_classes={ValueHead})
        )

    # --- 3) Modules identified by _fsdp_wrap_name in _no_split_names ---
    no_split_names = getattr(model, "_no_split_names", None)
    if no_split_names:

        def _name_policy_fn(module: torch.nn.Module) -> bool:
            return (
                hasattr(module, "_fsdp_wrap_name")
                and module._fsdp_wrap_name in no_split_names
            )

        policies.append(
            functools.partial(lambda_auto_wrap_policy, lambda_fn=_name_policy_fn)
        )

    if not policies:
        return None
    if len(policies) == 1:
        return policies[0]
    return functools.partial(_or_policy, policies=policies)


# ──────────────────────────── helpers ────────────────────────────


SHARDING_STRATEGIES = {
    "no_shard": ShardingStrategy.NO_SHARD,
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
}


def get_sharding_strategy(name: str) -> ShardingStrategy:
    name = (name or "no_shard").strip().lower()
    if name not in SHARDING_STRATEGIES:
        raise ValueError(
            f"Unknown sharding strategy: {name!r}. "
            f"Choose from {list(SHARDING_STRATEGIES)}"
        )
    return SHARDING_STRATEGIES[name]


def _resolve_dtype(val) -> torch.dtype | None:
    """Convert a config value (str / None) to a torch.dtype."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("none", "null", ""):
        return None
    mapping = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if s not in mapping:
        raise ValueError(f"Unsupported dtype string: {val!r}")
    return mapping[s]


def mixed_precision_from_cfg(mp_cfg: dict | None) -> MixedPrecision | None:
    if not mp_cfg:
        return None
    param_dtype = _resolve_dtype(mp_cfg.get("param_dtype"))
    reduce_dtype = _resolve_dtype(mp_cfg.get("reduce_dtype"))
    buffer_dtype = _resolve_dtype(mp_cfg.get("buffer_dtype"))
    if param_dtype is None and reduce_dtype is None and buffer_dtype is None:
        return None
    return MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
    )


# ──────────────────────────── entry point ────────────────────────────


def wrap_model_with_fsdp(model: BaseModel, fsdp_cfg: dict, rank: int) -> FSDP:
    wrap_policy = get_fsdp_wrap_policy(model)

    sharding_strategy = get_sharding_strategy(
        fsdp_cfg.get("sharding_strategy", "no_shard")
    )

    fsdp_kwargs: dict[str, Any] = {
        "sharding_strategy": sharding_strategy,
        "auto_wrap_policy": wrap_policy,
        "device_id": rank,
    }

    mp_cfg = fsdp_cfg.get("mixed_precision")
    if isinstance(mp_cfg, dict):
        mp = mixed_precision_from_cfg(mp_cfg)
        if mp is not None:
            fsdp_kwargs["mixed_precision"] = mp

    return FSDP(model, **fsdp_kwargs)
