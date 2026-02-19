"""Data sharding utilities for distributed training: flatten, broadcast, scatter nested dicts of tensors."""

from typing import Any

import torch
import torch.distributed as dist


# ── Helpers ──────────────────────────────────────────────────────────


def _flatten(batch: dict) -> list[tuple[tuple, torch.Tensor]]:
    """Recursively flatten a nested dict into a sorted list of (path, tensor) pairs."""
    items: list[tuple[tuple, torch.Tensor]] = []

    def _recurse(obj: Any, path: tuple) -> None:
        if isinstance(obj, dict):
            for k in sorted(obj.keys()):
                _recurse(obj[k], path + (k,))
        elif torch.is_tensor(obj):
            items.append((path, obj))
        else:
            raise TypeError(f"Unsupported type {type(obj)} at {path}")

    _recurse(batch, ())
    return items


def _unflatten(pairs: list[tuple[tuple, torch.Tensor]]) -> dict:
    """Reconstruct a nested dict from (path, tensor) pairs."""
    batch: dict = {}
    for path, tensor in pairs:
        d = batch
        for k in path[:-1]:
            d = d.setdefault(k, {})
        d[path[-1]] = tensor
    return batch


def _to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move nested dict of tensors to *device*."""
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if torch.is_tensor(obj):
        return obj.to(device)
    return obj


def _broadcast_objects(objects: list | None, src: int) -> list:
    """Broadcast a variable-length list of picklable objects from *src* to all ranks."""
    rank = dist.get_rank()
    n = [len(objects) if rank == src else 0]
    dist.broadcast_object_list(n, src=src)
    if rank != src:
        objects = [None] * n[0]
    dist.broadcast_object_list(objects, src=src)
    return objects


def _dtype_to_str(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def _str_to_dtype(s: str) -> torch.dtype:
    return getattr(torch, s.replace("torch.", ""), torch.float32)


# ── Public API ───────────────────────────────────────────────────────


def broadcast_batch(
    batch: dict, src: int = 0, device: torch.device | None = None
) -> dict:
    """Broadcast a nested dict of tensors from *src* rank to all ranks."""
    device = device or torch.device("cuda")
    rank = dist.get_rank()
    if dist.get_world_size() == 1:
        return _to_device(batch, device)

    if rank == src:
        items = _flatten(batch)
        meta = [(p, tuple(t.shape), _dtype_to_str(t.dtype)) for p, t in items]
        tensors = [t.to(device) for _, t in items]
    else:
        meta = None
        tensors = None

    meta = _broadcast_objects(meta, src)

    if rank != src:
        tensors = [
            torch.empty(s, dtype=_str_to_dtype(d), device=device)
            for _, s, d in meta
        ]

    for t in tensors:
        dist.broadcast(t, src=src)

    return _unflatten([(p, t) for (p, _, _), t in zip(meta, tensors)])


def shard_batch(
    batch: dict, src: int = 0, device: torch.device | None = None
) -> dict:
    """Scatter-shard a batch from *src* to all ranks along dim 0.

    All tensors in *batch* are expected to have a leading sample dimension N
    (already flattened by get_batch).  Each rank receives N // world_size
    samples.  Falls back to broadcast when N < world_size.
    """
    device = device or torch.device("cuda")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size == 1:
        return _to_device(batch, device)

    # ── src: flatten dict & read N ──
    if rank == src:
        items = _flatten(batch)
        n_total = items[0][1].shape[0]
        fallback = int(n_total < world_size)
    else:
        items = None
        fallback = 0

    # ── sync fallback decision ──
    flag = [fallback]
    dist.broadcast_object_list(flag, src=src)
    if flag[0]:
        return broadcast_batch(batch, src=src, device=device)

    # ── broadcast tensor metadata ──
    if rank == src:
        n_local = n_total // world_size
        n_used = n_local * world_size
        meta = [
            (p, list(t.shape), _dtype_to_str(t.dtype))
            for p, t in items
        ]
    else:
        n_local = n_used = 0
        meta = None

    meta = _broadcast_objects(meta, src)

    if rank != src:
        n_total = meta[0][1][0]
        n_local = n_total // world_size
        n_used = n_local * world_size

    # ── scatter each tensor along dim 0 ──
    shard_pairs: list[tuple[tuple, torch.Tensor]] = []
    for i, (path, shape, dtype_str) in enumerate(meta):
        recv = torch.empty(
            n_local, *shape[1:], dtype=_str_to_dtype(dtype_str), device=device
        )
        if rank == src:
            scatter_list = list(items[i][1][:n_used].to(device).chunk(world_size))
        else:
            scatter_list = None
        dist.scatter(recv, scatter_list, src=src)
        shard_pairs.append((path, recv))

    return _unflatten(shard_pairs)
