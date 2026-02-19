"""Data sharding utilities for distributed training: flatten batch by sample, shard/broadcast across ranks."""

from typing import Any

import torch
import torch.distributed as dist


def _flatten_batch(batch: dict) -> tuple[list, list]:
    """Flatten batch to deterministic list of (path, shape, dtype). Returns (meta_list, tensor_list)."""
    meta_list = []
    tensor_list = []

    def recurse(obj: Any, path: tuple) -> None:
        if isinstance(obj, dict):
            for k in sorted(obj.keys()):
                recurse(obj[k], path + (k,))
        elif torch.is_tensor(obj):
            meta_list.append((path, tuple(obj.shape), obj.dtype))
            tensor_list.append(obj)
        else:
            raise TypeError(f"Unsupported type {type(obj)} at path {path}")

    recurse(batch, ())
    return meta_list, tensor_list


def _empty_batch_from_meta(meta_list: list, device: torch.device) -> tuple[dict, list]:
    """Build batch skeleton and flat list of tensors from meta_list (path, shape, dtype_str)."""
    batch = {}
    tensor_list = []

    def set_nested(d: dict, path: tuple, value: Any) -> None:
        for k in path[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[path[-1]] = value

    for path, shape, dtype in meta_list:
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.replace("torch.", ""), torch.float32)
        t = torch.empty(shape, dtype=dtype, device=device)
        set_nested(batch, path, t)
        tensor_list.append(t)

    return batch, tensor_list


def _to_device_recursive(obj: Any, device: torch.device) -> Any:
    """Move nested dict of tensors to device."""
    if isinstance(obj, dict):
        return {k: _to_device_recursive(v, device) for k, v in obj.items()}
    if torch.is_tensor(obj):
        return obj.to(device)
    return obj


def _flatten_batch_by_sample(batch: dict) -> tuple[list, list]:
    """Flatten batch so each tensor has leading dim N = product of first two dims.
    Returns (meta_list, tensor_list) with meta_list items (path, shape=(N,...), dtype).
    """
    meta_list = []
    tensor_list = []

    def recurse(obj: Any, path: tuple) -> None:
        if isinstance(obj, dict):
            for k in sorted(obj.keys()):
                recurse(obj[k], path + (k,))
        elif torch.is_tensor(obj):
            shp = tuple(obj.shape)
            if len(shp) < 2:
                raise TypeError(f"Expected at least 2 dims at path {path}, got shape {shp}")
            N = shp[0] * shp[1]
            new_shape = (N,) + shp[2:]
            meta_list.append((path, new_shape, obj.dtype))
            tensor_list.append(obj.reshape(new_shape))
        else:
            raise TypeError(f"Unsupported type {type(obj)} at path {path}")

    recurse(batch, ())
    return meta_list, tensor_list


def broadcast_batch(batch: dict, src: int = 0, device: torch.device | None = None) -> dict:
    """Broadcast a nested dict of tensors from src rank to all ranks."""
    if device is None:
        device = torch.device("cuda")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size == 1:
        return _to_device_recursive(batch, device)

    if rank == src:
        meta_list, tensor_list = _flatten_batch(batch)
        meta_for_send = [(p, s, str(t.dtype).replace("torch.", "")) for p, s, t in meta_list]
    else:
        meta_for_send = [None]

    dist.broadcast_object_list(meta_for_send, src=src)
    if rank != src:
        meta_list = meta_for_send[0]
        batch, tensor_list = _empty_batch_from_meta(meta_list, device)
    else:
        tensor_list = [t.to(device) for t in tensor_list]

    for t in tensor_list:
        dist.broadcast(t, src=src)

    return batch


def shard_batch(batch: dict, src: int = 0, device: torch.device | None = None) -> dict:
    """Shard batch from src rank to all ranks by sample dimension. Each rank gets N_local = N_total // world_size samples. If N_total < world_size, fallback to broadcast."""
    if device is None:
        device = torch.device("cuda")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size == 1:
        return _to_device_recursive(batch, device)

    if rank == src:
        meta_list, full_tensor_list = _flatten_batch_by_sample(batch)
        N_total = full_tensor_list[0].shape[0]
        if N_total < world_size:
            return broadcast_batch(batch, src=src, device=device)
        N_local = N_total // world_size
        N_used = N_local * world_size
        meta_for_send = [
            (p, list(s), str(t.dtype).replace("torch.", ""))
            for (p, s, t) in meta_list
        ]
        meta_len_list = [len(meta_for_send)]
    else:
        meta_len_list = [None]
        meta_for_send = None
        N_total = None
        N_local = None
        N_used = None
        full_tensor_list = None

    dist.broadcast_object_list(meta_len_list, src=src)
    if rank != src:
        meta_for_send = [None] * meta_len_list[0]
    dist.broadcast_object_list(meta_for_send, src=src)
    if rank != src:
        meta_list = [(p, tuple(s), d) for (p, s, d) in meta_for_send]
        N_total = meta_list[0][1][0]
        N_local = N_total // world_size
        N_used = N_local * world_size

    meta_shard = [(path, (N_local,) + tuple(s[1:]), dtype) for (path, s, dtype) in meta_list]
    batch_shard, tensor_list_shard = _empty_batch_from_meta(meta_shard, device)

    for i in range(len(tensor_list_shard)):
        if rank == src:
            chunked = full_tensor_list[i][:N_used].chunk(world_size, dim=0)
            scatter_list = list(chunked)
        else:
            scatter_list = None
        dist.scatter(tensor_list_shard[i], scatter_list, src=src)

    for key in ("advantages", "prev_logprobs", "prev_values", "returns"):
        if key in batch_shard and torch.is_tensor(batch_shard[key]) and batch_shard[key].dim() == 1:
            batch_shard[key] = batch_shard[key].unsqueeze(-1)

    return batch_shard
