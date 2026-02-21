import numpy as np
import torch
from typing import Any


def to_numpy(x: Any) -> Any:
    """Convert a torch.Tensor or numpy array to numpy, no-op for other types."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    return x


def to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move nested dict of tensors to *device*."""
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if torch.is_tensor(obj):
        return obj.to(device)
    return obj