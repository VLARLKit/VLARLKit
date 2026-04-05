import torch


def extract_termination_from_info(info, num_envs, device):
    if "success" in info:
        if "fail" in info:
            terminated = torch.logical_or(info["success"], info["fail"])
        else:
            terminated = info["success"].clone()
    else:
        if "fail" in info:
            terminated = info["fail"].clone()
        else:
            terminated = torch.zeros(num_envs, dtype=bool, device=device)
    return terminated


def recursive_to_own(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone() if obj.is_shared() else obj
    elif isinstance(obj, list):
        return [recursive_to_own(elem) for elem in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to_own(elem) for elem in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to_own(v) for k, v in obj.items()}
    else:
        return obj


def recursive_to_numpy(obj):
    """Recursively convert torch tensors to numpy arrays for ZMQ serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        return {k: recursive_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [recursive_to_numpy(item) for item in obj]
        return type(obj)(converted)
    else:
        return obj
