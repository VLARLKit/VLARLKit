import json
import random
from pathlib import Path

import numpy as np
import torch

from vlarlkit.models import get_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def reference_path(cfg):
    return str(Path(cfg.runner.actor_checkpoint or cfg.model.model_path).resolve())


def load_actor(cfg, device):
    assert cfg.runner.actor_checkpoint or int(cfg.runner.iteration) == 1
    model = get_model(cfg.model).requires_grad_(False).to(device).eval()
    state = None
    if cfg.runner.actor_checkpoint:
        state = torch.load(cfg.runner.actor_checkpoint, map_location='cpu', weights_only=False)
        assert state['completed'], 'The next round requires a completed actor checkpoint'
        assert state['iteration'] == int(cfg.runner.iteration) - 1
        model.load_actor_state_dict(state['actor_state_dict'])
    return model, state
