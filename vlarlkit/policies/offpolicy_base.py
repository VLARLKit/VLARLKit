import torch
from omegaconf import DictConfig
from typing import Any

from vlarlkit.models.base import BaseModel
from vlarlkit.utils.fsdp_utils import wrap_model_with_fsdp


class OffPolicyBase:
    """Off-policy algorithm interface (placeholder).

    Subclasses should implement concrete update logic (e.g. SAC, TD3).
    The FSDP wrapping and optimizer setup pattern follows PPOPolicy.
    """

    def __init__(self, cfg: DictConfig, model: BaseModel, rank: int) -> None:
        self.cfg = cfg
        self.rank = rank
        self.device = torch.device(f"cuda:{self.rank}")
        self.fsdp_cfg = getattr(cfg.training, "fsdp_config", None) or {}

        self.model = wrap_model_with_fsdp(model, self.fsdp_cfg, rank)

    def run_update(self, batch: dict[str, Any]) -> dict[str, float]:
        """Run one gradient update step on a sampled batch.

        Returns:
            A dict of scalar metrics for logging.
        """
        raise NotImplementedError

    def get_model(self) -> torch.nn.Module:
        return self.model
