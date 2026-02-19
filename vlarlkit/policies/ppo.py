import torch
import torch.distributed as dist

from omegaconf import DictConfig
from typing import Any

from vlarlkit.models.base import BaseModel
from vlarlkit.utils.fsdp_utils import wrap_model_with_fsdp


class PPOPolicy:
    """
    PPO policy with FSDP for single-machine multi-GPU training.
    """

    def __init__(self, cfg: DictConfig, model: BaseModel, rank: int) -> None:
        self.cfg = cfg
        self.algo_cfg = cfg.algorithm
        self.optim_cfg = cfg.training.optim
        self.rank = rank
        self.device = torch.device(f"cuda:{self.rank}")
        self.fsdp_cfg = getattr(cfg.training, "fsdp_config", None) or {}

        self.model = wrap_model_with_fsdp(model, self.fsdp_cfg, rank)

        self._setup_optimizer()
        self._setup_lr_scheduler()

        self.clip_grad = float(self.optim_cfg.get("clip_grad", 0.0))
        self._global_step = 0

    def _setup_optimizer(self) -> None:
        value_lr = float(self.optim_cfg.get("value_lr"))
        lr = float(self.optim_cfg.get("lr"))
        beta1 = float(self.optim_cfg.get("adam_beta1", 0.9))
        beta2 = float(self.optim_cfg.get("adam_beta2", 0.999))
        eps = float(self.optim_cfg.get("adam_eps", 1e-8))
        weight_decay = float(self.optim_cfg.get("weight_decay", 0.0))

        if value_lr is not None:
            value_lr = float(value_lr)
            inner = self.model.module if hasattr(self.model, "module") else self.model
            value_params = []
            other_params = []
            for n, p in inner.named_parameters():
                if not p.requires_grad:
                    continue
                if "value_head" in n:
                    value_params.append(p)
                else:
                    other_params.append(p)
            if value_params:
                param_groups = [
                    {"params": other_params, "lr": lr},
                    {"params": value_params, "lr": value_lr},
                ]
                self.optimizer = torch.optim.AdamW(
                    param_groups, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay
                )
            else:
                params = [p for p in self.model.parameters() if p.requires_grad]
                self.optimizer = torch.optim.AdamW(
                    params,
                    lr=lr,
                    betas=(beta1, beta2),
                    eps=eps,
                    weight_decay=weight_decay,
                )
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay,
            )

    def _setup_lr_scheduler(self) -> None:
        sched_type = self.optim_cfg.get("lr_scheduler", "constant")
        total_steps = int(self.optim_cfg.get("total_training_steps", 100000))
        min_lr_rate = float(self.optim_cfg.get("min_lr_rate", 0.1))
        if sched_type == "cosine":
            def lr_lambda(step: int) -> float:
                if step >= total_steps:
                    return min_lr_rate
                import math
                return min_lr_rate + 0.5 * (1 - min_lr_rate) * (1 + math.cos(step / total_steps * math.pi))
            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda
            )
        elif sched_type == "constant":
            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda step: 1.0
            )
        else:
            raise ValueError(f"Invalid learning rate scheduler type: {sched_type}")

    def get_model(self) -> torch.nn.Module:
        return self.model

    def set_global_step(self, step: int) -> None:
        self._global_step = step
        inner = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(inner, "set_global_step"):
            inner.set_global_step(step)

    def _slice_batch(self, batch: dict, indices: torch.Tensor) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                out[k] = {
                    kk: vv[indices] if torch.is_tensor(vv) else vv
                    for kk, vv in v.items()
                }
            elif torch.is_tensor(v):
                out[k] = v[indices]
            else:
                out[k] = v
        return out

    def run_update(self, batch: dict[str, Any]) -> dict[str, float]:
        self.model.train()

        update_epochs = int(self.algo_cfg.get("update_epochs"))
        minibatch_size = max(
            1, int(self.algo_cfg.get("minibatch_size")) // dist.get_world_size()
        )
        clip_ratio_high = float(self.algo_cfg.get("clip_ratio_high", 0.2))
        clip_ratio_low = float(self.algo_cfg.get("clip_ratio_low", 0.2))
        value_clip = float(self.algo_cfg.get("value_clip", 0.2))
        huber_delta = float(self.algo_cfg.get("huber_delta", 10.0))
        entropy_bonus = float(self.algo_cfg.get("entropy_bonus", 0.0))

        advantages_flat = batch["advantages"]
        prev_logprobs_flat = batch["prev_logprobs"]
        prev_values_flat = batch["prev_values"]
        returns_flat = batch["returns"]
        forward_inputs_flat = batch["forward_inputs"]

        if prev_logprobs_flat.dim() > 1:
            prev_logprobs_flat = prev_logprobs_flat.sum(
                dim=tuple(range(1, prev_logprobs_flat.dim()))
            )

        N = advantages_flat.shape[0]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(update_epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, minibatch_size):
                end = min(start + minibatch_size, N)
                mb_inds = perm[start:end]

                mb_advantages = advantages_flat[mb_inds]
                mb_prev_logprobs = prev_logprobs_flat[mb_inds]
                mb_returns = returns_flat[mb_inds]
                mb_forward_inputs = self._slice_batch(forward_inputs_flat, mb_inds)

                out = self.model(
                    forward_inputs=mb_forward_inputs,
                    compute_values=True,
                )
                logprobs = out["logprobs"]
                values = out["values"]
                entropy = out["entropy"]

                if logprobs.dim() > 1:
                    logprobs = logprobs.sum(dim=tuple(range(1, logprobs.dim())))
                if values.dim() > 1:
                    values = values.squeeze(-1)
                entropy = entropy.reshape(-1).mean()

                ratio = torch.exp(logprobs - mb_prev_logprobs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                if value_clip > 0:
                    values_clipped = prev_values_flat[mb_inds] + torch.clamp(
                        values - prev_values_flat[mb_inds],
                        -value_clip,
                        value_clip,
                    )
                    value_target = values_clipped
                else:
                    value_target = values

                value_residual = value_target - mb_returns
                if huber_delta > 0:
                    value_loss = torch.where(
                        torch.abs(value_residual) <= huber_delta,
                        0.5 * value_residual**2,
                        huber_delta * (torch.abs(value_residual) - 0.5 * huber_delta),
                    ).mean()
                else:
                    value_loss = 0.5 * (value_residual**2).mean()

                loss = policy_loss + value_loss
                if entropy_bonus != 0:
                    loss = loss - entropy_bonus * entropy

                self.optimizer.zero_grad()
                loss.backward()
                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad
                    )
                self.optimizer.step()

                total_policy_loss += policy_loss.detach().item()
                total_value_loss += value_loss.detach().item()
                total_entropy += entropy.detach().item()
                num_updates += 1

        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        self._global_step += 1
        self.set_global_step(self._global_step)

        n = max(1, num_updates)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
        }
