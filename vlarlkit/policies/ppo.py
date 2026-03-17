import contextlib
import logging
import torch
import torch.distributed as dist

from omegaconf import DictConfig
from typing import Any

from vlarlkit.models.base import BaseModel
from vlarlkit.utils.conversion_utils import to_device
from vlarlkit.utils.fsdp_utils import clip_grad_norm_, wrap_model_with_fsdp

logger = logging.getLogger("vlarlkit.policy")


class PPOPolicy:
    """
    PPO policy with FSDP for single-machine multi-GPU training.
    """

    def __init__(self, cfg: DictConfig, model: BaseModel, rank: int) -> None:
        self.cfg = cfg
        self._algo_cfg = cfg.algorithm
        self._optim_cfg = cfg.training.optim
        self.rank = rank
        self.device = torch.device(f"cuda:{self.rank}")
        self.fsdp_cfg = getattr(cfg.training, "fsdp_config", None) or {}

        self.model = wrap_model_with_fsdp(model, self.fsdp_cfg, rank)

        self._setup_optimizer()
        self._setup_lr_scheduler()

        self._clip_grad = float(self._optim_cfg.get("clip_grad", 0.0))
        self._global_step = 0

    def _setup_optimizer(self) -> None:
        value_lr = float(self._optim_cfg.get("value_lr"))
        lr = float(self._optim_cfg.get("lr"))
        beta1 = float(self._optim_cfg.get("adam_beta1", 0.9))
        beta2 = float(self._optim_cfg.get("adam_beta2", 0.999))
        eps = float(self._optim_cfg.get("adam_eps", 1e-8))
        weight_decay = float(self._optim_cfg.get("weight_decay", 0.0))

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
                self._optimizer = torch.optim.AdamW(
                    param_groups, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay
                )
            else:
                params = [p for p in self.model.parameters() if p.requires_grad]
                self._optimizer = torch.optim.AdamW(
                    params,
                    lr=lr,
                    betas=(beta1, beta2),
                    eps=eps,
                    weight_decay=weight_decay,
                )
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self._optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay,
            )

        n_optim = sum(p.numel() for group in self._optimizer.param_groups for p in group["params"])
        if self.rank == 0:
            logger.info(f"Optimizable params: {n_optim:,}")

    def _setup_lr_scheduler(self) -> None:
        sched_type = self._optim_cfg.get("lr_scheduler", "constant")
        total_steps = int(self._optim_cfg.get("total_training_steps", 100000))
        min_lr_rate = float(self._optim_cfg.get("min_lr_rate", 0.1))
        if sched_type == "cosine":
            def lr_lambda(step: int) -> float:
                if step >= total_steps:
                    return min_lr_rate
                import math
                return min_lr_rate + 0.5 * (1 - min_lr_rate) * (1 + math.cos(step / total_steps * math.pi))
            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._optimizer, lr_lambda
            )
        elif sched_type == "constant":
            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._optimizer, lr_lambda=lambda step: 1.0
            )
        else:
            raise ValueError(f"Invalid learning rate scheduler type: {sched_type}")

    def state_dict(self) -> dict:
        return {
            "global_step": self._global_step,
            "lr_scheduler": self._lr_scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._global_step = state["global_step"]
        self.set_global_step(self._global_step)
        self._lr_scheduler.load_state_dict(state["lr_scheduler"])

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

        update_epochs = int(self._algo_cfg.get("update_epochs"))
        # Derive per-rank mini-batch size and gradient accumulation
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        global_mini_bs = self._algo_cfg.get("global_mini_batch_size")
        micro_bs = self._algo_cfg.get("micro_batch_size")
        micro_batch_size = int(micro_bs)
        gradient_accumulation_steps = int(global_mini_bs) // (micro_batch_size * world_size)

        clip_ratio_high = float(self._algo_cfg.get("clip_ratio_high", 0.2))
        clip_ratio_low = float(self._algo_cfg.get("clip_ratio_low", 0.2))
        clip_ratio_c = float(self._algo_cfg.get("clip_ratio_c", 0.0))
        value_clip = float(self._algo_cfg.get("value_clip", 0.2))
        huber_delta = float(self._algo_cfg.get("huber_delta", 10.0))
        entropy_bonus = float(self._algo_cfg.get("entropy_bonus", 0.0))

        advantages = batch["advantages"] # (batch_size,)
        prev_logprobs = batch["prev_logprobs"] # (batch_size, action_chunk, action_dim)
        prev_values = batch["prev_values"] # (batch_size, 1)
        returns = batch["returns"] # (batch_size,)
        forward_inputs = batch["forward_inputs"]
        loss_mask = batch.get("loss_mask", None)  # (batch_size,) or None
        loss_mask_ratio = batch.get("loss_mask_ratio", None)  # (batch_size,) or None

        if prev_logprobs.dim() > 1:
            prev_logprobs = prev_logprobs.sum(
                dim=tuple(range(1, prev_logprobs.dim())) # (batch_size,)
            )
        if prev_values.dim() > 1:
            prev_values = prev_values.squeeze(-1) # (batch_size,)

        N = advantages.shape[0]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_value_mean = 0.0
        num_minibatches = 0

        for _ in range(update_epochs):
            perm = torch.randperm(N)
            self._optimizer.zero_grad()
            accum_step = 0

            for start in range(0, N, micro_batch_size):
                end = min(start + micro_batch_size, N)
                mb_inds = perm[start:end]

                mb_advantages = advantages[mb_inds].to(self.device)
                mb_prev_logprobs = prev_logprobs[mb_inds].to(self.device)
                mb_prev_values = prev_values[mb_inds].to(self.device)
                mb_returns = returns[mb_inds].to(self.device)
                mb_forward_inputs = to_device(
                    self._slice_batch(forward_inputs, mb_inds), self.device
                )
                mb_mask = loss_mask[mb_inds].to(self.device) if loss_mask is not None else None
                mb_ratio = loss_mask_ratio[mb_inds].to(self.device) if loss_mask_ratio is not None else None

                out = self.model(
                    forward_inputs=mb_forward_inputs,
                    compute_values=True,
                )
                logprobs = out["logprobs"] # (batch_size, action_chunk, action_dim)
                values = out["values"] # (batch_size,)
                entropy = out["entropy"] # (batch_size, 1)

                if logprobs.dim() > 1:
                    logprobs = logprobs.sum(dim=tuple(range(1, logprobs.dim()))) # (batch_size,)
                if values.dim() > 1:
                    values = values.squeeze(-1) # (batch_size,)

                ratio = torch.exp(logprobs - mb_prev_logprobs)
                policy_loss1 = -mb_advantages * ratio
                policy_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
                )
                per_sample_policy_loss = torch.max(policy_loss1, policy_loss2)

                if clip_ratio_c > 1.0:
                    dual_clip_bound = clip_ratio_c * torch.abs(mb_advantages)
                    per_sample_policy_loss = torch.min(
                        per_sample_policy_loss, dual_clip_bound
                    )

                def _value_loss(residual: torch.Tensor) -> torch.Tensor:
                    if huber_delta > 0:
                        return torch.where(
                            torch.abs(residual) <= huber_delta,
                            0.5 * residual**2,
                            huber_delta * (torch.abs(residual) - 0.5 * huber_delta),
                        )
                    return 0.5 * (residual**2)

                if value_clip > 0:
                    values_clipped = mb_prev_values + torch.clamp(
                        values - mb_prev_values,
                        -value_clip,
                        value_clip,
                    )
                    per_sample_value_loss = torch.max(
                        _value_loss(values - mb_returns),
                        _value_loss(values_clipped - mb_returns),
                    )
                else:
                    per_sample_value_loss = _value_loss(values - mb_returns)

                per_sample_entropy = entropy.reshape(-1)

                if mb_mask is not None and mb_ratio is not None:
                    # masked_mean_ratio: up-weight short (successful) episodes
                    policy_loss = (per_sample_policy_loss / mb_ratio * mb_mask).mean()
                    value_loss = (per_sample_value_loss / mb_ratio * mb_mask).mean()
                    entropy_val = (per_sample_entropy / mb_ratio * mb_mask).mean()
                    value_mean = (values.detach() / mb_ratio * mb_mask).mean()
                elif mb_mask is not None:
                    mask_sum = mb_mask.sum().clamp(min=1)
                    policy_loss = (per_sample_policy_loss * mb_mask).sum() / mask_sum
                    value_loss = (per_sample_value_loss * mb_mask).sum() / mask_sum
                    entropy_val = (per_sample_entropy * mb_mask).sum() / mask_sum
                    value_mean = (values.detach() * mb_mask).sum() / mask_sum
                else:
                    policy_loss = per_sample_policy_loss.mean()
                    value_loss = per_sample_value_loss.mean()
                    entropy_val = per_sample_entropy.mean()
                    value_mean = values.detach().mean()

                loss = policy_loss + value_loss
                if entropy_bonus != 0:
                    loss = loss - entropy_bonus * entropy_val

                accum_step += 1
                should_sync = (accum_step == gradient_accumulation_steps) or (end >= N)
                sync_context = contextlib.nullcontext() if should_sync else self.model.no_sync()
                with sync_context:
                    (loss / gradient_accumulation_steps).backward()

                if should_sync:
                    if self._clip_grad > 0:
                        clip_grad_norm_(self.model, self._clip_grad)
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                    accum_step = 0

                total_policy_loss += policy_loss.detach().item()
                total_value_loss += value_loss.detach().item()
                total_entropy += entropy_val.detach().item()
                total_value_mean += value_mean.item()
                num_minibatches += 1

        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        self._global_step += 1
        self.set_global_step(self._global_step)

        n = max(1, num_minibatches)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "value_mean": total_value_mean / n,
        }
