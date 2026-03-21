import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

from vlarlkit.models.base import BaseModel
from vlarlkit.utils.fsdp_utils import (
    clip_grad_norm_,
    get_sharding_strategy,
    wrap_model_with_fsdp,
)

logger = logging.getLogger("vlarlkit.policy")


class RLTPolicy:
    """RLT (RL Token) policy — TD3-style, no alpha/entropy."""

    def __init__(
        self,
        cfg,
        model: BaseModel,
        target_model: BaseModel,
        rank: int,
    ) -> None:
        self.cfg = cfg
        self._algo_cfg = cfg.algorithm
        self._optim_cfg = cfg.training.optim
        self.rank = rank
        self.device = torch.device(f"cuda:{self.rank}")
        self._fsdp_cfg = getattr(cfg.training, "fsdp_config", None) or {}

        # FSDP wrap both models
        self.model = wrap_model_with_fsdp(model, self._fsdp_cfg, rank)
        self._target_model = wrap_model_with_fsdp(target_model, self._fsdp_cfg, rank)
        self._target_model.requires_grad_(False)
        self._sharding_strategy = get_sharding_strategy(
            self._fsdp_cfg.get("sharding_strategy", "no_shard")
        )

        self._critic_actor_ratio = int(self._algo_cfg.get("critic_actor_ratio", 2))
        self._gamma = float(self._algo_cfg.get("gamma", 0.999))
        self._tau = float(self._algo_cfg.get("tau", 0.005))
        self._agg_q = str(self._algo_cfg.get("agg_q", "min"))
        self._num_action_chunks = int(cfg.model.get("num_action_chunks", 1))
        self._bc_weight = float(self._algo_cfg.get("rlt_bc_weight", 1.0))

        # Gradient accumulation
        global_bs = self._algo_cfg.get("global_batch_size")
        micro_bs = self._algo_cfg.get("micro_batch_size")
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._gradient_accumulation = int(global_bs) // (int(micro_bs) * world_size)

        self._setup_optimizers()
        self._init_target_shadow()
        self._soft_update_target(tau=1.0)
        self._update_step = 0

    def _setup_optimizers(self) -> None:
        actor_lr = float(self._optim_cfg.get("lr", 1e-4))
        critic_lr = float(self._optim_cfg.get("value_lr", 3e-4))
        self._actor_clip_grad = float(self._optim_cfg.get("clip_grad", 3.5))
        self._critic_clip_grad = float(self._optim_cfg.get("critic_clip_grad", 10.0))

        params_actor, params_critic = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "q_head" in name or "critic_" in name:
                params_critic.append(param)
            else:
                params_actor.append(param)

        self._actor_optimizer = torch.optim.Adam(params_actor, lr=actor_lr)
        self._critic_optimizer = torch.optim.Adam(params_critic, lr=critic_lr)
        self._params_actor = params_actor
        self._params_critic = params_critic

        n_actor = sum(p.numel() for p in params_actor)
        n_critic = sum(p.numel() for p in params_critic)
        if self.rank == 0:
            logger.info(
                f"RLT params: actor={n_actor:,}, critic={n_critic:,}"
            )

    def _init_target_shadow(self):
        """Float32 shadow for EMA precision under bfloat16."""
        self._target_shadow_f32 = {}
        for name, param in self._target_model.named_parameters():
            self._target_shadow_f32[name] = param.data.float().clone()

    def process_batch_for_replay(self, batch: dict) -> dict:
        """Transform raw rollout batch into replay-ready data.

        Extracts rl_token/ref_actions from forward_inputs, filters by loss_mask.
        """
        mask = batch["loss_mask"].bool()
        return {
            "obs/rl_token": batch["forward_inputs"]["rl_token"][mask],
            "obs/states": batch["obs/states"][mask],
            "obs/ref_actions": batch["forward_inputs"]["ref_actions"][mask],
            "next_obs/rl_token": batch["next_forward_inputs"]["rl_token"][mask],
            "next_obs/states": batch["next_obs/states"][mask],
            "next_obs/ref_actions": batch["next_forward_inputs"]["ref_actions"][mask],
            "actions": batch["actions"][mask],
            "rewards": batch["rewards"][mask],
            "terminations": batch["terminations"][mask],
        }

    def run_update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """One TD3 update step with gradient accumulation."""
        self.model.train()

        micro_batches = self._split_micro_batches(batch)
        n_accum = len(micro_batches)

        # --- Critic update (every step) ---
        self._critic_optimizer.zero_grad()
        total_critic_loss = 0.0
        total_q_mean = 0.0
        for mb in micro_batches:
            obs, next_obs, actions, rewards, terminations = self._prepare_batch(mb)
            critic_loss, critic_metrics = self._compute_critic_loss(
                obs, next_obs, actions, rewards, terminations
            )
            (critic_loss / n_accum).backward()
            total_critic_loss += critic_loss.item()
            total_q_mean += critic_metrics["q_mean"]
        critic_grad_norm = clip_grad_norm_(
            self._params_critic, self._critic_clip_grad,
            sharding_strategy=self._sharding_strategy,
        )
        self._critic_optimizer.step()

        metrics: dict[str, float] = {
            "critic_loss": total_critic_loss / n_accum,
            "q_mean": total_q_mean / n_accum,
            "critic_grad_norm": critic_grad_norm,
        }

        # --- Actor update (delayed, every critic_actor_ratio steps) ---
        if self._update_step % self._critic_actor_ratio == 0:
            self._actor_optimizer.zero_grad()
            total_actor_loss = 0.0
            total_bc_loss = 0.0

            for mb in micro_batches:
                obs, _, _, _, _ = self._prepare_batch(mb)
                actor_loss, bc_loss = self._compute_actor_loss(obs)
                (actor_loss / n_accum).backward()
                total_actor_loss += actor_loss.item()
                total_bc_loss += bc_loss.item()

            actor_grad_norm = clip_grad_norm_(
                self._params_actor, self._actor_clip_grad,
                sharding_strategy=self._sharding_strategy,
            )
            self._actor_optimizer.step()

            metrics["actor_loss"] = total_actor_loss / n_accum
            metrics["bc_loss"] = total_bc_loss / n_accum
            metrics["actor_grad_norm"] = actor_grad_norm

        # --- Soft update target ---
        self._soft_update_target()
        self._update_step += 1

        return metrics

    def _split_micro_batches(
        self, batch: dict[str, np.ndarray]
    ) -> list[dict[str, np.ndarray]]:
        n_accum = self._gradient_accumulation
        if n_accum <= 1:
            return [batch]

        n = next(iter(batch.values())).shape[0]
        chunk_size = max(1, n // n_accum)
        micro_batches = []
        for i in range(0, n, chunk_size):
            mb = {k: v[i : i + chunk_size] for k, v in batch.items()}
            micro_batches.append(mb)
        return micro_batches

    def _compute_critic_loss(self, obs, next_obs, actions, rewards, terminations):
        """TD3 critic loss: MSE(Q(s,a), r + gamma^C * min Q_target(s', pi(s')))."""
        discount = self._gamma ** self._num_action_chunks

        with torch.no_grad():
            next_actions, _, _ = self.model.actor_forward(
                obs=next_obs, train=False, # should we use train=False?
            )
            target_q = self._target_model.critic_forward(
                obs=next_obs, actions=next_actions,
            )

            if self._agg_q == "min":
                qf_next, _ = torch.min(target_q, dim=1, keepdim=True)
            else:
                qf_next = torch.mean(target_q, dim=1, keepdim=True)

            target_q_values = (
                rewards.unsqueeze(-1)
                + (~terminations.unsqueeze(-1)) * discount * qf_next
            )

        current_q = self.model.critic_forward(
            obs=obs, actions=actions,
        )

        target_q_values = target_q_values.to(dtype=current_q.dtype)
        critic_loss = F.mse_loss(current_q, target_q_values.expand_as(current_q))

        return critic_loss, {"q_mean": current_q.mean().item()}

    def _compute_actor_loss(self, obs):
        """TD3 actor loss: -Q(s, pi(s)) + beta * ||pi(s) - ref_actions||^2."""
        actions, _, _ = self.model.actor_forward(
            obs=obs, train=True,
        )

        q_values = self.model.critic_forward(
            obs=obs, actions=actions,
        )

        if self._agg_q == "min":
            qf_pi, _ = torch.min(q_values, dim=1, keepdim=True)
        else:
            qf_pi = torch.mean(q_values, dim=1, keepdim=True)

        # BC regularization: penalize deviation from ref_actions
        ref_actions = obs["ref_actions"]
        if isinstance(ref_actions, np.ndarray):
            ref_actions = torch.from_numpy(ref_actions)
        ref_actions = ref_actions.to(device=actions.device, dtype=actions.dtype)
        if actions.dim() == 3:
            actions_flat = actions.squeeze(1)
        else:
            actions_flat = actions
        bc_loss = ((actions_flat - ref_actions) ** 2).sum(-1).mean()

        actor_loss = -qf_pi.mean() + self._bc_weight * bc_loss
        return actor_loss, bc_loss

    def _soft_update_target(self, tau: float | None = None) -> None:
        """EMA update target model using float32 shadow for bf16 precision."""
        if tau is None:
            tau = self._tau

        with torch.no_grad():
            for (name1, online_param), (name2, target_param) in zip(
                self.model.named_parameters(),
                self._target_model.named_parameters(),
            ):
                assert name1 == name2
                shadow = self._target_shadow_f32[name1]
                shadow.mul_(1.0 - tau).add_(online_param.data.float(), alpha=tau)
                target_param.data.copy_(shadow.to(target_param.data.dtype))

    def _prepare_batch(self, batch: dict[str, np.ndarray]):
        """Convert flat replay buffer batch to nested obs dicts on device."""
        def _build_obs_dict(prefix):
            obs_dict = {}
            for k, v in batch.items():
                if k.startswith(prefix + "/"):
                    key = k[len(prefix) + 1:]
                    t = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                    obs_dict[key] = t.to(self.device)
            return obs_dict

        obs = _build_obs_dict("obs")
        next_obs = _build_obs_dict("next_obs")

        raw_actions = batch["actions"]
        actions = torch.from_numpy(raw_actions) if isinstance(raw_actions, np.ndarray) else raw_actions
        actions = actions.to(self.device)

        raw_rewards = batch["rewards"]
        rewards = torch.from_numpy(raw_rewards).float() if isinstance(raw_rewards, np.ndarray) else raw_rewards.float()
        rewards = rewards.to(self.device)

        raw_term = batch["terminations"]
        terminations = torch.from_numpy(raw_term).bool() if isinstance(raw_term, np.ndarray) else raw_term.bool()
        terminations = terminations.to(self.device)

        return obs, next_obs, actions, rewards, terminations

    def state_dict(self) -> dict:
        return {
            "update_step": self._update_step,
            "target_shadow_f32": {k: v.cpu() for k, v in self._target_shadow_f32.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self._update_step = state["update_step"]
        self._target_shadow_f32 = {k: v.to(self.device) for k, v in state["target_shadow_f32"].items()}
        with torch.no_grad():
            for name, param in self._target_model.named_parameters():
                param.data.copy_(self._target_shadow_f32[name].to(param.dtype))

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_target_model(self) -> torch.nn.Module:
        return self._target_model
