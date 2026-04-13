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


from vlarlkit.models.modules.value_head import aggregate_q

logger = logging.getLogger("vlarlkit.policy")


class DSRLPolicy:
    """DSRL policy with FSDP for multi-GPU training"""

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

        self._critic_actor_ratio = int(self._algo_cfg.get("critic_actor_ratio", 1))
        self._gamma = float(self._algo_cfg.get("gamma", 0.999))
        self._tau = float(self._algo_cfg.get("tau", 0.005))
        self._backup_entropy = bool(self._algo_cfg.get("backup_entropy", False))
        self._agg_q = str(self._algo_cfg.get("agg_q", "mean"))

        # Derive gradient accumulation from global/micro batch sizes
        global_bs = self._algo_cfg.get("global_batch_size")
        micro_bs = self._algo_cfg.get("micro_batch_size")
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._gradient_accumulation = int(global_bs) // (int(micro_bs) * world_size)

        self._setup_optimizers()
        self._setup_alpha()
        self._init_ema_params()
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
                f"DSRL params: actor={n_actor:,}, critic={n_critic:,}"
            )

    def _setup_alpha(self) -> None:
        auto_entropy = bool(self._algo_cfg.get("auto_entropy_tuning", True))
        initial_alpha = float(self._algo_cfg.get("initial_alpha", 1.0))

        # softplus parameterization: alpha = softplus(base_alpha)
        # inverse softplus to get base_alpha from desired initial_alpha
        self._base_alpha = torch.nn.Parameter(
            torch.tensor(
                np.log(np.exp(initial_alpha) - 1.0),
                dtype=torch.float32,
                device=self.device,
            )
        )

        if auto_entropy:
            self._target_entropy = float(
                self._algo_cfg.get("target_entropy", -16)
            )
            alpha_lr = float(self._algo_cfg.get("alpha_lr", 3e-4))
            self._alpha_clip_grad = float(self._algo_cfg.get("alpha_clip_grad", 10.0))
            self._alpha_optimizer = torch.optim.Adam(
                [self._base_alpha], lr=alpha_lr
            )
        else:
            self._alpha_optimizer = None
            self._alpha_clip_grad = 0.0

    @property
    def _alpha(self) -> torch.Tensor:
        return F.softplus(self._base_alpha)

    def _init_ema_params(self):
        """Record trainable param names and sync target model once."""
        self._ema_param_names = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._ema_param_names.add(name)

        # Initial hard copy: sync all params (frozen + trainable) once
        with torch.no_grad():
            for (name, online_p), (_, target_p) in zip(
                self.model.named_parameters(),
                self._target_model.named_parameters(),
            ):
                target_p.data.copy_(online_p.data)

    def process_batch_for_replay(self, batch: dict) -> dict:
        """Transform raw rollout batch into replay-ready data.

        Extracts noise actions from forward_inputs, filters by loss_mask.
        """
        mask = batch["loss_mask"].bool()
        return {
            "obs/main_images": batch["obs/main_images"][mask],
            "obs/states": batch["obs/states"][mask],
            "next_obs/main_images": batch["next_obs/main_images"][mask],
            "next_obs/states": batch["next_obs/states"][mask],
            "actions": batch["forward_inputs"]["action"][mask],
            "rewards": batch["rewards"][mask],
            "terminations": batch["terminations"][mask],
        }

    def run_update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """One SAC update step with gradient accumulation."""
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

        # --- Actor + Alpha update (every critic_actor_ratio steps, after warmup) ---
        if self._update_step % self._critic_actor_ratio == 0:
            self._actor_optimizer.zero_grad()
            if self._alpha_optimizer is not None:
                self._alpha_optimizer.zero_grad()

            total_actor_loss = 0.0
            total_entropy = 0.0
            total_alpha_loss = 0.0

            for mb in micro_batches:
                obs, _, _, _, _ = self._prepare_batch(mb)
                actor_loss, entropy, log_pi = self._compute_actor_loss(obs)
                (actor_loss / n_accum).backward()
                total_actor_loss += actor_loss.item()
                total_entropy += entropy.item()

                # Alpha gradient accumulation (reuse log_pi, detached)
                if self._alpha_optimizer is not None:
                    alpha_loss = self._compute_alpha_loss(log_pi.detach())
                    (alpha_loss / n_accum).backward()
                    total_alpha_loss += alpha_loss.item()

            actor_grad_norm = clip_grad_norm_(
                self._params_actor, self._actor_clip_grad,
                sharding_strategy=self._sharding_strategy,
            )
            self._actor_optimizer.step()

            metrics["actor_loss"] = total_actor_loss / n_accum
            metrics["entropy"] = total_entropy / n_accum
            metrics["actor_grad_norm"] = actor_grad_norm

            # --- Alpha optimizer step ---
            if self._alpha_optimizer is not None:
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(
                        self._base_alpha.grad, op=dist.ReduceOp.AVG
                    )
                clip_grad_norm_([self._base_alpha], self._alpha_clip_grad)
                self._alpha_optimizer.step()
                metrics["alpha_loss"] = total_alpha_loss / n_accum

            metrics["alpha"] = self._alpha.item()

        # --- Soft update target ---
        self._soft_update_target()
        self._update_step += 1

        return metrics

    def _split_micro_batches(
        self, batch: dict[str, np.ndarray]
    ) -> list[dict[str, np.ndarray]]:
        """Split a batch into micro-batches for gradient accumulation."""
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
        """Compute critic loss: MSE(Q(s,a), r + gamma * (min Q_target(s', pi(s')) - alpha * log_pi))."""
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.model(
                forward_type="actor", obs=next_obs, aug_img=False,
            )
            if next_log_pi.ndim == 1:
                next_log_pi = next_log_pi.unsqueeze(-1)

            target_q = self._target_model(
                forward_type="critic", obs=next_obs, actions=next_actions, aug_img=False,
            )

            qf_next = aggregate_q(target_q, self._agg_q)

            if self._backup_entropy:
                qf_next = qf_next - self._alpha * next_log_pi

            target_q_values = (
                rewards.unsqueeze(-1)
                + (~terminations.unsqueeze(-1)) * self._gamma * qf_next
            )

        current_q = self.model(
            forward_type="critic", obs=obs, actions=actions, aug_img=False,
        )

        target_q_values = target_q_values.to(dtype=current_q.dtype)
        critic_loss = F.mse_loss(current_q, target_q_values.expand_as(current_q))

        return critic_loss, {"q_mean": current_q.mean().item()}

    def _compute_actor_loss(self, obs):
        """Compute actor loss and return log_pi for alpha reuse."""
        pi, log_pi, q_values = self.model(
            forward_type="actor_critic", obs=obs, aug_img=False,
            detach_encoder=True,
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)

        qf_pi = aggregate_q(q_values, self._agg_q)

        actor_loss = (self._alpha.detach() * log_pi - qf_pi).mean()
        entropy = -log_pi.mean()
        return actor_loss, entropy, log_pi

    def _compute_alpha_loss(self, log_pi: torch.Tensor) -> torch.Tensor:
        """Compute alpha loss from cached (detached) log_pi."""
        return -self._alpha * (log_pi.mean() + self._target_entropy)

    def _soft_update_target(self, tau: float | None = None) -> None:
        """EMA update only trainable params (MLP heads) in target model."""
        if tau is None:
            tau = self._tau

        with torch.no_grad():
            for (name, online_p), (_, target_p) in zip(
                self.model.named_parameters(),
                self._target_model.named_parameters(),
            ):
                if name not in self._ema_param_names:
                    continue
                target_p.data.mul_(1.0 - tau).add_(online_p.data, alpha=tau)

    def _prepare_batch(self, batch: dict[str, np.ndarray]):
        """Convert flat replay buffer batch to nested obs dicts on device.

        Expected keys: "obs/main_images", "obs/states", "next_obs/main_images",
        "next_obs/states", "actions", "rewards", "terminations".
        """
        def _build_obs_dict(prefix):
            obs_dict = {}
            for k, v in batch.items():
                if k.startswith(prefix + "/"):
                    key = k[len(prefix) + 1:]
                    t = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                    obs_dict[key] = t.to(self.device)
            # Convert to format expected by actor_forward: {"images": [...], "states": ...}
            result = {"states": obs_dict["states"]}
            if "main_images" in obs_dict:
                result["images"] = [obs_dict["main_images"]]
            return result

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
            "base_alpha": self._base_alpha.data.cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._update_step = state["update_step"]
        self._base_alpha.data.copy_(state["base_alpha"].to(self.device))

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_target_model(self) -> torch.nn.Module:
        return self._target_model
