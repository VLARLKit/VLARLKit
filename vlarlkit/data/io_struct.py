import numpy as np
import torch

from dataclasses import dataclass, field
from typing import Any, Optional

from vlarlkit.utils.conversion_utils import to_numpy


@dataclass(kw_only=True)
class RolloutResult:
    """
    Results of multiple-epoch rollouts.
    """

    obs: list[dict[str, Any]] = field(default_factory=list)
    next_obs: list[dict[str, Any]] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[np.ndarray] = field(default_factory=list)
    terminations: list[np.ndarray] = field(default_factory=list)
    truncations: list[np.ndarray] = field(default_factory=list)
    prev_logprobs: list[np.ndarray] = field(default_factory=list)
    prev_values: list[np.ndarray] = field(default_factory=list)
    forward_inputs: list[dict[str, Any]] = field(default_factory=list)
    next_forward_inputs: list[dict[str, Any]] = field(default_factory=list)

    returns: np.ndarray | None = field(default=None, repr=False)
    advantages: np.ndarray | None = field(default=None, repr=False)

    def clear(self) -> None:
        """Clear all list fields and reset returns/advantages for a new rollout."""
        self.obs.clear()
        self.next_obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.terminations.clear()
        self.truncations.clear()
        self.prev_logprobs.clear()
        self.prev_values.clear()
        self.forward_inputs.clear()
        self.next_forward_inputs.clear()
        self.returns = None
        self.advantages = None

    def append_step(
        self,
        obs: Any,
        next_obs: Any,
        actions: Any,
        rewards: Any,
        terminations: Any,
        truncations: Any,
        prev_logprobs: Optional[Any] = None,
        prev_values: Optional[Any] = None,
        forward_inputs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.obs.append(obs)
        self.next_obs.append(next_obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.terminations.append(terminations)
        self.truncations.append(truncations)
        if prev_logprobs is not None:
            self.prev_logprobs.append(to_numpy(prev_logprobs))
        if prev_values is not None:
            self.prev_values.append(to_numpy(prev_values))
        if forward_inputs is not None:
            self.forward_inputs.append(to_numpy(forward_inputs))

    def build_next_forward_inputs(self):
        """Pure temporal shift: next_fi[t] = fi[t+1]. Requires len(forward_inputs) == T+1."""
        T = len(self.obs)
        assert len(self.forward_inputs) == T + 1, (
            f"Expected {T + 1} forward_inputs, got {len(self.forward_inputs)}"
        )
        self.next_forward_inputs = [self.forward_inputs[t + 1] for t in range(T)]

    def compute_returns_and_advantages(
        self,
        gamma: float,
        gae_lambda: float,
    ):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).

        Bootstrap is handled upstream by embedding gamma * V(next_obs) into
        the rewards at episode boundaries (see Rollout._get_bootstrap_values).

        Args:
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter controlling the bias-variance trade-off.
        """
        num_steps = len(self.rewards)
        rewards = np.stack(self.rewards).astype(np.float32)         # (T, n_envs)
        values = np.stack(self.prev_values).astype(np.float32)      # (T, n_envs, 1)
        values = values.squeeze(-1)                                 # (T, n_envs)
        terminations = np.stack(self.terminations)                  # (T, n_envs)
        truncations = np.stack(self.truncations)                    # (T, n_envs)

        dones = np.logical_or(terminations, truncations)

        advantages = np.zeros_like(rewards)
        last_gae_lam = np.zeros_like(rewards[0])

        for t in reversed(range(num_steps)):
            next_values = np.zeros_like(values[0]) if t == num_steps - 1 else values[t + 1]
            not_done = (~dones[t]).astype(np.float32)

            delta = rewards[t] + gamma * next_values * not_done - values[t]
            last_gae_lam = delta + gamma * gae_lambda * not_done * last_gae_lam
            advantages[t] = last_gae_lam

        self.returns = advantages + values
        self.advantages = advantages

    def compute_loss_mask(
        self, episode_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute a boolean mask and per-sample loss_mask_ratio.

        Returns:
            mask: (T*n_envs,) bool — True before the first termination per env.
            loss_mask_ratio: (T*n_envs,) float32 — valid_steps / episode_len
                per (episode, env), broadcast to every step. Used to up-weight
                short (successful) episodes in the loss.
        """
        terminations = np.stack(self.terminations).astype(np.float32)  # (T, n_envs)
        T, n_envs = terminations.shape
        assert T % episode_len == 0, (
            f"T ({T}) must be divisible by episode_len ({episode_len})"
        )
        num_episodes = T // episode_len
        term = terminations.reshape(num_episodes, episode_len, n_envs)
        cum = np.cumsum(term, axis=1)
        shifted = np.zeros_like(cum)
        shifted[:, 1:] = cum[:, :-1]
        mask = (shifted == 0)  # (num_episodes, episode_len, n_envs)

        # valid steps per (episode, env), broadcast back to all steps
        valid_counts = mask.sum(axis=1, keepdims=True)  # (num_episodes, 1, n_envs)
        ratio = (valid_counts / episode_len).astype(np.float32)  # (num_episodes, 1, n_envs)
        ratio = np.broadcast_to(ratio, mask.shape)  # (num_episodes, episode_len, n_envs)

        return mask.reshape(-1).astype(bool), ratio.reshape(-1).copy()

    def norm_adv(self, mean: float, std: float) -> None:
        """Normalize advantages in-place with the given global mean and std."""
        self.advantages = (self.advantages - mean) / std

    def get_batch(
        self,
        compute_loss_masks: bool = False,
        episode_len: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Stack and flatten rollout data into a single batch dict.

        Flattens (T, n_envs, ...) into (N, ...) where N = T * n_envs.
        Obs/next_obs nested dicts are flattened to "obs/key", "next_obs/key".

        Returns:
            A dict with flat keys and flattened torch tensors, each (N, ...).
        """
        def _stack_and_flatten(arrays: list[np.ndarray]) -> torch.Tensor:
            t = torch.from_numpy(np.stack(arrays))       # (T, n_envs, ...)
            return t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])  # (N, ...)

        batch: dict[str, Any] = {}

        # obs/next_obs → flat keys "obs/key", "next_obs/key"
        for prefix in ("obs", "next_obs"):
            data_list = getattr(self, prefix)
            if not data_list:
                continue
            for k in data_list[0].keys():
                vals = [d[k] for d in data_list]
                if isinstance(vals[0], np.ndarray):
                    batch[f"{prefix}/{k}"] = _stack_and_flatten(vals)

        for name in ("actions", "rewards", "terminations", "truncations",
                      "prev_logprobs", "prev_values"):
            data_list = getattr(self, name)
            if data_list:
                batch[name] = _stack_and_flatten(data_list)

        if self.forward_inputs:
            # forward_inputs may have T+1 entries (extra for last next_obs);
            # only include the first T for the batch
            fi_list = self.forward_inputs[:len(self.obs)] if len(self.forward_inputs) > len(self.obs) else self.forward_inputs
            keys = fi_list[0].keys()
            batch["forward_inputs"] = {
                k: _stack_and_flatten([d[k] for d in fi_list])
                for k in keys
            }

        if self.next_forward_inputs:
            keys = self.next_forward_inputs[0].keys()
            batch["next_forward_inputs"] = {
                k: _stack_and_flatten([d[k] for d in self.next_forward_inputs])
                for k in keys
            }

        if compute_loss_masks:
            mask, ratio = self.compute_loss_mask(episode_len=episode_len)
            batch["loss_mask"] = torch.from_numpy(mask.astype(np.float32))
            batch["loss_mask_ratio"] = torch.from_numpy(ratio)

        if self.returns is not None:
            batch["returns"] = torch.from_numpy(self.returns.reshape(-1))
        if self.advantages is not None:
            batch["advantages"] = torch.from_numpy(self.advantages.reshape(-1))

        return batch
