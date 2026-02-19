from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch


def _to_numpy(x: Any) -> Any:
    """Convert a torch.Tensor or numpy array to numpy, no-op for other types."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    return x


@dataclass(kw_only=True)
class RolloutResult:
    """
    Results of one-epoch rollouts.
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
            self.prev_logprobs.append(_to_numpy(prev_logprobs))
        if prev_values is not None:
            self.prev_values.append(_to_numpy(prev_values))
        if forward_inputs is not None:
            self.forward_inputs.append(_to_numpy(forward_inputs))

    def compute_returns_and_advantages(
        self,
        gamma: float,
        gae_lambda: float,
        last_values: np.ndarray | None = None,
    ):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).

        Args:
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter controlling the bias-variance trade-off.
            last_values: Value estimate for the last next_obs, shape (n_envs,).
                         Defaults to zeros (no bootstrapping) if None.
        """
        num_steps = len(self.rewards)
        rewards = np.stack(self.rewards).astype(np.float32)         # (T, n_envs)
        values = np.stack(self.prev_values).astype(np.float32)      # (T, n_envs, 1)
        values = values.squeeze(-1)                                 # (T, n_envs)
        terminations = np.stack(self.terminations)                  # (T, n_envs)
        truncations = np.stack(self.truncations)                    # (T, n_envs)

        dones = np.logical_or(terminations, truncations)

        if last_values is None:
            last_values = np.zeros_like(values[0])
        else:
            last_values = np.asarray(last_values, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        last_gae_lam = np.zeros_like(rewards[0])

        for t in reversed(range(num_steps)):
            next_values = last_values if t == num_steps - 1 else values[t + 1]
            not_done = (~dones[t]).astype(np.float32)

            delta = rewards[t] + gamma * next_values * not_done - values[t]
            last_gae_lam = delta + gamma * gae_lambda * not_done * last_gae_lam
            advantages[t] = last_gae_lam

        self.returns = advantages + values
        self.advantages = advantages

    def get_batch(
        self,
        world_size: int = 1,
        normalize_advantages: bool = True,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """
        Stack all rollout data and flatten (T, n_envs) into a single sample
        dimension N = T * n_envs.  Every tensor in the returned dict has
        leading dim N (or is a dict of such tensors for obs / forward_inputs).

        Returns:
            A dict with flattened torch tensors, each of shape (N, ...).
        """
        def _stack_and_flatten(arrays: list[np.ndarray]) -> torch.Tensor:
            t = torch.from_numpy(np.stack(arrays))       # (T, n_envs, ...)
            return t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])  # (N, ...)

        batch: dict[str, Any] = {}

        for name in ("obs", "next_obs"):
            data_list = getattr(self, name)
            stacked = {}
            for k in data_list[0].keys():
                vals = [d[k] for d in data_list]
                if not isinstance(vals[0], np.ndarray):
                    continue
                stacked[k] = _stack_and_flatten(vals)
            batch[name] = stacked

        for name in ("actions", "rewards", "terminations", "truncations", "prev_logprobs", "prev_values"):
            data_list = getattr(self, name)
            if len(data_list) > 0:
                batch[name] = _stack_and_flatten(data_list)

        if self.forward_inputs:
            data_list = self.forward_inputs
            keys = data_list[0].keys()
            batch["forward_inputs"] = {
                k: _stack_and_flatten([d[k] for d in data_list])
                for k in keys
            }

        if self.returns is not None:
            batch["returns"] = torch.from_numpy(self.returns.reshape(-1))
        if self.advantages is not None:
            batch["advantages"] = torch.from_numpy(self.advantages.reshape(-1))

        if normalize_advantages and "advantages" in batch:
            adv = batch["advantages"]
            N_used = (adv.shape[0] // world_size) * world_size
            adv_used = adv[:N_used]
            mean = adv_used.mean().item()
            std = (adv_used.std() + 1e-8).item()
            batch["advantages"] = (adv - mean) / std

        return batch
