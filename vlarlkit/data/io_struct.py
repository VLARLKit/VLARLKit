from dataclasses import dataclass, field

import torch


@dataclass(kw_only=True)
class RolloutResult:
    """
    Results of one-epoch rollouts.
    """

    obs: list[torch.Tensor] = field(default_factory=list) # each element is (n_envs, *obs_shape), length is num_chunk_steps
    next_obs: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    rewards: list[torch.Tensor] = field(default_factory=list)
    terminations: list[torch.Tensor] = field(default_factory=list)
    truncations: list[torch.Tensor] = field(default_factory=list)
    prev_logprobs: list[torch.Tensor] = field(default_factory=list)
    prev_values: list[torch.Tensor] = field(default_factory=list)

    returns: torch.Tensor | None = field(default=None, repr=False)
    advantages: torch.Tensor | None = field(default=None, repr=False)

    def append_step(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        prev_logprobs: torch.Tensor=None,
        prev_values: torch.Tensor=None,
    ) -> None:
        self.obs.append(obs)
        self.next_obs.append(next_obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.terminations.append(terminations)
        self.truncations.append(truncations)
        if prev_logprobs is not None:
            self.prev_logprobs.append(prev_logprobs)
        if prev_values is not None:
            self.prev_values.append(prev_values)

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float, last_values: torch.Tensor | None = None):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).

        Args:
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter controlling the bias-variance trade-off.
            last_values: Value estimate for the last next_obs, shape (n_envs,).
                         Defaults to zeros (no bootstrapping) if None.
        """
        num_steps = len(self.rewards)
        rewards = torch.stack(self.rewards)            # (T, n_envs)
        values = torch.stack(self.prev_values)          # (T, n_envs)
        terminations = torch.stack(self.terminations)   # (T, n_envs)
        truncations = torch.stack(self.truncations)     # (T, n_envs)

        dones = torch.logical_or(terminations, truncations) # (T, n_envs)

        if last_values is None:
            last_values = torch.zeros_like(values[0])

        advantages = torch.zeros_like(rewards)
        last_gae_lam = torch.zeros_like(rewards[0])

        for t in reversed(range(num_steps)):
            next_values = last_values if t == num_steps - 1 else values[t + 1]
            not_done = 1.0 - dones[t]

            delta = rewards[t] + gamma * next_values * not_done - values[t]
            last_gae_lam = delta + gamma * gae_lambda * not_done * last_gae_lam
            advantages[t] = last_gae_lam

        self.returns = advantages + values
        self.advantages = advantages

    def get_batch(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """
        Stack all rollout data into a single batch dict.

        obs and next_obs are dicts of arrays, so they are stacked per key.
        returns and advantages are included only if they have been computed.

        Returns:
            A dict with stacked arrays, each of shape (T, n_envs, ...).
        """
        batch = {}

        # obs and next_obs: list[dict[str, torch.Tensor]] -> dict[str, torch.Tensor]
        for name in ("obs", "next_obs"):
            data_list = getattr(self, name)
            keys = data_list[0].keys()
            batch[name] = {k: torch.stack([d[k] for d in data_list]) for k in keys}

        # Other list fields: list[torch.Tensor] -> torch.Tensor
        for name in ("actions", "rewards", "terminations", "truncations", "prev_logprobs", "prev_values"):
            data_list = getattr(self, name)
            if len(data_list) > 0:
                batch[name] = torch.stack(data_list)

        # Computed fields (already stacked)
        if self.returns is not None:
            batch["returns"] = self.returns
        if self.advantages is not None:
            batch["advantages"] = self.advantages

        return batch
