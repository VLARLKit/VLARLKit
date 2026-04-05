from typing import Any

from vlarlkit.data.io_struct import RolloutResult
from vlarlkit.utils.conversion_utils import to_numpy
from vlarlkit.utils.action_utils import prepare_actions

import torch
import numpy as np


class Rollout:
    def __init__(self, cfg, env, actor_model, rollout_result: RolloutResult | None = None, mode="train",
                 is_onpolicy: bool = True):
        self.cfg = cfg
        self.env = env
        self.actor_model = actor_model
        self.actor_model.eval()
        self.rollout_result = rollout_result
        self.mode = mode
        self._is_onpolicy = is_onpolicy

        self.init_rollout()

    def init_rollout(self) -> None:
        if self.rollout_result is not None:
            self.rollout_result.clear()

    def rollout_one_epoch(self):
        num_chunk_steps = (
            self.cfg.env[self.mode].max_episode_steps
            // self.cfg.model.num_action_chunks
        )

        obs, _ = self.env.reset()

        for _ in range(num_chunk_steps):
            with torch.no_grad():
                actions, info = self.actor_model.predict_action_batch(obs, mode=self.mode)
            actions = prepare_actions(
                raw_chunk_actions=actions,
                env_type=self.cfg.env[self.mode].env_type,
                model_type=self.cfg.model.model_type, 
                num_action_chunks=self.cfg.model.num_action_chunks,
                action_dim=self.cfg.model.action_dim,
                policy=self.cfg.model.get("policy_setup", None)
            )
            next_obs, rewards, terminations, truncations, env_info = self.env.chunk_step(actions)

            if self._is_onpolicy:
                rewards = rewards.sum(-1)
            else:
                # Truncated sum: only accumulate rewards up to (and including)
                # the first termination step; post-termination rewards are invalid.
                # cum_term = np.cumsum(terminations, axis=-1)
                # valid_mask = (cum_term <= 1).astype(rewards.dtype)
                # rewards = (rewards * valid_mask).sum(-1)
                rewards = rewards.max(-1)

            terminations = terminations.any(-1)
            truncations = truncations.any(-1)

            if self.rollout_result is not None:
                self.rollout_result.append_step(
                    obs=obs,
                    next_obs=next_obs,
                    actions=actions,
                    rewards=rewards,
                    terminations=terminations,
                    truncations=truncations,
                    prev_logprobs=info.get("prev_logprobs"),
                    prev_values=info.get("prev_values"),
                    forward_inputs=info.get("forward_inputs"),
                )
            obs = next_obs.copy()

        # Off-policy: run one extra predict to get forward_inputs for the
        # last next_obs, then temporal-shift to build next_forward_inputs.
        if not self._is_onpolicy and self.rollout_result is not None:
            with torch.no_grad():
                _, extra_info = self.actor_model.predict_action_batch(obs, mode=self.mode)
            extra_fi = extra_info.get("forward_inputs")
            if extra_fi is not None:
                self.rollout_result.forward_inputs.append(to_numpy(extra_fi))
                self.rollout_result.build_next_forward_inputs()
        
        # Update reset state ids for next epoch rollouts
        self.env.update_reset_state_ids()

        return env_info["episode"]

    def run_rollout(self, rollout_epochs: int):
        rollout_infos = []
        for epoch in range(rollout_epochs):
            rollout_infos.append(self.rollout_one_epoch())
        rollout_infos = {k: np.concatenate([info[k] for info in rollout_infos]) for k in rollout_infos[0]}
        return rollout_infos
