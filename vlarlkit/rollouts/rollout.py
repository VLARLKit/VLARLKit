from typing import Any

from vlarlkit.data.io_struct import RolloutResult

import torch
import numpy as np


class Rollout:
    def __init__(self, cfg, env, actor_model, rollout_result: RolloutResult | None = None, mode="train"):
        self.cfg = cfg
        self.env = env
        self.actor_model = actor_model
        self.actor_model.eval()
        self.rollout_result = rollout_result
        self.mode = mode
        self._auto_reset = self.cfg.env[self.mode].auto_reset
        self._use_dsrl = bool(self.cfg.model.get("openpi", {}).get("use_dsrl", False))

        self.init_rollout()

    def init_rollout(self) -> None:
        if self._auto_reset:
            self._last_obs, _ = self.env.reset()
        if self.rollout_result is not None:
            self.rollout_result.clear()

    def rollout_one_epoch(self):
        num_chunk_steps = (
            self.cfg.env[self.mode].max_steps_per_rollout
            // self.cfg.model.num_action_chunks
        )

        # If auto_reset is False, it will reset the environment at the beginning of each epoch
        # If auto_reset is True, it will consume from last_obs in last epoch
        if not self._auto_reset:
            obs, _ = self.env.reset()
        else:
            obs = self._last_obs

        for _ in range(num_chunk_steps):
            with torch.no_grad():
                actions, info = self.actor_model.predict_action_batch(obs, mode=self.mode)
            next_obs, rewards, terminations, truncations, env_info = self.env.chunk_step(actions)
            rewards = rewards.sum(-1) # [num_envs, chunk_steps] -> [num_envs,]
            terminations = terminations.any(-1) # [num_envs, chunk_steps] -> [num_envs,]
            truncations = truncations.any(-1) # [num_envs, chunk_steps] -> [num_envs,]

            # Fix next_obs for done envs under auto_reset: use final_observation
            # (the true terminal obs) instead of the reset obs from the new episode.
            dones = np.logical_or(terminations, truncations)
            if dones.any() and self._auto_reset and "final_observation" in env_info:
                final_obs = env_info["final_observation"]
                next_obs_for_buffer = {
                    k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in next_obs.items()
                }
                for k in next_obs_for_buffer:
                    if isinstance(next_obs_for_buffer[k], np.ndarray):
                        next_obs_for_buffer[k][dones] = final_obs[k][dones]
            else:
                next_obs_for_buffer = next_obs

            if self.rollout_result is not None:
                # DSRL: store noise_actions (from forward_inputs) instead of real_actions
                stored_actions = actions
                if self._use_dsrl and "action" in info.get("forward_inputs", {}):
                    stored_actions = info["forward_inputs"]["action"]
                    if hasattr(stored_actions, "detach"):
                        stored_actions = stored_actions.detach().cpu().numpy()

                self.rollout_result.append_step(
                    obs=obs,
                    next_obs=next_obs_for_buffer,
                    actions=stored_actions,
                    rewards=rewards,
                    terminations=terminations,
                    truncations=truncations,
                    prev_logprobs=info["prev_logprobs"],
                    prev_values=info["prev_values"],
                    forward_inputs=info["forward_inputs"],
                )
            obs = next_obs.copy()

        # When eval + auto_reset, _handle_auto_reset already calls
        # update_reset_state_ids on every auto-reset. Calling it again
        # here would waste ordered IDs. In all other cases this is the
        # only place that rotates task/trial IDs between epochs.
        if not (self.mode == "eval" and self._auto_reset):
            self.env.update_reset_state_ids()

        # update last_obs for next epoch rollout (if auto_reset is True)
        if self._auto_reset:
            self._last_obs = obs

        # return the episode info
        if "final_info" in env_info:
            return env_info["final_info"]["episode"]
        else:
            return env_info["episode"]

    def run_rollout(self, rollout_epochs: int):
        rollout_infos = []
        for epoch in range(rollout_epochs):
            rollout_infos.append(self.rollout_one_epoch())
        # list of dicts -> dict of lists
        rollout_infos = {k: np.concatenate([info[k] for info in rollout_infos]) for k in rollout_infos[0]}
        return rollout_infos
