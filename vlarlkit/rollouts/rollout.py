from typing import Any

from vlarlkit.data.io_struct import RolloutResult

import torch
import numpy as np


class Rollout:
    def __init__(self, cfg, env, actor_model, rollout_result: RolloutResult, mode="train"):
        self.cfg = cfg
        self.env = env
        self.actor_model = actor_model
        self.rollout_result = rollout_result
        self.mode = mode
        self.auto_reset = self.cfg.env[self.mode].auto_reset

        self.init_rollout()

    def init_rollout(self) -> None:
        if not self.auto_reset:
            obs, _ = self.env.reset()
            self.last_obs = self.prepare_observations(obs)
        self.rollout_result.clear()

    def prepare_observations(self, obs: dict[str, Any]) -> dict[str, Any]:
        image_tensor = obs["main_images"] if "main_images" in obs else None
        wrist_image_tensor = obs["wrist_images"] if "wrist_images" in obs else None
        extra_view_image_tensor = (
            obs["extra_view_images"] if "extra_view_images" in obs else None
        )
        states = obs["states"] if "states" in obs else None
        task_descriptions = (
            list(obs["task_descriptions"]) if "task_descriptions" in obs else None
        )

        return {
            "main_images": image_tensor,  # [N_ENV, H, W, C]
            "wrist_images": wrist_image_tensor,  # [N_ENV, H, W, C] or [N_ENV, N_IMG, H, W, C]
            "extra_view_images": extra_view_image_tensor,  # [N_ENV, N_IMG, H, W, C]
            "states": states,
            "task_descriptions": task_descriptions,
        }

    def rollout_one_epoch(self):
        num_chunk_steps = (
            self.cfg.env[self.mode].max_steps_per_rollout
            // self.cfg.model.num_action_chunks
        )
        
        # If auto_reset is False, it will reset the environment at the beginning of each epoch
        # If auto_reset is True, it will consume from last_obs in last epoch
        if not self.auto_reset:
            obs, _ = self.env.reset()
            obs = self.prepare_observations(obs)
        else:
            obs = self.last_obs

        for _ in range(num_chunk_steps):
            with torch.no_grad():
                actions, info = self.actor_model.predict_action_batch(obs, mode=self.mode)
            next_obs, rewards, terminations, truncations, env_info = self.env.chunk_step(actions)
            next_obs = self.prepare_observations(next_obs)
            rewards = rewards.sum(-1) # [num_envs, chunk_steps] -> [num_envs,]
            terminations = terminations.any(-1) # [num_envs, chunk_steps] -> [num_envs,]
            truncations = truncations.any(-1) # [num_envs, chunk_steps] -> [num_envs,]

            # bootstrap value for truncated rollouts
            if truncations.any() and self.auto_reset:
                final_obs = self.prepare_observations(env_info["final_observation"])
                with torch.no_grad():
                    _, _info = self.actor_model.predict_action_batch(final_obs, mode=self.mode)
                    _final_values = _info["prev_values"].detach().cpu().numpy().reshape(-1)
                final_values = np.zeros_like(_final_values)
                final_values[truncations] = _final_values[truncations]
                rewards += self.cfg.algorithm.gamma * final_values

            self.rollout_result.append_step(
                obs=obs,
                next_obs=next_obs,
                actions=actions,
                rewards=rewards,
                terminations=terminations,
                truncations=truncations,
                prev_logprobs=info["prev_logprobs"],
                prev_values=info["prev_values"],
                forward_inputs=info["forward_inputs"],
            )
            obs = next_obs.copy()

        # update reset_state_ids for next epoch rollout
        self.env.update_reset_state_ids()

        # update last_obs for next epoch rollout (if auto_reset is True)
        if self.auto_reset:
            self.last_obs = obs

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