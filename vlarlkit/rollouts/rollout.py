from vlarlkit.data.io_struct import RolloutResult

import torch


class Rollout:
    def __init__(self, cfg, env, policy, rollout_result: RolloutResult):
        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.rollout_result = rollout_result
        self.auto_reset = self.cfg.env.train.auto_reset

        if self.auto_reset:
            self.last_obs = self.env.reset()

    def rollout_one_epoch(self):
        num_chunk_steps = (
            self.cfg.env.max_steps_per_rollout
            // self.cfg.policy.num_action_chunks
        )
        
        # If auto_reset is False, it will reset the environment at the beginning of each epoch
        # If auto_reset is True, it will consume from last_obs in last epoch
        if not self.auto_reset:
            obs = self.env.reset()
        else:
            obs = self.last_obs

        for _ in range(num_chunk_steps):
            actions, policy_info = self.policy.predict(obs)
            next_obs, rewards, terminations, truncations, env_info = self.env.step(actions)
            rewards = rewards.sum(-1) # [num_envs, chunk_steps] -> [num_envs,]
            terminations = terminations.any(-1) # [num_envs, chunk_steps] -> [num_envs,]
            truncations = truncations.any(-1) # [num_envs, chunk_steps] -> [num_envs,]

            # bootstrap value for truncated rollouts
            if truncations.any() and self.auto_reset:
                final_obs = env_info["final_obs"]
                with torch.no_grad:
                    _, _policy_info = self.policy.predict(final_obs)
                    _final_values = _policy_info["prev_values"]
                final_values = torch.zeros_like(_final_values)
                final_values[truncations] = _final_values
                rewards += self.cfg.algorithm.gamma * final_values

            self.rollout_result.append_step(
                obs=obs,
                next_obs=next_obs,
                actions=actions,
                rewards=rewards,
                terminations=terminations,
                truncations=truncations,
                prev_logprobs=policy_info["logprobs"],
                prev_values=policy_info["values"],
            )
            obs = next_obs.copy()

        # update reset_state_ids for next epoch rollout
        self.env.update_reset_state_ids()

        # update last_obs for next epoch rollout (if auto_reset is True)
        if self.auto_reset:
            self.last_obs = obs

    def rollout(self):
        for epoch in range(self.cfg.algorithm.rollout_epochs):
            self.rollout_one_epoch()