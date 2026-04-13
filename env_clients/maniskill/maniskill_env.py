# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# --------------------------------------------------------------------
# Modifications:
#   Modified by VLARLKit Authors.
# --------------------------------------------------------------------

from typing import Optional, OrderedDict, Union

import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.structs.types import Array
from mani_skill.utils.visualization.misc import put_info_on_image, tile_images
from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf

from .utils import extract_termination_from_info, recursive_to_numpy

__all__ = ["ManiskillEnv"]


class ManiskillEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        total_num_processes,
        rank: int = 0,
    ):
        self.cfg = cfg
        self.rank = rank
        self.seed = cfg.seed + rank
        self.total_num_processes = total_num_processes
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations
        self.use_full_state = bool(getattr(cfg, "use_full_state", False))
        self.num_group = num_envs // cfg.group_size
        self.group_size = cfg.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids

        self.video_cfg = cfg.video_cfg

        with open_dict(cfg):
            cfg.init_params.num_envs = num_envs
        env_args = OmegaConf.to_container(cfg.init_params, resolve=True)
        self.env: BaseEnv = gym.make(**env_args)
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.device
        )  # [B, ]
        self._init_reset_state_ids()
        self.info_logging_keys = ["is_src_obj_grasped", "consecutive_grasp", "success"]
        self._show_goal_site_visual()
        self._init_metrics()

    @property
    def total_num_group_envs(self):
        if hasattr(self.env.unwrapped, "total_num_trials"):
            return self.env.unwrapped.total_num_trials
        if hasattr(self.env, "xyz_configs") and hasattr(self.env, "quat_configs"):
            return len(self.env.xyz_configs) * len(self.env.quat_configs)
        return np.iinfo(np.uint8).max // 2  # TODO

    @property
    def num_envs(self):
        return self.env.unwrapped.num_envs

    @property
    def device(self):
        return self.env.unwrapped.device

    @property
    def elapsed_steps(self):
        return self.env.unwrapped.elapsed_steps

    @property
    def instruction(self):
        return self.env.unwrapped.get_language_instruction()

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    def _show_goal_site_visual(self):
        """Keep ManiSkill goal-site visualization visible for reward-model RGB input."""
        if not hasattr(self.env.unwrapped, "goal_site"):
            return

        goal_site = self.env.unwrapped.goal_site
        if hasattr(self.env.unwrapped, "_hidden_objects"):
            while goal_site in self.env.unwrapped._hidden_objects:
                self.env.unwrapped._hidden_objects.remove(goal_site)
        if hasattr(goal_site, "show_visual"):
            goal_site.show_visual()

    def _wrap_obs(self, raw_obs, infos=None):
        wrap_obs_mode = getattr(self.cfg, "wrap_obs_mode", "default")
        if wrap_obs_mode == "raw":
            assert infos is not None
            return recursive_to_numpy(infos["extracted_obs"])

        if wrap_obs_mode == "simple":
            if self.env.unwrapped.obs_mode == "state":
                return recursive_to_numpy({"states": raw_obs})
            elif self.env.unwrapped.obs_mode == "rgb":
                sensor_data = raw_obs.pop("sensor_data")
                raw_obs.pop("sensor_param")
                if self.use_full_state:
                    state = self._get_full_state_obs()
                else:
                    state = common.flatten_state_dict(
                        raw_obs, use_torch=True, device=self.device
                    )

                main_images = sensor_data["base_camera"]["rgb"]
                sorted_images = OrderedDict(sorted(sensor_data.items()))
                sorted_images.pop("base_camera")
                extra_view_images = (
                    torch.stack([v["rgb"] for v in sorted_images.values()], dim=1)
                    if sorted_images
                    else None
                )
                return recursive_to_numpy({
                    "main_images": main_images,
                    "extra_view_images": extra_view_images,
                    "states": state,
                })

        # Default
        obs_image = raw_obs["sensor_data"]["3rd_view_camera"]["rgb"].to(
            torch.uint8
        )  # [B, H, W, C]
        proprioception: torch.Tensor = self.env.unwrapped.agent.robot.get_qpos().to(
            obs_image.device, dtype=torch.float32
        )
        return {
            "main_images": obs_image.cpu().numpy(),
            "states": proprioception.cpu().numpy(),
            "task_descriptions": self.instruction,
        }

    def _get_full_state_obs(self):
        base_env = self.env.unwrapped
        mode_attr = "_obs_mode" if hasattr(base_env, "_obs_mode") else "obs_mode"
        original_mode = getattr(base_env, mode_attr)
        setattr(base_env, mode_attr, "state")
        try:
            state_obs = base_env.get_obs()
        finally:
            setattr(base_env, mode_attr, original_mode)

        if isinstance(state_obs, dict):
            return common.flatten_state_dict(
                state_obs, use_torch=True, device=self.device
            )
        return state_obs

    def _calc_step_reward(self, reward, info):
        if getattr(self.cfg, "reward_mode", "default") == "raw":
            pass
        elif getattr(self.cfg, "reward_mode", "default") == "only_success":
            reward = info["success"] * 1.0
        elif getattr(self.cfg, "reward_mode", "default") == "penalty":
            reward = info["success"].float() - 1.0
        else:
            reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
                self.env.unwrapped.device
            )  # [B, ]
            reward += info["is_src_obj_grasped"] * 0.1
            reward += info["consecutive_grasp"] * 0.1
            reward += (info["success"] & info["is_src_obj_grasped"]) * 1.0
        # diff
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.cpu().numpy().copy()
        if "fail" in infos:
            self.fail_once = self.fail_once | infos["fail"]
            episode_info["fail_once"] = self.fail_once.cpu().numpy().copy()
        episode_info["return"] = self.returns.cpu().numpy().copy()
        episode_info["episode_len"] = self.elapsed_steps.cpu().numpy().copy()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
    ):
        if options is None:
            seed = self.seed
            options = (
                {"episode_id": self.reset_state_ids}
                if self.use_fixed_reset_state_ids
                else {}
            )
        raw_obs, infos = self.env.reset(seed=seed, options=options)
        self._show_goal_site_visual()
        extracted_obs = self._wrap_obs(raw_obs, infos=infos)
        if "env_idx" in options:
            env_idx = options["env_idx"]
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        return extracted_obs, recursive_to_numpy(infos)

    def step(
        self, actions: Union[Array, dict] = None,
    ) -> tuple[Array, Array, Array, Array, dict]:
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        extracted_obs = self._wrap_obs(raw_obs, infos=infos)
        step_reward = self._calc_step_reward(_reward, infos)

        infos = self._record_metrics(step_reward, infos)
        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)
        if isinstance(truncations, bool):
            truncations = torch.tensor([truncations], device=self.device)
            truncations = truncations.repeat(self.num_envs)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = infos.get("success", torch.zeros(self.num_envs, dtype=bool, device=self.device)).cpu().numpy().copy()
            if "fail" in infos:
                infos["episode"]["fail_at_end"] = infos["fail"].cpu().numpy().copy()
            terminations[:] = False

        return (
            extracted_obs,
            step_reward.cpu().numpy().astype(np.float32),
            terminations.cpu().numpy(),
            truncations.cpu().numpy(),
            recursive_to_numpy(infos),
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions
            )

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = np.stack(chunk_rewards, axis=1)              # [num_envs, chunk_steps]
        raw_chunk_terminations = np.stack(raw_chunk_terminations, axis=1)  # [num_envs, chunk_steps]
        raw_chunk_truncations = np.stack(raw_chunk_truncations, axis=1)    # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(axis=1)
        past_truncations = raw_chunk_truncations.any(axis=1)

        if self.ignore_terminations:
            chunk_terminations = np.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = np.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.copy()
            chunk_truncations = raw_chunk_truncations.copy()
        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def run(self):
        obs, info = self.reset()
        for step in range(100):
            action = self.env.action_space.sample()
            obs, rew, terminations, truncations, infos = self.step(action)
            print(
                f"Step {step}: obs={obs.keys()}, rew={rew.mean()}, terminations={terminations.astype(float).mean()}, truncations={truncations.astype(float).mean()}"
            )

    # render utils
    def capture_image(self, infos=None):
        img = self.env.render()
        img = common.to_numpy(img)
        if len(img.shape) == 3:
            img = img[None]

        if infos is not None:
            for i in range(len(img)):
                info_item = {
                    k: v if np.size(v) == 1 else v[i] for k, v in infos.items()
                }
                img[i] = put_info_on_image(img[i], info_item)
        if len(img.shape) > 3:
            if len(img) == 1:
                img = img[0]
            else:
                img = tile_images(img, nrows=int(np.sqrt(self.num_envs)))
        return img

    def render(self, info, rew=None):
        if self.video_cfg.info_on_video:
            scalar_info = gym_utils.extract_scalars_from_info(
                common.to_numpy(info), batch_size=self.num_envs
            )
            if rew is not None:
                scalar_info["reward"] = common.to_numpy(rew)
                if np.size(scalar_info["reward"]) > 1:
                    scalar_info["reward"] = [
                        float(rew) for rew in scalar_info["reward"]
                    ]
                else:
                    scalar_info["reward"] = float(scalar_info["reward"])
            image = self.capture_image(scalar_info)
        else:
            image = self.capture_image()
        return image

    def sample_action_space(self):
        return self.env.action_space.sample()
