"""
Test script for LiberoEnv: random action rollout with libero_spatial.

Usage:
    python tests/test_libero_env.py
    python tests/test_libero_env.py --num_envs 4 --max_steps 240 --num_rollouts 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from omegaconf import OmegaConf

from env_clients.libero.libero_env import LiberoEnv


def make_cfg(num_envs: int, max_episode_steps: int):
    cfg = OmegaConf.create({
        "seed": 0,
        "ignore_terminations": False,
        "use_rel_reward": True,
        "reward_coef": 1.0,
        "reset_gripper_open": True,
        "is_eval": False,
        "task_suite_name": "libero_spatial",
        "group_size": 1,
        "use_ordered_reset_state_ids": False,
        "specific_reset_id": None,
        "max_episode_steps": max_episode_steps,
        "video_cfg": {
            "save_video": False,
            "info_on_video": True,
        },
        "init_params": {
            "camera_heights": 256,
            "camera_widths": 256,
        },
    })
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=240)
    parser.add_argument("--num_rollouts", type=int, default=2)
    args = parser.parse_args()

    cfg = make_cfg(args.num_envs, args.max_steps)
    env = LiberoEnv(cfg, num_envs=args.num_envs, total_num_processes=1, rank=0)

    print(f"Environment: libero_spatial")
    print(f"  num_envs: {env.num_envs}")
    print(f"  total_num_group_envs: {env.total_num_group_envs}")
    print(f"  task_descriptions: {env.task_descriptions[:2]}...")
    print()

    for rollout_idx in range(args.num_rollouts):
        print(f"=== Rollout {rollout_idx + 1}/{args.num_rollouts} ===")
        obs, _ = env.reset()
        print(f"  Reset obs keys: {obs.keys()}")
        for k, v in obs.items():
            if hasattr(v, "shape"):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"    {k}: {type(v).__name__}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")

        total_reward = 0.0
        for step in range(args.max_steps):
            action = np.random.uniform(-1, 1, size=(args.num_envs, 7)).astype(np.float32)
            obs, rew, terminations, truncations, infos = env.step(action)
            total_reward += rew.sum()

            if step % 30 == 0 or step == args.max_steps - 1:
                print(
                    f"  Step {step:3d}: "
                    f"reward={rew.mean():.4f}, "
                    f"terminated={terminations.any()}, "
                    f"truncated={truncations.any()}"
                )

            if "episode" in infos:
                ep = infos["episode"]
                if "success_once" in ep and ep["success_once"].any():
                    print(f"  Step {step:3d}: success detected!")

        print(f"  Total reward: {total_reward:.4f}")
        env.update_reset_state_ids()
        print()

    print("Test passed.")


if __name__ == "__main__":
    main()
