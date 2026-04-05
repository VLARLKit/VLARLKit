"""
Test script for ManiskillEnv: random action rollout with PutOnPlateInScene25Main-v3.

Usage:
    python tests/test_maniskill_env.py
    python tests/test_maniskill_env.py --num_envs 4 --max_steps 80 --num_rollouts 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from env_clients.maniskill.maniskill_env import ManiskillEnv


def make_cfg(num_envs: int, max_episode_steps: int):
    cfg = OmegaConf.create({
        "seed": 0,
        "auto_reset": False,
        "ignore_terminations": False,
        "use_rel_reward": False,
        "reward_mode": "default",
        "use_full_state": False,
        "wrap_obs_mode": "default",
        "group_size": 1,
        "use_fixed_reset_state_ids": False,
        "video_cfg": {
            "save_video": False,
            "info_on_video": True,
        },
        "init_params": {
            "id": "PutOnPlateInScene25Main-v3",
            "num_envs": num_envs,
            "obs_mode": "rgb+segmentation",
            "control_mode": None,
            "sim_backend": "gpu",
            "sim_config": {
                "sim_freq": 500,
                "control_freq": 5,
            },
            "max_episode_steps": max_episode_steps,
            "sensor_configs": {"shader_pack": "default"},
            "render_mode": "all",
            "obj_set": "train",
            "use_multiple_plates": False,
        },
    })
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=40)
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--num_rollouts", type=int, default=2)
    args = parser.parse_args()

    cfg = make_cfg(args.num_envs, args.max_steps)
    env = ManiskillEnv(cfg, num_envs=args.num_envs, total_num_processes=1, rank=0)

    print(f"Environment: PutOnPlateInScene25Main-v3")
    print(f"  num_envs: {env.num_envs}")
    print(f"  device: {env.device}")
    print(f"  total_num_group_envs: {env.total_num_group_envs}")
    print(f"  instruction: {env.instruction}")
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
            action = env.sample_action_space()
            obs, rew, terminations, truncations, infos = env.step(action)
            total_reward += rew.sum()

            if step % 10 == 0 or step == args.max_steps - 1:
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
