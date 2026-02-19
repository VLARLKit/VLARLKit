"""
Usage (multi-GPU with torchrun):
    torchrun --nproc_per_node=2 examples/run.py
"""
import logging
import torch.distributed as dist

import hydra
from omegaconf import DictConfig

from vlarlkit.data.io_struct import RolloutResult
from vlarlkit.envs.libero.libero_env import LiberoEnv
from vlarlkit.models.openpi import get_model
from vlarlkit.policies import PPOPolicy
from vlarlkit.rollouts import Rollout
from vlarlkit.runners import Runner


def get_env(cfg: DictConfig, mode: str = "train"):
    """Build training or eval env from config."""
    env_cfg = cfg.env.get(mode, cfg.env.train)
    num_envs = int(env_cfg.get("total_num_envs"))
    return LiberoEnv(env_cfg, num_envs=num_envs)


@hydra.main(
    config_path="configs",
    config_name="libero_10_ppo_pi05",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    # Hydra already configures logging; just grab the logger on rank 0
    logger = logging.getLogger("vlarlkit.runner") if rank == 0 else None

    # All ranks: model + policy. Rank 0 only: train/eval envs + rollout model + train/eval rollout workers
    model = get_model(cfg.model)
    policy = PPOPolicy(cfg, model, rank)

    train_rollout_worker = None
    eval_rollout_worker = None
    if rank == 0:
        train_env = get_env(cfg, "train")
        eval_env = get_env(cfg, "eval")
        actor_model = get_model(cfg.model)
        actor_model.to(f"cuda:{rank}")
        train_rollout_result = RolloutResult()
        eval_rollout_result = RolloutResult()
        train_rollout_worker = Rollout(cfg, train_env, actor_model, train_rollout_result, mode="train")
        eval_rollout_worker = Rollout(cfg, eval_env, actor_model, eval_rollout_result, mode="eval")

    runner = Runner(
        cfg=cfg,
        policy=policy,
        train_rollout_worker=train_rollout_worker,
        eval_rollout_worker=eval_rollout_worker,
        logger=logger,
    )
    runner.run()


if __name__ == "__main__":
    main()
