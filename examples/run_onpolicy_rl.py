"""
Usage (multi-GPU with torchrun):
    torchrun --nproc_per_node=2 examples/run.py
"""
import logging
import torch
import torch.distributed as dist

import hydra
from omegaconf import DictConfig

from vlarlkit.data.io_struct import RolloutResult
from vlarlkit.envs.libero.libero_env import LiberoEnv
from vlarlkit.models.openpi import get_model
from vlarlkit.policies import PPOPolicy
from vlarlkit.rollouts import Rollout
from vlarlkit.runners import OnPolicyRunner


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

    # initialize logger
    logger = logging.getLogger("vlarlkit.runner") if rank == 0 else None

    # Must be called before any model creation: sets the default CUDA device
    # for this process so that implicit CUDA ops (e.g. inside torch.compile,
    # transformers init) land on the correct GPU.
    torch.cuda.set_device(rank)

    # All ranks: policy.
    # Rank 0 only: envs + rollout workers.
    model = get_model(cfg.model)
    policy = PPOPolicy(cfg, model, rank)

    train_rollout_worker = None
    eval_rollout_worker = None
    if rank == 0:
        train_env = get_env(cfg, "train")
        eval_env = get_env(cfg, "eval")
        actor_model = get_model(cfg.model)
        actor_model.to(f"cuda:{rank}")
        train_rollout_result = RolloutResult() # store rollout data
        eval_rollout_result = RolloutResult()
        train_rollout_worker = Rollout(cfg, train_env, actor_model, train_rollout_result, mode="train")
        eval_rollout_worker = Rollout(cfg, eval_env, actor_model, eval_rollout_result, mode="eval")

    runner = OnPolicyRunner(
        cfg=cfg,
        policy=policy,
        train_rollout_worker=train_rollout_worker,
        eval_rollout_worker=eval_rollout_worker,
        logger=logger,
    )
    runner.run()


if __name__ == "__main__":
    main()
