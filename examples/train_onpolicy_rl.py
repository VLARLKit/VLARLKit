import logging
import random

import numpy as np
import torch
import torch.distributed as dist

import hydra
from omegaconf import DictConfig

from vlarlkit.data.io_struct import RolloutResult
from vlarlkit.utils.remote_env import RemoteEnv
from vlarlkit.models.openpi import get_model
from vlarlkit.policies import PPOPolicy
from vlarlkit.rollouts import Rollout
from vlarlkit.runners import OnPolicyRunner


def get_env(cfg: DictConfig, mode: str, world_size: int, rank: int):
    """Connect to the remote env client for this rank."""
    host = cfg.env.get("env_client_host", "localhost")
    base_port = int(cfg.env.get("env_client_base_port", 5550))
    port = base_port + rank
    return RemoteEnv(host=host, port=port, env_mode=mode)


@hydra.main(
    config_path="configs",
    config_name="libero_10_ppo_pi05",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # initialize logger
    logger = logging.getLogger("vlarlkit.runner") if rank == 0 else None

    # Must be called before any model creation: sets the default CUDA device
    # for this process so that implicit CUDA ops (e.g. inside torch.compile,
    # transformers init) land on the correct GPU.
    torch.cuda.set_device(rank)

    # Set global random seeds (different per rank for data diversity)
    seed = int(cfg.runner.seed)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    # Initialize policy
    model = get_model(cfg.model)
    policy = PPOPolicy(cfg, model, rank)

    # Initialize envs
    train_env = get_env(cfg, "train", world_size, rank)
    eval_env = get_env(cfg, "eval", world_size, rank)

    # Initialize rollout workers
    actor_model = get_model(cfg.model)
    actor_model.to(f"cuda:{rank}")
    train_rollout_result = RolloutResult()
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
