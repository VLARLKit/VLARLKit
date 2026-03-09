import random

import numpy as np
import torch
import torch.distributed as dist

import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from vlarlkit.data.io_struct import RolloutResult
from vlarlkit.utils.remote_env import RemoteEnv
from vlarlkit.models.openpi import get_model
from vlarlkit.policies import PPOPolicy
from vlarlkit.rollouts import Rollout
from vlarlkit.runners import OnPolicyRunner


def get_env(cfg: DictConfig, mode: str, rank: int):
    """Connect to the remote env client for this rank."""
    host = cfg.env.get("env_client_host", "localhost")
    base_port = int(cfg.env.get("env_client_base_port", 5550))
    port = base_port + rank
    return RemoteEnv(host=host, port=port, env_mode=mode)


@hydra.main(
    config_path="configs",
    config_name="libero_spatial_ppo_pi05",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    # Initialize wandb-logger on rank 0 only
    if not cfg.runner.is_debug and rank == 0:
        logger_cfg = cfg.runner.get("logger", {})
        wandb.init(
            project=logger_cfg.get("project", "VLARLKit"),
            name=logger_cfg.get("experiment_name", "default"),
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=HydraConfig.get().runtime.output_dir,
        )
        metric_logger = wandb
    else:
        metric_logger = None

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
    train_env = get_env(cfg, "train", rank)
    eval_env = get_env(cfg, "eval", rank)

    # Initialize rollout workers
    actor_model = get_model(cfg.model)
    actor_model.to(f"cuda:{rank}")
    train_rollout_result = RolloutResult()
    train_rollout_worker = Rollout(cfg, train_env, actor_model, train_rollout_result, mode="train")
    eval_rollout_worker = Rollout(cfg, eval_env, actor_model, mode="eval")

    runner = OnPolicyRunner(
        cfg=cfg,
        policy=policy,
        train_rollout_worker=train_rollout_worker,
        eval_rollout_worker=eval_rollout_worker,
        metric_logger=metric_logger,
    )
    runner.run()


if __name__ == "__main__":
    main()
