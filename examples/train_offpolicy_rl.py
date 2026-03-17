import random

import numpy as np
import torch
import torch.distributed as dist

import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from vlarlkit.data.io_struct import RolloutResult
from vlarlkit.data.replay_buffer import ReplayBuffer
from vlarlkit.utils.checkpoint import load_checkpoint
from vlarlkit.utils.fsdp_utils import sync_fsdp_to_model
from vlarlkit.utils.remote_env import RemoteEnv
from vlarlkit.models.openpi import get_model
from vlarlkit.policies import DSRLPolicy
from vlarlkit.rollouts import Rollout
from vlarlkit.runners import OffPolicyRunner


def get_env(cfg: DictConfig, mode: str, rank: int):
    """Connect to the remote env client for this rank."""
    host = cfg.env.get("env_client_host", "localhost")
    base_port = int(cfg.env.get("env_client_base_port", 5550))
    port = base_port + rank
    return RemoteEnv(host=host, port=port, env_mode=mode)


@hydra.main(
    config_path="configs",
    config_name="libero_spatial_dsrl_pi05",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    output_dir = HydraConfig.get().runtime.output_dir

    torch.cuda.set_device(rank)

    # Set global random seeds (different per rank for data diversity)
    seed = int(cfg.runner.seed)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    # Initialize policy model and target model
    model = get_model(cfg.model)
    target_model = get_model(cfg.model)

    policy = DSRLPolicy(cfg, model, target_model, rank)

    # Initialize replay buffer
    world_size = dist.get_world_size()
    global_buffer_size = int(cfg.algorithm.get("global_replay_buffer_size", 100000))
    replay_buffer = ReplayBuffer(
        max_size=global_buffer_size // world_size,
        seed=seed + rank,
    )

    # Initialize envs
    train_env = get_env(cfg, "train", rank)
    eval_env = get_env(cfg, "eval", rank)

    # Initialize rollout workers (actor_model is a separate copy for async rollout)
    actor_model = get_model(cfg.model)
    actor_model.to(f"cuda:{rank}")
    train_rollout_result = RolloutResult()
    train_rollout_worker = Rollout(cfg, train_env, actor_model, train_rollout_result, mode="train")
    eval_rollout_worker = Rollout(cfg, eval_env, actor_model, mode="eval")

    # Load checkpoint if resuming
    resume_from = cfg.runner.get("resume_from", None)
    start_step = 0
    wandb_run_id = None
    if resume_from:
        ckpt_meta = load_checkpoint(resume_from, policy, step_key="update_step", replay_buffer=replay_buffer, rank=rank)
        if ckpt_meta:
            start_step = ckpt_meta["update_step"]
            wandb_run_id = ckpt_meta.get("wandb_run_id")
            sync_fsdp_to_model(policy.get_model(), actor_model)

    # Initialize wandb-logger on rank 0 only (after checkpoint load for resume)
    if not cfg.runner.is_debug and rank == 0:
        logger_cfg = cfg.runner.get("logger", {})
        wandb_kwargs = dict(
            project=logger_cfg.get("project", "VLARLKit"),
            name=logger_cfg.get("experiment_name", "default"),
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=output_dir,
        )
        if wandb_run_id:
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = "must"
        wandb.init(**wandb_kwargs)
        metric_logger = wandb
    else:
        metric_logger = None

    runner = OffPolicyRunner(
        cfg=cfg,
        policy=policy,
        train_rollout_worker=train_rollout_worker,
        eval_rollout_worker=eval_rollout_worker,
        replay_buffer=replay_buffer,
        metric_logger=metric_logger,
        output_dir=output_dir,
    )
    runner.run(start_step=start_step)


if __name__ == "__main__":
    main()
