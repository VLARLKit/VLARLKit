import time
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from vlarlkit.rollouts.rollout import Rollout
from vlarlkit.utils.conversion_utils import to_device
from vlarlkit.utils.fsdp_utils import allreduce_mean, allreduce_mean_std, sync_fsdp_to_model


class OnPolicyRunner:
    """
    On-Policy RL runner: all ranks perform rollout independently, then
    all-reduce advantage stats for normalization. Training uses FSDP
    for gradient synchronization.
    """

    def __init__(
        self,
        cfg: DictConfig,
        policy: Any,
        train_rollout_worker: Rollout,
        eval_rollout_worker: Rollout | None = None,
        logger: Any = None,
    ) -> None:

        self.cfg = cfg
        self.policy = policy
        self.train_rollout_worker = train_rollout_worker
        self.eval_rollout_worker = eval_rollout_worker
        self.logger = logger

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")

    def run(self) -> None:
        """Main RL loop: all ranks rollout -> all-reduce adv stats -> learn -> sync actor; periodically eval."""
        max_epochs = int(self.cfg.runner.max_epochs)
        eval_interval = int(self.cfg.runner.eval_interval)

        gamma = float(self.cfg.algorithm.gamma)
        gae_lambda = float(self.cfg.algorithm.gae_lambda)
        normalize_advantages = self.cfg.algorithm.get("normalize_advantages", True)
        train_env_cfg = self.cfg.env.train
        compute_loss_masks = (
            not train_env_cfg.auto_reset and
            not train_env_cfg.ignore_terminations
        )

        if self.rank == 0:
            self.logger.info(f"Starting training for {max_epochs} epochs")

        start_time = time.time()

        for epoch in range(max_epochs):
            # rollout
            rollout_start_time = time.time()
            rr = self.train_rollout_worker.rollout_result
            self.train_rollout_worker.init_rollout()
            self.train_rollout_worker.run_rollout(self.cfg.algorithm.rollout_epochs)
            rollout_end_time = time.time()
            if self.rank == 0 and self.logger:
                self.logger.info(f"Collected training data in {rollout_end_time - rollout_start_time:.2f}s")

            rr.compute_returns_and_advantages(
                gamma=gamma,
                gae_lambda=gae_lambda,
                last_values=None,
            )

            if normalize_advantages:
                mask = rr.compute_loss_mask() if compute_loss_masks else None
                stats = allreduce_mean_std(
                    {"adv": rr.advantages}, self.device, mask=mask,
                )
                mean, std = stats["adv"]
                rr.norm_adv(mean, std + 1e-8)

            # update
            batch = rr.get_batch(compute_loss_masks=compute_loss_masks)
            batch = to_device(batch, self.device)
            update_start_time = time.time()
            metrics = self.policy.run_update(batch)
            update_end_time = time.time()

            metrics = allreduce_mean(metrics, self.device)

            if self.rank == 0:
                self.logger.info(f"Updated policy in {update_end_time - update_start_time:.2f}s")
                train_metrics_str = ", ".join(
                    [f"{k}={v:.4f}" for k, v in metrics.items()]
                )
                self.logger.info(f"Epoch {epoch}/{max_epochs} - Train: {train_metrics_str}")

            sync_fsdp_to_model(self.policy.get_model(), self.train_rollout_worker.actor_model)

            if (
                eval_interval > 0
                and (epoch + 1) % eval_interval == 0
                and self.eval_rollout_worker is not None
            ):
                self._run_evaluate()

            dist.barrier()

        total_time = time.time() - start_time
        if self.rank == 0 and self.logger:
            self.logger.info(f"Training completed in {total_time:.2f}s")

    def _run_evaluate(self) -> None:
        """Run eval on all ranks and all-reduce the results."""
        if self.eval_rollout_worker is None:
            return

        self.eval_rollout_worker.init_rollout()
        rollout_result = self.eval_rollout_worker.run_rollout(self.cfg.algorithm.eval_rollout_epochs)

        stats = allreduce_mean_std({
            "success": rollout_result["success_once"],
            "episode_len": rollout_result["episode_len"],
        }, self.device)

        if self.rank == 0:
            eval_metrics = {
                "eval/success_rate_mean": stats["success"][0],
                "eval/success_rate_std": stats["success"][1],
                "eval/episode_length_mean": stats["episode_len"][0],
                "eval/episode_length_std": stats["episode_len"][1],
            }
            eval_metrics_str = ", ".join(
                [f"{k}={v:.4f}" for k, v in eval_metrics.items()]
            )
            self.logger.info(f"Eval metrics: {eval_metrics_str}")
