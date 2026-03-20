import gc
import logging
import time
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from vlarlkit.rollouts.rollout import Rollout
from vlarlkit.utils.checkpoint import save_checkpoint
from vlarlkit.utils.fsdp_utils import allreduce_mean, allreduce_mean_std, sync_fsdp_to_model

logger = logging.getLogger("vlarlkit.runner")


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
        metric_logger: Any = None,
        output_dir: str = "",
    ) -> None:

        self.cfg = cfg
        self.policy = policy
        self.train_rollout_worker = train_rollout_worker
        self.eval_rollout_worker = eval_rollout_worker
        self.metric_logger = metric_logger
        self._output_dir = output_dir

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")

    def run(self, start_epoch: int = 1) -> None:
        """Main RL loop: all ranks rollout -> all-reduce adv stats -> learn -> sync actor; periodically eval."""
        max_epochs = int(self.cfg.runner.max_epochs)
        eval_interval = int(self.cfg.runner.eval_interval)
        save_interval = int(self.cfg.runner.get("save_interval", 0))

        gamma = float(self.cfg.algorithm.gamma)
        gae_lambda = float(self.cfg.algorithm.gae_lambda)
        normalize_advantages = self.cfg.algorithm.get("normalize_advantages", True)
        train_env_cfg = self.cfg.env.train
        compute_loss_masks = (
            not train_env_cfg.auto_reset and
            not train_env_cfg.ignore_terminations
        )
        max_steps = int(train_env_cfg.max_steps_per_rollout)
        num_action_chunks = int(self.cfg.model.num_action_chunks)
        episode_len = max_steps // num_action_chunks

        if not train_env_cfg.auto_reset:
            max_ep = int(train_env_cfg.max_episode_steps)
            assert max_steps == max_ep, (
                f"max_steps_per_rollout ({max_steps}) != max_episode_steps ({max_ep})"
            )
            assert max_steps % num_action_chunks == 0, (
                f"max_steps_per_rollout ({max_steps}) % num_action_chunks ({num_action_chunks}) != 0"
            )

        if self.rank == 0:
            logger.info("Starting training for %d epochs", max_epochs)

        start_time = time.time()

        for epoch in range(start_epoch, max_epochs + 1):
            # Rollout
            rollout_start_time = time.time()
            rr = self.train_rollout_worker.rollout_result
            self.train_rollout_worker.init_rollout()
            rollout_infos = self.train_rollout_worker.run_rollout(self.cfg.algorithm.rollout_epochs)
            rollout_end_time = time.time()

            rollout_stats = allreduce_mean_std({
                "success_rate": rollout_infos["success_once"].astype(float),
            }, self.device)

            if self.rank == 0:
                logger.info("Collected training data in %.2fs", rollout_end_time - rollout_start_time)

            rr.compute_returns_and_advantages(
                gamma=gamma,
                gae_lambda=gae_lambda,
                last_values=None,
            )

            if normalize_advantages:
                mask, _ = rr.compute_loss_mask(episode_len=episode_len) if compute_loss_masks else (None, None)
                stats = allreduce_mean_std(
                    {"adv": rr.advantages}, self.device, mask=mask,
                )
                mean, std = stats["adv"]
                rr.norm_adv(mean, std + 1e-5)

            # Update policy
            batch = rr.get_batch(compute_loss_masks=compute_loss_masks, episode_len=episode_len)

            batch_stats = allreduce_mean({
                "adv_mean": batch["advantages"].mean().item(),
                "returns_mean": batch["returns"].mean().item(),
            }, self.device)

            update_start_time = time.time()
            train_metrics = self.policy.run_update(batch)
            update_end_time = time.time()

            train_metrics = allreduce_mean(train_metrics, self.device)

            epoch_log: dict[str, float] = {}

            if self.rank == 0:
                logger.info("Updated policy in %.2fs", update_end_time - update_start_time)
                train_metrics_str = ", ".join(
                    f"{k}={v:.4f}" for k, v in train_metrics.items()
                )
                logger.info("Epoch %d/%d - Train: %s", epoch, max_epochs, train_metrics_str)
                epoch_log.update({f"rollout/{k}": v for k, v in batch_stats.items()})
                epoch_log["rollout/success_rate"] = rollout_stats["success_rate"][0]
                epoch_log.update({f"train/{k}": v for k, v in train_metrics.items()})

            sync_fsdp_to_model(self.policy.get_model(), self.train_rollout_worker.actor_model)
            gc.collect()
            torch.cuda.empty_cache()

            # Eval
            if eval_interval > 0 and epoch % eval_interval == 0:
                eval_metrics = self._run_evaluate(epoch)
                if self.rank == 0 and eval_metrics:
                    epoch_log.update(eval_metrics)

            if self.rank == 0 and self.metric_logger is not None and epoch_log:
                self.metric_logger.log(epoch_log, step=epoch)

            # Save checkpoint
            if save_interval > 0 and epoch % save_interval == 0:
                wandb_run_id = getattr(getattr(self.metric_logger, "run", None), "id", None)
                save_checkpoint(
                    output_dir=self._output_dir,
                    policy=self.policy,
                    epoch=epoch,
                    wandb_run_id=wandb_run_id,
                    rank=self.rank,
                )
                # FSDP state_dict() all-gather leaves temporary GPU buffers
                # that Python GC may not immediately reclaim
                gc.collect()
                torch.cuda.empty_cache()

            dist.barrier()

        total_time = time.time() - start_time
        if self.rank == 0:
            logger.info("Training completed in %.2fs", total_time)
            if self.metric_logger is not None:
                self.metric_logger.finish()

    def _run_evaluate(self, epoch: int = 0) -> dict[str, float] | None:
        """Run eval on all ranks and all-reduce results."""
        if self.eval_rollout_worker is None:
            return None

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
                f"{k}={v:.4f}" for k, v in eval_metrics.items()
            )
            logger.info("Eval metrics: %s", eval_metrics_str)
            return eval_metrics

        return None
