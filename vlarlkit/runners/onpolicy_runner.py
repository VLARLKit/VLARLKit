import time
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from vlarlkit.rollouts.rollout import Rollout
from vlarlkit.utils.data_sharding import shard_batch


class OnPolicyRunner:
    """
    RL runner: holds cfg, policy, train_rollout_worker and eval_rollout_worker (on rank 0).
    Core loop: train_rollout_worker.rollout() -> broadcast batch -> policy.learn -> sync to workers' actor_model.
    Eval uses pre-initialized eval_rollout_worker (no new env/rollout).
    """

    def __init__(
        self,
        cfg: DictConfig,
        policy: Any,
        train_rollout_worker: Rollout | None = None,
        eval_rollout_worker: Rollout | None = None,
        logger: Any = None,
    ) -> None:
        """
        Args:
            cfg: Full config (algorithm, training, env, etc.).
            policy: Policy instance (e.g. PPO), created in run.py. Required on all ranks.
            train_rollout_worker: Rollout instance for training data collection on rank 0.
            eval_rollout_worker: Rollout instance for evaluation on rank 0; used when eval_interval > 0.
            logger: Logger instance (optional). If None, no logging will be performed.
        """
        self.cfg = cfg
        self.policy = policy
        self.train_rollout_worker = train_rollout_worker
        self.eval_rollout_worker = eval_rollout_worker
        self.logger = logger

        self.rank = dist.get_rank()
        self.device = torch.device(f"cuda:{self.rank}")

    def run(self) -> None:
        """Main RL loop: train rollout -> broadcast batch -> learn -> sync to train worker; periodically run eval worker."""
        max_epochs = int(
            getattr(self.cfg.runner, "max_epochs", 1000)
            if hasattr(self.cfg, "runner") else 1000
        )
        eval_interval = (
            int(getattr(self.cfg.runner, "eval_interval", 0))
            if hasattr(self.cfg, "runner") else 0
        )

        if self.rank == 0:
            self.logger.info(f"Starting training for {max_epochs} epochs")

        start_time = time.time()

        world_size = dist.get_world_size()
        for epoch in range(max_epochs):
            if self.rank == 0 and self.train_rollout_worker is not None:
                rr = self.train_rollout_worker.rollout_result
                rr.clear()

                # Roll out training data
                rollout_start_time = time.time()
                self.train_rollout_worker.rollout()
                rollout_end_time = time.time()
                self.logger.info(f"Collected training data in {rollout_end_time - rollout_start_time:.2f}s")

                gamma = float(self.cfg.algorithm.gamma)
                gae_lambda = float(self.cfg.algorithm.gae_lambda)
                rr.compute_returns_and_advantages(
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    last_values=None,
                )
                batch = rr.get_batch(
                    world_size=world_size,
                    normalize_advantages=self.cfg.algorithm.get("normalize_advantages", True),
                )
            else:
                batch = None

            dist.barrier()

            batch = shard_batch(
                batch if self.rank == 0 else {},
                src=0,
                device=self.device,
            )

            learn_start_time = time.time()
            metrics = self.policy.run_update(batch)
            learn_end_time = time.time()
            if world_size > 1:
                metrics_tensor = torch.tensor(
                    [metrics.get(k, 0.0) for k in sorted(metrics.keys())],
                    device=self.device,
                )
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
                metrics = dict(zip(sorted(metrics.keys()), metrics_tensor.tolist()))
            if self.rank == 0 and self.logger:
                self.logger.info(f"Updated policy in {learn_end_time - learn_start_time:.2f}s")
                train_metrics_str = ", ".join(
                    [f"{k}={v:.4f}" for k, v in metrics.items()]
                )
                self.logger.info(f"Epoch {epoch}/{max_epochs} - Train: {train_metrics_str}")

            if self.rank == 0 and self.train_rollout_worker is not None:
                self._sync_fsdp_to_model(self.train_rollout_worker.actor_model)

            if (
                eval_interval > 0
                and (epoch + 1) % eval_interval == 0
                and self.rank == 0
                and self.eval_rollout_worker is not None
            ):
                self._run_evaluate()

            dist.barrier()

        # Log total training time
        total_time = time.time() - start_time
        if self.rank == 0 and self.logger:
            self.logger.info(f"Training completed in {total_time:.2f}s")

    def _sync_fsdp_to_model(self, target_model: Any) -> None:
        """Copy FSDP model state to target_model (rank 0 only)."""
        if self.rank != 0 or target_model is None:
            return
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        with FSDP.state_dict_type(
            self.policy.get_model(),
            StateDictType.FULL_STATE_DICT,
        ):
            state = self.policy.get_model().state_dict()

        def strip_prefix(sd: dict, prefix: str) -> dict:
            return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

        for prefix in ("_fsdp_module.module.", "module."):
            if state and any(k.startswith(prefix) for k in state):
                state = strip_prefix(state, prefix)
                break
        target_model.load_state_dict(state, strict=False)

    def _run_evaluate(self) -> None:
        """Run eval using pre-initialized eval_rollout_worker (sync latest policy then rollout)."""
        if self.rank != 0 or self.eval_rollout_worker is None:
            return

        if self.logger:
            self.logger.info("Running evaluation...")

        self._sync_fsdp_to_model(self.eval_rollout_worker.actor_model)
        self.eval_rollout_worker.init_rollout()
        rollout_result = self.eval_rollout_worker.rollout_one_epoch()
        eval_metrics = {
            "eval/success_rate_mean": rollout_result["success_once"].mean().cpu().numpy(),
            "eval/success_rate_std": rollout_result["success_once"].std().cpu().numpy(),
            "eval/episode_length_mean": rollout_result["episode_len"].mean().cpu().numpy(),
            "eval/episode_length_std": rollout_result["episode_len"].std().cpu().numpy(),
        }
        eval_metrics_str = ", ".join(
            [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in eval_metrics.items()]
        )
        self.logger.info(f"Eval metrics: {eval_metrics_str}")
