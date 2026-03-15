import logging
import queue
import threading
import time
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from vlarlkit.data.replay_buffer import ReplayBuffer
from vlarlkit.rollouts.rollout import Rollout
from vlarlkit.utils.fsdp_utils import allreduce_mean_std, sync_fsdp_to_model

logger = logging.getLogger("vlarlkit.runner")


class OffPolicyRunner:
    """
    Off-policy async RL runner: rollout and update run in separate threads,
    connected by a bounded queue. The main thread consumes data and runs
    gradient updates; a daemon thread continuously collects rollouts.

    Key design choices:
    - Lock-free weight sync: sync_fsdp_to_model writes actor_model directly
      while rollout thread reads it. Off-policy tolerates stale weights.
    - _data_queue (queue.Queue) is inherently thread-safe.
    - replay_buffer is only accessed from the main thread.
    """

    def __init__(
        self,
        cfg: DictConfig,
        policy: Any,
        train_rollout_worker: Rollout,
        eval_rollout_worker: Rollout | None = None,
        replay_buffer: ReplayBuffer | None = None,
        metric_logger: Any = None,
    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self.train_rollout_worker = train_rollout_worker
        self.eval_rollout_worker = eval_rollout_worker
        self.replay_buffer = replay_buffer
        self.metric_logger = metric_logger

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")

        queue_size = int(cfg.algorithm.get("queue_size", 4))
        self._data_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._should_stop = threading.Event()

    # ------------------------------------------------------------------
    # Thread-safe queue helpers (interruptible by should_stop)
    # ------------------------------------------------------------------

    def _queue_get(self, timeout: float = 1.0) -> dict | None:
        """Blocking get that exits cleanly when should_stop is set."""
        while not self._should_stop.is_set():
            try:
                return self._data_queue.get(timeout=timeout)
            except queue.Empty:
                continue
        return None

    def _queue_put(self, data: dict, timeout: float = 1.0) -> None:
        """Blocking put that exits cleanly when should_stop is set."""
        while not self._should_stop.is_set():
            try:
                self._data_queue.put(data, timeout=timeout)
                return
            except queue.Full:
                continue

    # ------------------------------------------------------------------
    # Rollout thread
    # ------------------------------------------------------------------

    def _rollout_loop(self) -> None:
        """Daemon thread: continuously collects rollouts and enqueues transitions."""
        try:
            self.train_rollout_worker.init_rollout()
            while not self._should_stop.is_set():
                rollout_info = self.train_rollout_worker.rollout_one_epoch()
                rr = self.train_rollout_worker.rollout_result
                transitions = rr.get_transitions()
                transitions["rollout_info"] = rollout_info
                rr.clear()
                self._queue_put(transitions)
        except Exception:
            logger.exception("Rollout thread crashed")
            self._should_stop.set()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main entry point: start rollout thread, then consume data and update."""
        max_update_steps = int(self.cfg.runner.get("max_update_steps", 1000))
        eval_interval = int(self.cfg.runner.get("eval_interval", 0))
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 200))

        # Batch size per rank: global_batch_size / world_size, or fallback to minibatch_size
        global_bs = self.cfg.algorithm.get("global_batch_size", None)
        if global_bs is not None:
            batch_size = int(global_bs) // self.world_size
        else:
            batch_size = int(self.cfg.algorithm.get("minibatch_size", 256))

        # Start rollout thread
        rollout_thread = threading.Thread(
            target=self._rollout_loop, daemon=True, name="rollout-thread"
        )
        rollout_thread.start()

        if self.rank == 0:
            logger.info(
                "Off-policy training started (max_update_steps=%d, update_epoch=%d, batch_size=%d)",
                max_update_steps, update_epoch, batch_size,
            )

        start_time = time.time()
        update_step = 0

        while not self._should_stop.is_set() and update_step < max_update_steps:
            # Wait for a chunk of data from rollout thread
            data = self._queue_get()
            if data is None:
                break

            rollout_info = data.pop("rollout_info")
            self.replay_buffer.add(data)
            rollout_stats = allreduce_mean_std({
                "success_rate": rollout_info["success_once"].astype(float),
            }, self.device)

            # Run update_epoch gradient updates per data collection
            update_start = time.time()
            for _ in range(update_epoch):
                batch = self.replay_buffer.sample(batch_size)
                train_metrics = self.policy.run_update(batch)
                update_step += 1
                if update_step >= max_update_steps:
                    break
            update_end = time.time()

            # Sync weights: policy -> actor (lock-free)
            sync_fsdp_to_model(
                self.policy.get_model(), self.train_rollout_worker.actor_model
            )

            epoch_log: dict[str, float] = {}
            if self.rank == 0:
                epoch_log.update({f"train/{k}": v for k, v in train_metrics.items()})
                epoch_log["rollout/success_rate"] = rollout_stats["success_rate"][0]
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
                logger.info(
                    "Step %d/%d - %s (%.2fs)",
                    update_step, max_update_steps, metrics_str,
                    update_end - update_start,
                )

            # Eval (all ranks must participate in allreduce inside _run_evaluate)
            if eval_interval > 0 and update_step % eval_interval == 0:
                eval_metrics = self._run_evaluate(update_step)
                if self.rank == 0 and eval_metrics is not None:
                    epoch_log.update(eval_metrics)

            if self.rank == 0 and self.metric_logger is not None:
                self.metric_logger.log(epoch_log, step=update_step)

            dist.barrier()

        # Shutdown
        self._should_stop.set()
        rollout_thread.join(timeout=10)

        total_time = time.time() - start_time
        if self.rank == 0:
            logger.info("Training completed in %.2fs (%d updates)", total_time, update_step)
            if self.metric_logger is not None:
                self.metric_logger.finish()

    def _run_evaluate(self, step: int = 0) -> dict[str, float] | None:
        """Run eval on all ranks and all-reduce results."""
        if self.eval_rollout_worker is None:
            return None

        self.eval_rollout_worker.init_rollout()
        rollout_result = self.eval_rollout_worker.run_rollout(
            self.cfg.algorithm.get("eval_rollout_epochs", 1)
        )

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
            eval_str = ", ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items())
            logger.info("Eval step %d: %s", step, eval_str)
            return eval_metrics

        return None
