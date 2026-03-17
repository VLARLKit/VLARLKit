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
from vlarlkit.utils.checkpoint import save_checkpoint
from vlarlkit.utils.fsdp_utils import allreduce_mean_std, sync_fsdp_to_model

logger = logging.getLogger("vlarlkit.runner")

_REPLAY_PREFIXES = ("obs/", "next_obs/")
_REPLAY_KEYS = {"actions", "rewards", "terminations"}


def _filter_replay_keys(batch: dict) -> dict:
    """Keep only the keys needed by the replay buffer."""
    return {
        k: v for k, v in batch.items()
        if k in _REPLAY_KEYS or any(k.startswith(p) for p in _REPLAY_PREFIXES)
    }


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
        output_dir: str = "",
    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self.train_rollout_worker = train_rollout_worker
        self.eval_rollout_worker = eval_rollout_worker
        self.replay_buffer = replay_buffer
        self.metric_logger = metric_logger
        self._output_dir = output_dir

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
                transitions = rr.get_batch()
                transitions["rollout_info"] = rollout_info
                rr.clear()
                self._queue_put(transitions)
        except Exception:
            logger.exception("Rollout thread crashed")
            self._should_stop.set()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _is_warmup_ready(self, warmup_steps: int) -> bool:
        """Check if all ranks' replay buffers have enough warmup data."""
        local_ready = torch.tensor(
            1 if self.replay_buffer.ready(warmup_steps) else 0,
            device=self.device,
        )
        dist.all_reduce(local_ready, op=dist.ReduceOp.MIN)
        return local_ready.item() == 1

    def run(self, start_epoch: int = 0) -> None:
        """Main entry point: start rollout thread, then consume data and update."""
        max_epochs = int(self.cfg.runner.get("max_epochs", 1000))
        eval_interval = int(self.cfg.runner.get("eval_interval", 0))
        save_interval = int(self.cfg.runner.get("save_interval", 0))
        utd_ratio = float(self.cfg.algorithm.get("utd_ratio", 1.0))

        # Batch size per rank
        global_bs = self.cfg.algorithm.get("global_batch_size", None)
        if global_bs is not None:
            batch_size = int(global_bs) // self.world_size
        else:
            batch_size = int(self.cfg.algorithm.get("minibatch_size", 256))

        # Warmup: global warmup steps divided across ranks
        global_warmup = int(self.cfg.algorithm.get("global_warmup_steps", 0))
        warmup_steps = max(1, global_warmup // self.world_size)

        # Start rollout thread
        rollout_thread = threading.Thread(
            target=self._rollout_loop, daemon=True, name="rollout-thread"
        )
        rollout_thread.start()

        if self.rank == 0:
            logger.info(
                "Off-policy training started (max_epochs=%d, utd_ratio=%.1f, "
                "batch_size=%d, warmup_steps=%d per rank)",
                max_epochs, utd_ratio, batch_size, warmup_steps,
            )

        start_time = time.time()
        epoch = start_epoch
        warmed_up = False

        while not self._should_stop.is_set() and epoch < max_epochs:
            data = self._queue_get()
            if data is None:
                break

            rollout_info = data.pop("rollout_info")
            filtered = _filter_replay_keys(data)
            new_samples = next(iter(filtered.values())).shape[0]
            self.replay_buffer.add(filtered)
            rollout_stats = allreduce_mean_std({
                "success_rate": rollout_info["success_once"].astype(float),
            }, self.device)

            # Wait for all ranks to have enough warmup data
            if not warmed_up:
                if not self._is_warmup_ready(warmup_steps):
                    if self.rank == 0:
                        logger.info(
                            "Warming up replay buffer... (%d/%d per rank)",
                            len(self.replay_buffer), warmup_steps,
                        )
                    continue
                warmed_up = True
                if self.rank == 0:
                    logger.info("All ranks warmed up, starting training")

            # Update policy
            num_updates = max(1, int(utd_ratio * new_samples))
            update_start = time.time()
            for _ in range(num_updates):
                batch = self.replay_buffer.sample(batch_size)
                train_metrics = self.policy.run_update(batch)
            update_end = time.time()

            # Sync weights: policy -> actor (lock-free)
            sync_fsdp_to_model(
                self.policy.get_model(), self.train_rollout_worker.actor_model
            )

            epoch += 1
            epoch_log: dict[str, float] = {}
            if self.rank == 0:
                epoch_log.update({f"train/{k}": v for k, v in train_metrics.items()})
                epoch_log["rollout/success_rate"] = rollout_stats["success_rate"][0]
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
                logger.info(
                    "Epoch %d/%d - %s (%d updates, %.2fs)",
                    epoch, max_epochs, metrics_str,
                    num_updates, update_end - update_start,
                )

            # Eval
            if eval_interval > 0 and epoch % eval_interval == 0:
                eval_metrics = self._run_evaluate(epoch)
                if self.rank == 0 and eval_metrics is not None:
                    epoch_log.update(eval_metrics)

            if self.rank == 0 and self.metric_logger is not None:
                self.metric_logger.log(epoch_log, step=epoch)
            
            # Save checkpoint
            if save_interval > 0 and epoch % save_interval == 0:
                wandb_run_id = getattr(getattr(self.metric_logger, "run", None), "id", None)
                save_checkpoint(
                    output_dir=self._output_dir,
                    policy=self.policy,
                    epoch=epoch,
                    wandb_run_id=wandb_run_id,
                    replay_buffer=self.replay_buffer,
                    rank=self.rank,
                )

            dist.barrier()

        # Shutdown
        self._should_stop.set()
        rollout_thread.join(timeout=10)

        total_time = time.time() - start_time
        if self.rank == 0:
            logger.info("Training completed in %.2fs (%d epochs)", total_time, epoch)
            if self.metric_logger is not None:
                self.metric_logger.finish()

    def _run_evaluate(self, epoch: int = 0) -> dict[str, float] | None:
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
            logger.info("Eval epoch %d: %s", epoch, eval_str)
            return eval_metrics

        return None
