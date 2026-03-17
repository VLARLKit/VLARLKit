import logging
import os
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

logger = logging.getLogger("vlarlkit.checkpoint")


def _extract_fsdp_state_dict(fsdp_model: FSDP) -> dict:
    """Extract full state dict from FSDP model (collective op, all ranks must call).

    Uses rank0_only=True + offload_to_cpu=True so that under FULL_SHARD only
    rank 0 materializes the full state dict (on CPU), avoiding GPU OOM.
    Under NO_SHARD this is a harmless extra D2H copy.
    """
    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, full_cfg):
        return fsdp_model.state_dict()


def _load_fsdp_state_dict(fsdp_model: FSDP, state_dict: dict) -> None:
    """Load full state dict into FSDP model (symmetric with _extract).

    FullStateDictConfig is only used by state_dict(), not load_state_dict().
    The FULL_STATE_DICT context is sufficient for load — FSDP will re-shard
    as needed (FULL_SHARD) or load directly (NO_SHARD).
    """
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
        fsdp_model.load_state_dict(state_dict)


def save_checkpoint(
    output_dir: str,
    policy,
    epoch: int,
    wandb_run_id: str | None = None,
    replay_buffer=None,
    rank: int = 0,
) -> None:
    """Save training checkpoint. All ranks must call (FSDP collective ops)."""
    ckpt_dir = os.path.join(output_dir, "checkpoint")

    model_state = _extract_fsdp_state_dict(policy.get_model())

    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

        save_dict = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "policy_state": policy.state_dict(),
            "wandb_run_id": wandb_run_id,
        }

        # Atomic write via temp file + rename
        dst = os.path.join(ckpt_dir, "latest.pt")
        fd, tmp_path = tempfile.mkstemp(dir=ckpt_dir, suffix=".tmp")
        os.close(fd)
        try:
            torch.save(save_dict, tmp_path)
            os.replace(tmp_path, dst)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        logger.info("Checkpoint saved at epoch %d -> %s", epoch, dst)

    # Each rank saves its own replay buffer (contents differ per rank)
    if replay_buffer is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        replay_buffer.save(os.path.join(ckpt_dir, f"replay_buffer_rank{rank}.npz"))

    dist.barrier()


def load_checkpoint(
    resume_from: str,
    policy,
    replay_buffer=None,
    rank: int = 0,
) -> dict | None:
    """Load checkpoint and restore policy state. All ranks must call."""
    ckpt_path = os.path.join(resume_from, "checkpoint", "latest.pt")
    if not os.path.exists(ckpt_path):
        if rank == 0:
            logger.warning("No checkpoint found at %s", ckpt_path)
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    _load_fsdp_state_dict(policy.get_model(), ckpt.pop("model_state_dict"))
    policy.load_state_dict(ckpt.pop("policy_state"))

    if replay_buffer is not None:
        buf_path = os.path.join(resume_from, "checkpoint", f"replay_buffer_rank{rank}.npz")
        if os.path.exists(buf_path):
            replay_buffer.load(buf_path)
            logger.info("Rank %d: replay buffer loaded (%d transitions)", rank, len(replay_buffer))

    meta = {
        "epoch": ckpt["epoch"],
        "wandb_run_id": ckpt.get("wandb_run_id"),
    }
    del ckpt

    if rank == 0:
        logger.info("Checkpoint loaded from %s (epoch=%d)", ckpt_path, meta["epoch"])

    dist.barrier()
    return meta
