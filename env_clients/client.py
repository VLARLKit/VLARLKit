"""
Environment Client Server — hosts environments and serves requests via ZMQ.

Usage:
    python -m env_clients.client \
        --config examples/configs/libero_10_ppo_pi05.yaml \
        --host 0.0.0.0 --port 5550 \
        --rank 0 --world_size 2
"""

import argparse
import importlib
import io
import logging
import os
import pickle
import signal
import sys

import torch
import zmq
from hydra import compose, initialize_config_dir


logging.basicConfig(
    level=logging.INFO,
    format="[EnvServer %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ENV_REGISTRY = {
    "libero": "env_clients.libero.libero_env.LiberoEnv",
    "maniskill": "env_clients.maniskill.maniskill_env.ManiskillEnv",
}


def import_env_class(env_type: str):
    if env_type not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown env_type '{env_type}'. Available: {list(ENV_REGISTRY.keys())}"
        )
    module_path, class_name = ENV_REGISTRY[env_type].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class EnvServer:
    """ZMQ REP server that hosts train/eval environments for a single rank."""

    CALLABLE_METHODS = frozenset({
        "reset", "step", "chunk_step",
        "update_reset_state_ids", "flush_video",
    })

    READABLE_ATTRS = frozenset({
        "num_envs", "elapsed_steps", "auto_reset",
    })

    def __init__(self, host: str, port: int, envs: dict):
        self.envs = envs
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        logger.info("Listening on tcp://%s:%d", host, port)
        logger.info("Registered env_modes: %s", list(self.envs.keys()))

    @staticmethod
    def _safe_pickle_loads(data):
        """pickle.loads that remaps CUDA tensors to the local default device."""
        _orig = torch.storage._load_from_bytes

        def _patched(b):
            return torch.load(io.BytesIO(b), map_location="cuda", weights_only=False)

        torch.storage._load_from_bytes = _patched
        try:
            return pickle.loads(data)
        finally:
            torch.storage._load_from_bytes = _orig

    def serve_forever(self):
        logger.info("Ready to serve requests.")
        try:
            while True:
                raw = self.socket.recv()
                msg = self._safe_pickle_loads(raw)
                response = self._dispatch(msg)
                self.socket.send(pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL))
        except KeyboardInterrupt:
            logger.info("Shutting down.")
        finally:
            self.socket.close()
            self.ctx.term()

    def _dispatch(self, msg: dict):
        env_mode = msg.get("env_mode")
        method = msg.get("method")
        kwargs = msg.get("kwargs", {})

        if env_mode not in self.envs:
            return {"error": f"Unknown env_mode '{env_mode}'. Available: {list(self.envs.keys())}"}

        env = self.envs[env_mode]

        if method == "get_attr":
            attr_name = kwargs.get("name")
            if attr_name not in self.READABLE_ATTRS:
                return {"error": f"Attribute '{attr_name}' is not readable."}
            return {"result": getattr(env, attr_name)}

        if method not in self.CALLABLE_METHODS:
            return {"error": f"Method '{method}' is not allowed."}

        try:
            result = getattr(env, method)(**kwargs)
            return {"result": result}
        except Exception as e:
            logger.exception("Error calling %s.%s", env_mode, method)
            return {"error": str(e)}


def create_envs(cfg, rank: int, world_size: int) -> dict:
    """Create train and eval environment instances from config."""
    envs = {}
    env_type = cfg.env.train.get("env_type", "libero")
    EnvClass = import_env_class(env_type)

    for mode in ("train", "eval"):
        env_cfg = cfg.env.get(mode)
        if env_cfg is None:
            continue
        total_num_envs = int(env_cfg.get("total_num_envs"))
        num_envs = total_num_envs // world_size
        assert num_envs > 0, (
            f"total_num_envs ({total_num_envs}) must be >= world_size ({world_size})"
        )
        logger.info("Creating %s env: %s with %d envs (rank %d)", mode, env_type, num_envs, rank)
        envs[mode] = EnvClass(env_cfg, num_envs=num_envs, total_num_processes=world_size, rank=rank)
        logger.info("Created %s env successfully.", mode)

    return envs


def main():
    parser = argparse.ArgumentParser(description="Environment Client Server")
    parser.add_argument("--config", type=str, required=True, help="Path to Hydra config YAML")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5550, help="Port to bind (default: 5550)")
    parser.add_argument("--rank", type=int, default=0, help="Rank of this env client")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of ranks")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)

    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    envs = create_envs(cfg, rank=args.rank, world_size=args.world_size)
    server = EnvServer(host=args.host, port=args.port, envs=envs)
    server.serve_forever()


if __name__ == "__main__":
    main()
