"""
RemoteEnv — a drop-in proxy that forwards environment calls to an EnvServer via ZMQ.
"""

import pickle
from typing import Any, Optional, Union

import numpy as np
import zmq


class RemoteEnv:
    """Drop-in replacement for LiberoEnv that communicates with a remote EnvServer."""

    def __init__(self, host: str, port: int, env_mode: str):
        """
        Args:
            host: EnvServer hostname or IP.
            port: EnvServer port.
            env_mode: "train" or "eval" — selects which env instance on the server.
        """
        self.env_mode = env_mode
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def _call(self, method: str, **kwargs) -> Any:
        msg = {"env_mode": self.env_mode, "method": method, "kwargs": kwargs}
        self._socket.send(pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL))
        response = pickle.loads(self._socket.recv())
        if "error" in response:
            raise RuntimeError(f"EnvServer error: {response['error']}")
        return response["result"]

    def _get_attr(self, name: str) -> Any:
        msg = {"env_mode": self.env_mode, "method": "get_attr", "kwargs": {"name": name}}
        self._socket.send(pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL))
        response = pickle.loads(self._socket.recv())
        if "error" in response:
            raise RuntimeError(f"EnvServer error: {response['error']}")
        return response["result"]

    # ---- Environment interface (matches LiberoEnv) ----

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        kwargs = {}
        if env_idx is not None:
            kwargs["env_idx"] = env_idx
        if reset_state_ids is not None:
            kwargs["reset_state_ids"] = reset_state_ids
        return self._call("reset", **kwargs)

    def step(self, actions=None, auto_reset=True):
        return self._call("step", actions=actions, auto_reset=auto_reset)

    def chunk_step(self, chunk_actions):
        return self._call("chunk_step", chunk_actions=chunk_actions)

    def update_reset_state_ids(self):
        return self._call("update_reset_state_ids")

    def flush_video(self, video_sub_dir: Optional[str] = None):
        return self._call("flush_video", video_sub_dir=video_sub_dir)

    @property
    def num_envs(self) -> int:
        return self._get_attr("num_envs")

    @property
    def elapsed_steps(self) -> np.ndarray:
        return self._get_attr("elapsed_steps")

    @property
    def auto_reset(self) -> bool:
        return self._get_attr("auto_reset")

    def close(self):
        self._socket.close()
        self._ctx.term()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
