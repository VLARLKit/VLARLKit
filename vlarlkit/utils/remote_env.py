"""
RemoteEnv — a drop-in proxy that forwards environment calls to an EnvServer via ZMQ.
"""

import logging
import pickle
from typing import Any, Optional, Union

import numpy as np
import zmq

logger = logging.getLogger("vlarlkit.remote_env")


class RemoteEnv:
    """Drop-in replacement for LiberoEnv that communicates with a remote EnvServer."""

    def __init__(
        self,
        host: str,
        port: int,
        env_mode: str,
        recv_timeout_ms: int = 300_000,
        max_retries: int = 1,
    ):
        """
        Args:
            host: EnvServer hostname or IP.
            port: EnvServer port.
            env_mode: "train" or "eval" — selects which env instance on the server.
            recv_timeout_ms: Timeout in milliseconds for receiving a response.
                             Defaults to 300 000 (5 minutes).
            max_retries: Max attempts per call before raising TimeoutError.
                         1 means no retry (fail immediately on first timeout).
        """
        self.env_mode = env_mode
        self._host = host
        self._port = port
        self._recv_timeout_ms = recv_timeout_ms
        self._max_retries = max_retries
        self._ctx = zmq.Context()
        self._socket: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        """Create (or recreate) the REQ socket and connect to the server."""
        if self._socket is not None:
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.close()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        self._socket.connect(f"tcp://{self._host}:{self._port}")

    def _send_and_recv(self, msg: dict, max_retries: int | None = None) -> Any:
        """Send a request and wait for the response, with timeout and retry."""
        retries = max_retries if max_retries is not None else self._max_retries
        method = msg.get("method", "unknown")

        for attempt in range(1, retries + 1):
            self._socket.send(pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL))
            try:
                response = pickle.loads(self._socket.recv())
            except zmq.Again:
                # REQ socket is now in a broken state; reconnect before retry.
                self._connect()
                if attempt < retries:
                    logger.warning(
                        "RemoteEnv '%s' timed out (attempt %d/%d, %ds). "
                        "Reconnected, retrying… [%s:%d, env_mode=%s]",
                        method, attempt, retries,
                        self._recv_timeout_ms // 1000,
                        self._host, self._port, self.env_mode,
                    )
                    continue
                raise TimeoutError(
                    f"RemoteEnv: '{method}' call to {self._host}:{self._port} "
                    f"(env_mode={self.env_mode}) timed out after "
                    f"{retries} attempts ({self._recv_timeout_ms / 1000:.0f}s each)"
                )
            if "error" in response:
                raise RuntimeError(f"EnvServer error: {response['error']}")
            return response["result"]

    def _call(self, method: str, **kwargs) -> Any:
        msg = {"env_mode": self.env_mode, "method": method, "kwargs": kwargs}
        return self._send_and_recv(msg)

    def _get_attr(self, name: str) -> Any:
        msg = {"env_mode": self.env_mode, "method": "get_attr", "kwargs": {"name": name}}
        return self._send_and_recv(msg)

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
