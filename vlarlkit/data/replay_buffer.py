import numpy as np
import torch


class ReplayBuffer:
    """In-memory replay buffer with FIFO eviction (ring buffer).

    Storage: dict[str, np.ndarray], flat keys, each shape [max_size, ...].
    obs/next_obs nested dicts are expected to be flattened to "obs/key",
    "next_obs/key" before calling add(). Torch tensors are auto-converted
    to numpy on add().
    """

    def __init__(self, max_size: int, seed: int = 0):
        self._max_size = max_size
        self._size = 0
        self._ptr = 0
        self._storage: dict[str, np.ndarray] = {}
        self._initialized = False
        self._rng = np.random.RandomState(seed)

    def _init_storage(self, transitions: dict[str, np.ndarray]) -> None:
        """Allocate fixed-size arrays on first add, based on data shapes."""
        for key, val in transitions.items():
            self._storage[key] = np.zeros(
                (self._max_size, *val.shape[1:]), dtype=val.dtype
            )
        self._initialized = True

    def add(self, transitions: dict[str, np.ndarray | torch.Tensor]) -> None:
        """Add flattened transitions {key: [N, ...]} to buffer with FIFO wrap.

        Torch tensors are auto-converted to numpy.
        """
        transitions = {
            k: v.numpy() if isinstance(v, torch.Tensor) else v
            for k, v in transitions.items()
        }
        if not self._initialized:
            self._init_storage(transitions)

        n = next(iter(transitions.values())).shape[0]

        if self._ptr + n <= self._max_size:
            for k, v in transitions.items():
                self._storage[k][self._ptr : self._ptr + n] = v
        else:
            overflow = (self._ptr + n) - self._max_size
            first = n - overflow
            for k, v in transitions.items():
                self._storage[k][self._ptr : self._max_size] = v[:first]
                self._storage[k][:overflow] = v[first:]

        self._ptr = (self._ptr + n) % self._max_size
        self._size = min(self._size + n, self._max_size)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Uniform random sample of transitions."""
        indices = self._rng.randint(0, self._size, size=batch_size)
        return {k: v[indices] for k, v in self._storage.items()}

    def ready(self, min_size: int = 1) -> bool:
        return self._size >= min_size

    def save(self, path: str) -> None:
        """Save buffer to .npz file. Only saves populated portion."""
        save_dict = {"_ptr": self._ptr, "_size": self._size}
        for k, v in self._storage.items():
            save_dict[f"storage/{k}"] = v[:self._size]
        np.savez(path, **save_dict)

    def load(self, path: str) -> None:
        """Load buffer from .npz file."""
        data = np.load(path, allow_pickle=True)
        self._ptr = int(data["_ptr"])
        self._size = int(data["_size"])
        for key in data.files:
            if key.startswith("storage/"):
                storage_key = key[len("storage/"):]
                arr = data[key]
                self._storage[storage_key] = np.zeros(
                    (self._max_size, *arr.shape[1:]), dtype=arr.dtype
                )
                self._storage[storage_key][:self._size] = arr
        self._initialized = True

    def __len__(self) -> int:
        return self._size
