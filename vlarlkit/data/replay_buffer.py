import numpy as np


class ReplayBuffer:
    """In-memory replay buffer with FIFO eviction (ring buffer).

    Storage: dict[str, np.ndarray], flat keys, each shape [max_size, ...].
    obs/next_obs nested dicts are expected to be flattened to "obs/key",
    "next_obs/key" before calling add().
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

    def add(self, transitions: dict[str, np.ndarray]) -> None:
        """Add flattened transitions {key: [N, ...]} to buffer with FIFO wrap."""
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

    def ready(self) -> bool:
        return self._size > 0

    def __len__(self) -> int:
        return self._size
