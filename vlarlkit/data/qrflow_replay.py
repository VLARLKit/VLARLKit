import json
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch


def n_step_windows(success, truncations, horizon=10, gamma=0.999):
    success = np.asarray(success, dtype=bool)
    truncations = np.asarray(truncations, dtype=bool)
    assert success.shape == truncations.shape and len(success) >= horizon
    hits = np.flatnonzero(success)
    last = min(int(hits[0]) if len(hits) else len(success) - horizon, len(success) - horizon)
    starts = np.arange(last + 1, dtype=np.int64)
    terminal = success[starts[:, None] + np.arange(horizon)]
    done = terminal.any(1)
    duration = np.where(done, terminal.argmax(1) + 1, horizon)
    return dict(starts=starts, next_indices=starts + duration,
                rewards=(-np.cumsum(gamma ** np.arange(horizon))[duration - 1]).astype(np.float32),
                discounts=(gamma ** duration).astype(np.float32), terminations=done,
                truncations=truncations[starts + duration - 1] & ~done)


class QRFlowReplay:
    """Cumulative episodes on disk; sample an episode, then an eligible window."""

    def __init__(self, path, seed=0):
        self.path = str(Path(path).resolve())
        self.metadata = json.loads(Path(path).read_text())
        self._rows = self.metadata['episodes']
        assert self._rows and all(row['reference'] == self.metadata['reference'] for row in self._rows)
        self._rng = np.random.RandomState(seed)
        self._files = OrderedDict()
        self._banks = OrderedDict()

    def critic_tensors(self, device):
        features, banks, actions, counts = [], [], [], []
        fields = {key: [] for key in ('current', 'next', 'actions', 'rewards', 'discounts', 'terminations')}
        observation_offset = action_offset = 0
        for row in self._rows:
            feature = np.load(row['features'])
            bank = np.load(row['reference_actions'])
            assert len(feature) == len(bank) == row['observations']
            assert bank.shape[1] == 8
            features.append(feature)
            banks.append(bank)
            with h5py.File(row['path'], 'r') as file:
                executed = file['actions'][:]
                starts = file['window_starts'][:]
                actions.append(executed)
                fields['current'].append(starts + observation_offset)
                fields['next'].append(file['window_next_indices'][:] + observation_offset)
                fields['actions'].append(starts + action_offset)
                for key in ('rewards', 'discounts', 'terminations'):
                    fields[key].append(file['window_' + key][:])
                counts.append(len(starts))
            observation_offset += len(feature)
            action_offset += len(executed)
        data = {key: torch.from_numpy(np.concatenate(value)).to(device) for key, value in fields.items()}
        data.update(features=torch.from_numpy(np.concatenate(features)).to(device),
                    bank=torch.from_numpy(np.concatenate(banks)).to(device),
                    executed=torch.from_numpy(np.concatenate(actions)).to(device),
                    counts=torch.tensor(counts, dtype=torch.long, device=device),
                    offsets=torch.tensor(np.cumsum([0] + counts[:-1]), dtype=torch.long, device=device))
        return data

    def _file(self, episode):
        if episode not in self._files:
            if len(self._files) >= 64:
                _, file = self._files.popitem(last=False)
                file.close()
            self._files[episode] = h5py.File(self._rows[episode]['path'], 'r')
        self._files.move_to_end(episode)
        return self._files[episode]

    def sample_actor(self, batch_size, process):
        episodes = self._rng.randint(len(self._rows), size=batch_size)
        starts = (self._rng.rand(batch_size) * np.array([self._rows[i]['windows'] for i in episodes])).astype(np.int64)
        raw = {key: [] for key in ('main_images', 'wrist_images', 'states', 'task_descriptions')}
        references = []
        for episode, start in zip(episodes, starts):
            episode = int(episode)
            row, file = self._rows[episode], self._file(episode)
            start = int(file['window_starts'][start])
            for key in ('main_images', 'wrist_images', 'states'):
                raw[key].append(file[key][start])
            raw['task_descriptions'].append(row['instruction'])
            if episode not in self._banks:
                if len(self._banks) >= 64:
                    self._banks.popitem(last=False)
                self._banks[episode] = np.load(row['reference_actions'], mmap_mode='r')
            self._banks.move_to_end(episode)
            references.append(self._banks[episode][start])
        for key in ('main_images', 'wrist_images', 'states'):
            raw[key] = np.stack(raw[key])
        processed = process(raw)
        result = {}
        for key, value in processed.items():
            if isinstance(value, dict):
                for name, tensor in value.items():
                    if key == 'image' and not processed['image_mask'][name].any():
                        continue
                    result[f'obs/{key}/{name}'] = tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
            else:
                result[f'obs/{key}'] = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        result['reference_actions'] = np.stack(references)
        return result

    def rng_state(self):
        return self._rng.get_state()

    def load_rng_state(self, state):
        self._rng.set_state(state)

    def close(self):
        for file in self._files.values():
            file.close()
        self._files.clear()
        self._banks.clear()
