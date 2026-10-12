from pathlib import Path

import h5py
import numpy as np
import torch

from vlarlkit.data.qrflow_replay import n_step_windows


def _append(file, key, value):
    value = np.asarray(value)
    if key not in file:
        options = dict(chunks=(1,) + value.shape, maxshape=(None,) + value.shape)
        if value.dtype == np.uint8:
            options.update(compression='gzip', compression_opts=1)
        file.create_dataset(key, shape=(0,) + value.shape, dtype=value.dtype, **options)
    dataset = file[key]
    dataset.resize(len(dataset) + 1, axis=0)
    dataset[-1] = value


@torch.inference_mode()
def collect_episodes(cfg, model, env, output_dir, rank=0, world_size=1):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    count = int(cfg.algorithm.episodes_per_iteration)
    ids = np.array_split(np.arange(count), world_size)[rank]
    num_envs = env.num_envs
    horizon = int(cfg.model.num_action_chunks)
    assert horizon == model.config.action_horizon == int(cfg.algorithm.n_step) == 10
    assert len(ids) % num_envs == 0
    assert int(cfg.env.train.max_episode_steps) % horizon == 0
    rows = []
    for start in range(0, len(ids), num_envs):
        reset_ids = ids[start:start + num_envs]
        obs, _ = env.reset(reset_state_ids=reset_ids)
        files = [h5py.File(directory / f'episode-{reset_id:03d}.h5', 'x') for reset_id in reset_ids]
        instructions = list(obs['task_descriptions'])
        active = np.ones(num_envs, dtype=bool)
        try:
            for slot, file in enumerate(files):
                for key in ('main_images', 'wrist_images', 'states'):
                    _append(file, key, obs[key][slot])
            for _ in range(int(cfg.env.train.max_episode_steps) // horizon):
                observation, _ = model.prepare_observation(obs)
                context = model.encode_prefix(observation)
                latent = model.sample_actions(observation, context=context, stochastic=True)['actions']
                command = model.output_transform(dict(actions=latent, state=observation.state))['actions']
                command = np.asarray(command.cpu() if torch.is_tensor(command) else command).astype(np.float32)
                assert command.shape == (num_envs, horizon, 7) and np.isfinite(command).all()
                executed = command.copy()
                executed[..., :6] = np.clip(executed[..., :6], -1, 1)
                live = active.copy()
                done = np.zeros(num_envs, dtype=bool)
                for step in range(horizon):
                    normalized = model.normalize_actions(obs, executed[:, step:step + 1])
                    normalized = np.asarray(normalized.cpu() if torch.is_tensor(normalized) else normalized)[:, 0]
                    next_obs, _, terms, truncs, _ = env.chunk_step(executed[:, step:step + 1])
                    terms, truncs = np.asarray(terms)[:, 0], np.asarray(truncs)[:, 0]
                    done |= terms | truncs
                    for slot in np.flatnonzero(live):
                        for key in ('main_images', 'wrist_images', 'states'):
                            _append(files[slot], key, next_obs[key][slot])
                        _append(files[slot], 'actions', normalized[slot])
                        _append(files[slot], 'success', terms[slot])
                        _append(files[slot], 'truncations', truncs[slot])
                    obs = next_obs
                active &= ~done
                if not active.any():
                    break
            for slot, file in enumerate(files):
                windows = n_step_windows(file['success'][:], file['truncations'][:], horizon, float(cfg.algorithm.gamma))
                for key, value in windows.items():
                    file.create_dataset('window_' + key, data=value)
                file.attrs['instruction'] = instructions[slot]
                rows.append(dict(path=str(Path(file.filename).resolve()), reset_id=int(reset_ids[slot]),
                                 iteration=int(cfg.runner.iteration),
                                 task_id=int(reset_ids[slot]) // int(cfg.algorithm.episodes_per_task),
                                 instruction=instructions[slot], observations=len(file['states']),
                                 steps=len(file['actions']), windows=len(windows['starts']),
                                 success=bool(file['success'][:].any())))
        finally:
            for file in files:
                file.close()
    return rows


@torch.inference_mode()
def refresh_reference_actions(model, rows, output_dir, rank=0, world_size=1, reference=None, batch_size=8):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(rank, len(rows), world_size):
        row = rows[index]
        bank_path = directory / f'actions-{index:04d}.npy'
        feature_path = directory / f'features-{index:04d}.npy'
        assert not bank_path.exists() and not feature_path.exists(), 'Use a new output directory to refresh replay'
        bank = np.lib.format.open_memmap(bank_path, mode='w+', dtype=np.float32,
                                        shape=(row['observations'], 8, 10, 7))
        features = None
        with h5py.File(row['path'], 'r') as file:
            for start in range(0, row['observations'], batch_size):
                stop = min(start + batch_size, row['observations'])
                raw = {key: file[key][start:stop] for key in ('main_images', 'wrist_images', 'states')}
                raw['task_descriptions'] = [row['instruction']] * (stop - start)
                observation, _ = model.prepare_observation(raw)
                context = model.encode_prefix(observation)
                value = context.features.cpu().numpy()
                if features is None:
                    features = np.lib.format.open_memmap(feature_path, mode='w+', dtype=np.float32,
                                                         shape=(row['observations'], value.shape[-1]))
                features[start:stop] = value
                bank[start:stop] = model.reference_actions(context).cpu().numpy()
        bank.flush()
        features.flush()
        assert np.isfinite(bank).all() and np.isfinite(features).all()
        records.append(dict(index=index, row=dict(row, features=str(feature_path.resolve()),
                                                  reference_actions=str(bank_path.resolve()), reference=reference)))
    return records
