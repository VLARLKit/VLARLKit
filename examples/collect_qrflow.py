import json
import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from vlarlkit.rollouts.qrflow_rollout import collect_episodes, refresh_reference_actions
from vlarlkit.utils.qrflow import load_actor, reference_path, save_json, set_seed
from vlarlkit.utils.remote_env import RemoteEnv


def collect(cfg, output_dir, rank=0, world_size=1, device='cuda:0'):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / 'replay.json').exists(), 'Use a new output directory for each round'
    iteration = int(cfg.runner.iteration)
    assert iteration >= 1
    assert int(cfg.algorithm.episodes_per_iteration) == 10 * int(cfg.algorithm.episodes_per_task)
    set_seed(int(cfg.runner.seed) + iteration * 10000 + rank)
    model, _ = load_actor(cfg, device)
    reference = reference_path(cfg)
    previous = []
    if cfg.runner.previous_replay:
        metadata = json.loads(Path(cfg.runner.previous_replay).read_text())
        assert metadata['iteration'] == iteration - 1
        previous = metadata['episodes']
    else:
        assert iteration == 1, 'Subsequent rounds require the cumulative replay'

    rows = []
    if cfg.runner.collect_data:
        env = RemoteEnv(cfg.env.env_client_host, int(cfg.env.env_client_base_port) + rank, 'train')
        try:
            rows = collect_episodes(cfg, model, env, directory / 'episodes', rank, world_size)
        finally:
            env.close()
    else:
        assert previous, 'Refreshing actions requires runner.previous_replay'
    part = directory / f'episodes-rank{rank}.json'
    assert not part.exists(), 'The collection stage has already been started'
    save_json(part, rows)
    if dist.is_initialized():
        dist.barrier()
    added = []
    for worker in range(world_size):
        added.extend(json.loads((directory / f'episodes-rank{worker}.json').read_text()))
    added.sort(key=lambda row: row['reset_id'])
    if cfg.runner.collect_data:
        assert [row['reset_id'] for row in added] == list(range(int(cfg.algorithm.episodes_per_iteration)))
        assert all(sum(row['task_id'] == task for row in added) == int(cfg.algorithm.episodes_per_task)
                   for task in range(10))
    cumulative = previous + added
    assert len({(row['iteration'], row['reset_id']) for row in cumulative}) == len(cumulative)

    records = refresh_reference_actions(model, cumulative, directory / 'cache', rank, world_size, reference)
    save_json(directory / f'cache-rank{rank}.json', records)
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        records = []
        for worker in range(world_size):
            records.extend(json.loads((directory / f'cache-rank{worker}.json').read_text()))
        records.sort(key=lambda record: record['index'])
        assert [record['index'] for record in records] == list(range(len(cumulative)))
        successes = sum(row['success'] for row in added)
        save_json(directory / 'replay.json', dict(
            iteration=iteration, reference=reference, episodes=[record['row'] for record in records],
            gamma=float(cfg.algorithm.gamma), horizon=int(cfg.algorithm.n_step),
            collection=dict(episodes=len(added), successes=successes,
                            success_rate=successes / len(added) if added else None),
        ))
        print(f'Round {iteration}: collected {len(added)} episodes, {successes} successes; replay {len(cumulative)} episodes', flush=True)


@hydra.main(config_path='configs', config_name='libero_10_qrflow_pi05', version_base=None)
def main(cfg: DictConfig):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(local_rank)
    if 'WORLD_SIZE' in os.environ and not dist.is_initialized():
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    try:
        collect(cfg, HydraConfig.get().runtime.output_dir, rank, world_size, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
