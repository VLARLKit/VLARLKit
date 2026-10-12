import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from vlarlkit.data.qrflow_replay import QRFlowReplay
from vlarlkit.models.modules.qrflow_critic import ActionInjectedEnsembleQ
from vlarlkit.policies.qrflow import QRFlowPolicy
from vlarlkit.utils.qrflow import load_actor, reference_path, save_checkpoint, set_seed


def train(cfg, output_dir, rank=0, world_size=1, device='cuda:0'):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    assert cfg.runner.resume_from or not (directory / 'checkpoint.pt').exists(), 'Use runner.resume_from to resume training'
    assert cfg.runner.replay_path and cfg.runner.critic_checkpoint
    set_seed(int(cfg.runner.seed) + int(cfg.runner.iteration) * 10000 + rank)
    model, previous = load_actor(cfg, device)
    replay = QRFlowReplay(cfg.runner.replay_path, int(cfg.runner.seed) + int(cfg.runner.iteration) * 10000 + rank)
    assert replay.metadata['reference'] == reference_path(cfg)
    assert replay.metadata['iteration'] == int(cfg.runner.iteration)
    state = torch.load(cfg.runner.critic_checkpoint, map_location='cpu', weights_only=False)
    assert state['completed'], 'Actor training requires a completed critic checkpoint'
    assert state['iteration'] == int(cfg.runner.iteration)
    assert state['replay_path'] == replay.path and state['reference'] == replay.metadata['reference']
    critic = ActionInjectedEnsembleQ(**state['model_config']).to(device)
    critic.load_state_dict(state['target_q_state_dict'])
    policy = QRFlowPolicy(cfg, model, critic, rank)
    previous_steps = int(previous['total_steps']) if previous else 0
    if cfg.runner.resume_from:
        state = torch.load(cfg.runner.resume_from, map_location='cpu', weights_only=False)
        assert state['iteration'] == int(cfg.runner.iteration)
        assert state['replay_path'] == replay.path and state['reference'] == replay.metadata['reference']
        assert len(state['replay_rng_states']) == world_size, 'Resume requires the same world size'
        policy.load_state_dict(state)
        replay.load_rng_state(state['replay_rng_states'][rank])
    updates = int(cfg.algorithm.actor_updates)
    assert updates > 0 and policy.update_step <= updates
    if policy.update_step == updates:
        if rank == 0:
            save_checkpoint(directory / 'checkpoint.pt', state)
        replay.close()
        return
    process = lambda raw: model.input_transform(model.obs_processor(raw), transpose=False)
    try:
        for step in range(policy.update_step + 1, updates + 1):
            metrics = policy.run_update(replay.sample_actor(policy.replay_batch_size(), process))
            if rank == 0 and (step % int(cfg.runner.log_interval) == 0 or step == updates):
                print(dict(step=step, **metrics), flush=True)
            if step % int(cfg.runner.save_interval) == 0 or step == updates:
                state = policy.state_dict()
                rng_states = [None] * world_size
                if dist.is_initialized():
                    dist.all_gather_object(rng_states, replay.rng_state())
                else:
                    rng_states[0] = replay.rng_state()
                if rank == 0:
                    state.update(iteration=int(cfg.runner.iteration), completed=step == updates,
                                 total_steps=previous_steps + step, replay_path=replay.path,
                                 reference=replay.metadata['reference'], replay_rng_states=rng_states)
                    save_checkpoint(directory / 'checkpoint.pt', state)
                if dist.is_initialized():
                    dist.barrier()
    finally:
        replay.close()


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
        train(cfg, HydraConfig.get().runtime.output_dir, rank, world_size, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
