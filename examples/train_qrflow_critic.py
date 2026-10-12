import copy
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from vlarlkit.data.qrflow_replay import QRFlowReplay
from vlarlkit.models.modules.qrflow_critic import ActionInjectedEnsembleQ, reference_td_loss
from vlarlkit.utils.qrflow import save_checkpoint, set_seed


def train(cfg, output_dir, device='cuda:0'):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    assert cfg.runner.resume_from or not (directory / 'checkpoint.pt').exists(), 'Use runner.resume_from to resume training'
    assert cfg.runner.replay_path, 'Critic training requires runner.replay_path'
    replay = QRFlowReplay(cfg.runner.replay_path)
    assert replay.metadata['horizon'] == int(cfg.algorithm.n_step)
    assert replay.metadata['gamma'] == float(cfg.algorithm.gamma)
    assert replay.metadata['iteration'] == int(cfg.runner.iteration)
    set_seed(int(cfg.runner.seed) + int(cfg.runner.iteration))
    data = replay.critic_tensors(device)
    model_config = dict(feature_dim=data['features'].shape[-1],
                        action_dim=data['executed'].shape[-1] * int(cfg.algorithm.n_step),
                        hidden_dims=list(cfg.algorithm.critic_hidden_dims), heads=10)
    critic = ActionInjectedEnsembleQ(**model_config).to(device)
    target = copy.deepcopy(critic).requires_grad_(False).eval()
    optimizer = torch.optim.Adam(critic.parameters(), lr=float(cfg.training.optim.value_lr))
    generator = torch.Generator(device=device).manual_seed(int(cfg.runner.seed) + int(cfg.runner.iteration))
    training_config = dict(gamma=float(cfg.algorithm.gamma), horizon=int(cfg.algorithm.n_step),
                           batch_size=int(cfg.algorithm.critic_global_batch_size), tau=float(cfg.algorithm.tau),
                           lr=float(cfg.training.optim.value_lr), grad_clip=float(cfg.training.optim.critic_clip_grad))
    step = previous_steps = 0
    checkpoint_path = cfg.runner.resume_from or cfg.runner.critic_checkpoint
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert state['model_config'] == model_config
        assert state['training_config'] == training_config
        critic.load_state_dict(state['q_state_dict'])
        target.load_state_dict(state['target_q_state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        if cfg.runner.resume_from:
            assert state['replay_path'] == replay.path and state['reference'] == replay.metadata['reference']
            assert state['iteration'] == int(cfg.runner.iteration)
            step = int(state['step'])
            previous_steps = int(state['total_steps']) - step
            generator.set_state(state['generator_state'].cpu())
        else:
            assert state['completed'] and state['iteration'] == int(cfg.runner.iteration) - 1
            previous_steps = int(state['total_steps'])
    elif int(cfg.runner.iteration) > 1:
        raise ValueError('Subsequent rounds require runner.critic_checkpoint to continue Q, target Q and Adam')
    updates = int(cfg.algorithm.critic_updates)
    assert updates > 0 and 0 <= step <= updates
    if step == updates:
        save_checkpoint(directory / 'checkpoint.pt', state)
        replay.close()
        return
    loss_function = torch.compile(reference_td_loss) if cfg.training.compile_critic else reference_td_loss
    action_offsets = torch.arange(int(cfg.algorithm.n_step), device=device)[None]
    batch_size = int(cfg.algorithm.critic_global_batch_size)
    for step in range(step + 1, updates + 1):
        episode = torch.randint(len(data['counts']), (batch_size,), generator=generator, device=device)
        window = (torch.rand(batch_size, generator=generator, device=device) * data['counts'][episode]).long()
        index = data['offsets'][episode] + window
        current, following = data['current'][index], data['next'][index]
        actions = data['executed'][data['actions'][index, None] + action_offsets]
        loss, predictions, targets, _ = loss_function(
            critic, target, data['features'][current], actions, data['features'][following],
            data['bank'][following], data['rewards'][index], data['discounts'][index], data['terminations'][index],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), float(cfg.training.optim.critic_clip_grad))
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError('Nonfinite critic gradient; optimizer was not updated')
        optimizer.step()
        with torch.no_grad():
            for old, new in zip(target.parameters(), critic.parameters()):
                old.lerp_(new, float(cfg.algorithm.tau))
        if step % int(cfg.runner.log_interval) == 0 or step == updates:
            print(dict(step=step, q_loss=float(loss.detach()) / 10,
                       q_mean=float(predictions.detach().mean()), target_mean=float(targets.mean())), flush=True)
        if step % int(cfg.runner.save_interval) == 0 or step == updates:
            save_checkpoint(directory / 'checkpoint.pt', dict(
                iteration=int(cfg.runner.iteration), step=step, total_steps=previous_steps + step,
                completed=step == updates, model_config=model_config, replay_path=replay.path,
                reference=replay.metadata['reference'], q_state_dict=critic.state_dict(),
                target_q_state_dict=target.state_dict(), optimizer=optimizer.state_dict(),
                generator_state=generator.get_state(),
                training_config=training_config,
            ))
    replay.close()


@hydra.main(config_path='configs', config_name='libero_10_qrflow_pi05', version_base=None)
def main(cfg: DictConfig):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    train(cfg, HydraConfig.get().runtime.output_dir, device)


if __name__ == '__main__':
    main()
