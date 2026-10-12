import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn

from examples import collect_qrflow, train_qrflow_actor, train_qrflow_critic
from vlarlkit.data.qrflow_replay import QRFlowReplay, n_step_windows
from vlarlkit.models.modules.flow_sde import chain_coefficients, sample_single_sde
from vlarlkit.models.modules.qrflow_critic import ActionInjectedEnsembleQ, reference_td_targets
from vlarlkit.policies.qrflow import QRFlowPolicy, return_moments, return_statistics


class Context:
    def __init__(self, features):
        self.features = features

    def select(self, indices):
        return Context(self.features[indices])


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.config = SimpleNamespace(num_steps=5, action_horizon=10, action_chunk=10,
                                      action_dim=7, action_env_dim=7)
        self.fast_actor = nn.Linear(7, 7)
        self._reference_actor = copy.deepcopy(self.fast_actor).requires_grad_(False)
        self.backbone = nn.Linear(2, 2).requires_grad_(False)

    def load_actor_state_dict(self, state, update_reference=True):
        self.fast_actor.load_state_dict(state)
        if update_reference:
            self._reference_actor.load_state_dict(state)

    def reference_state_dict(self):
        return copy.deepcopy(self._reference_actor.state_dict())

    def load_reference_state_dict(self, state):
        self._reference_actor.load_state_dict(state)

    def prepare_observation(self, raw):
        return SimpleNamespace(state=torch.as_tensor(raw['states'])), {}

    def encode_prefix(self, observation):
        return Context(self.backbone(observation.state.float()))

    def prefix_from_replay(self, obs):
        return self.encode_prefix(SimpleNamespace(state=obs['state']))

    def velocity(self, positions, times, context, reference=False):
        actor = self._reference_actor if reference else self.fast_actor
        return actor(positions) + context.features.mean(-1)[:, None, None] * 0.01

    def critic_actions(self, actions):
        return actions

    def canonical_actions(self, actions):
        return torch.cat((actions[..., :6].clamp(-1, 1), actions[..., 6:7]), dim=-1)

    def sample_actions(self, observation, context=None, stochastic=False):
        context = context or self.encode_prefix(observation)
        batch = len(observation.state)
        initial = torch.randn(batch, 1, 10, 7)
        velocity = lambda x, t, ref: self.velocity(x[:, 0], t[:, 0], context, ref)[:, None]
        actions = sample_single_sde(velocity, initial, torch.randn_like(initial), torch.randint(5, (batch,)))[:, 0]
        return dict(actions=actions)

    def reference_actions(self, context):
        batch = len(context.features)
        expanded = context.select(torch.arange(batch).repeat_interleave(8))
        actions = torch.randn(batch * 8, 10, 7)
        for node in range(5):
            actions = actions + self.velocity(actions, torch.full((batch * 8,), node / 5), expanded, True) / 5
        return actions.reshape(batch, 8, 10, 7)

    def normalize_actions(self, obs, actions):
        return torch.as_tensor(actions)

    def output_transform(self, value):
        return value

    def obs_processor(self, raw):
        return raw

    def input_transform(self, raw, transpose=False):
        return dict(state=torch.as_tensor(raw['states']))


class TinyEnv:
    num_envs = 2

    def reset(self, reset_state_ids):
        self._ids = np.asarray(reset_state_ids)
        self._step = 0
        return self._obs(), {}

    def _obs(self):
        return dict(main_images=np.zeros((2, 2, 2, 3), dtype=np.uint8),
                    wrist_images=np.ones((2, 2, 2, 3), dtype=np.uint8),
                    states=np.stack([self._ids / 10, np.full(2, self._step / 10)], axis=1).astype(np.float32),
                    task_descriptions=['move'] * 2)

    def chunk_step(self, actions):
        self._step += 1
        success = ((self._ids % 2 == 0) & (self._step >= 5))[:, None]
        truncs = np.full((2, 1), self._step >= 20)
        return self._obs(), success.astype(np.float32) - 1, success, truncs, {}

    def close(self):
        pass


@pytest.fixture
def cfg():
    torch.set_num_threads(1)
    directory = Path(__file__).resolve().parents[1] / 'examples/configs'
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        cfg = compose(config_name='libero_10_qrflow_pi05')
    cfg.algorithm.episodes_per_iteration = 10
    cfg.algorithm.episodes_per_task = 1
    cfg.algorithm.critic_hidden_dims = [8]
    cfg.algorithm.critic_updates = 2
    cfg.algorithm.critic_global_batch_size = 4
    cfg.algorithm.actor_updates = 2
    cfg.algorithm.actor_global_batch_size = 4
    cfg.algorithm.actor_grad_micro_batch_size = 1
    cfg.algorithm.qrflow_num_paths = 3
    cfg.training.compile_critic = False
    cfg.runner.save_interval = 1
    cfg.env.train.max_episode_steps = 20
    return cfg


def _load_actor(cfg, device):
    model, state = TinyModel(), None
    if cfg.runner.actor_checkpoint:
        state = torch.load(cfg.runner.actor_checkpoint, weights_only=False)
        assert state['completed'] and state['iteration'] == cfg.runner.iteration - 1
        model.load_actor_state_dict(state['actor_state_dict'])
    return model.to(device).requires_grad_(False), state


def test_terminal_timeout_windows_and_max8_lcb():
    success = [False] * 4 + [True] + [False] * 15
    windows = n_step_windows(success, [False] * 19 + [True])
    np.testing.assert_array_equal(windows['starts'], np.arange(5))
    np.testing.assert_array_equal(windows['next_indices'], 5)
    assert windows['terminations'].all() and not windows['truncations'].any()
    expected = np.array([-sum(0.999 ** i for i in range(5 - start)) for start in range(5)])
    np.testing.assert_allclose(windows['rewards'], expected, rtol=1e-6)
    timeout = n_step_windows([False] * 20, [False] * 19 + [True])
    assert timeout['truncations'][-1] and not timeout['terminations'].any()
    q = torch.zeros(2, 8, 10)
    q[:, 0] = -1
    q[:, 1:] = -5
    targets = reference_td_targets(torch.tensor([-3., -2.]), torch.tensor([0.9, 0.9]),
                                   torch.tensor([False, True]), q)
    torch.testing.assert_close(targets[0], torch.full((10,), -3.9))
    torch.testing.assert_close(targets[1], torch.full((10,), -2.))
    clipped = reference_td_targets(torch.tensor([-2000., 20.]), torch.ones(2), torch.ones(2, dtype=torch.bool), q)
    assert clipped[0, 0] == -1000 and clipped[1, 0] == 0


def test_single_sde_matches_ppo_coefficients_and_one_selected_node():
    indices = torch.arange(5)
    times, sigma, scale, steps, shrink = chain_coefficients(indices)
    native = torch.linspace(1, 0.2, 5)
    adjusted = native.clone()
    adjusted[0] = native[1]
    expected_sigma = 0.5 * torch.sqrt(native / (1 - adjusted))
    torch.testing.assert_close(sigma, expected_sigma)
    torch.testing.assert_close(steps, native - torch.cat([native[1:], native.new_zeros(1)]))
    initial = torch.ones(5, 2, 1, 1)
    innovation = torch.full_like(initial, 0.7)
    nodes = torch.arange(5)
    velocity = lambda x, t, ref: torch.full_like(x, 0.1)
    actual = sample_single_sde(velocity, initial, innovation, nodes)
    expected = initial.clone()
    for node in range(5):
        mean = (1 - steps[node] * shrink[node]) * expected + steps[node] * scale[node] * 0.1
        gaussian = mean + steps[node].sqrt() * sigma[node] * innovation
        expected = torch.where((nodes == node)[:, None, None, None], gaussian, expected + 0.02)
    torch.testing.assert_close(actual, expected)


def test_three_stages_two_rounds_and_optimizer_continuity(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(collect_qrflow, 'load_actor', _load_actor)
    monkeypatch.setattr(train_qrflow_actor, 'load_actor', _load_actor)
    monkeypatch.setattr(collect_qrflow, 'RemoteEnv', lambda *args: TinyEnv())
    first_q = first_actor = first_replay = None
    for iteration in (1, 2):
        cfg.runner.iteration = iteration
        collection = tmp_path / str(iteration) / 'collection'
        collect_qrflow.collect(cfg, collection, device='cpu')
        replay = QRFlowReplay(collection / 'replay.json')
        assert len(replay.metadata['episodes']) == iteration * 10
        assert replay.metadata['collection'] == dict(episodes=10, successes=5, success_rate=0.5)
        assert all(Path(row['reference_actions']).parent == collection / 'cache' for row in replay.metadata['episodes'])
        cfg.runner.replay_path = str(collection / 'replay.json')
        critic_dir = tmp_path / str(iteration) / 'critic'
        train_qrflow_critic.train(cfg, critic_dir, 'cpu')
        q = torch.load(critic_dir / 'checkpoint.pt', weights_only=False)
        assert q['completed'] and q['total_steps'] == iteration * 2
        assert {int(value['step']) for value in q['optimizer']['state'].values()} == {iteration * 2}
        cfg.runner.critic_checkpoint = str(critic_dir / 'checkpoint.pt')
        actor_dir = tmp_path / str(iteration) / 'actor'
        train_qrflow_actor.train(cfg, actor_dir, device='cpu')
        actor = torch.load(actor_dir / 'checkpoint.pt', weights_only=False)
        assert actor['completed'] and actor['total_steps'] == iteration * 2
        assert {int(value['step']) for value in actor['optimizer']['state'].values()} == {2}
        reference = first_actor['actor_state_dict'] if first_actor else TinyModel().reference_state_dict()
        for key, value in reference.items():
            torch.testing.assert_close(actor['reference_state_dict'][key], value, rtol=0, atol=0)
        assert any(not torch.equal(actor['actor_state_dict'][key], value) for key, value in reference.items())
        if iteration == 2:
            assert replay.metadata['episodes'][0]['path'] == first_replay['episodes'][0]['path']
            assert replay.metadata['episodes'][0]['reference_actions'] != first_replay['episodes'][0]['reference_actions']
            assert replay.metadata['reference'] == str(Path(cfg.runner.actor_checkpoint).resolve())
        first_q, first_actor, first_replay = q, actor, replay.metadata
        cfg.runner.actor_checkpoint = str(actor_dir / 'checkpoint.pt')
        cfg.runner.previous_replay = cfg.runner.replay_path
        replay.close()


def test_global_normalization_and_microbatch_invariance(cfg):
    mean, scale = return_statistics(return_moments(torch.tensor([-4., 0., 4., 8.])))
    assert mean == 2
    torch.testing.assert_close(scale, torch.tensor(20.).sqrt())
    model = TinyModel()
    critic = ActionInjectedEnsembleQ(feature_dim=2, action_dim=70, hidden_dims=[8])
    frozen = copy.deepcopy(critic.state_dict())
    batch = {'obs/state': np.array([[0., 0.], [1., 0.], [2., 1.], [3., 1.]], dtype=np.float32),
             'reference_actions': np.random.RandomState(2).randn(4, 8, 10, 7).astype(np.float32)}
    policy = QRFlowPolicy(cfg, model, critic, 0)
    other_model = TinyModel()
    other_cfg = copy.deepcopy(cfg)
    other_cfg.algorithm.actor_grad_micro_batch_size = 2
    other_policy = QRFlowPolicy(other_cfg, other_model, copy.deepcopy(critic), 0)
    policy.run_update(batch)
    other_policy.run_update(batch)
    for key, value in model.fast_actor.state_dict().items():
        torch.testing.assert_close(value, other_model.fast_actor.state_dict()[key], rtol=1e-5, atol=1e-7)
    for key, value in frozen.items():
        torch.testing.assert_close(critic.state_dict()[key], value, rtol=0, atol=0)


def test_exact_resume_and_stale_stage_rejection(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(collect_qrflow, 'load_actor', _load_actor)
    monkeypatch.setattr(train_qrflow_actor, 'load_actor', _load_actor)
    monkeypatch.setattr(collect_qrflow, 'RemoteEnv', lambda *args: TinyEnv())
    collection = tmp_path / 'collection'
    collect_qrflow.collect(cfg, collection, device='cpu')
    cfg.runner.replay_path = str(collection / 'replay.json')
    train_qrflow_critic.train(cfg, tmp_path / 'critic-full', 'cpu')
    expected_q = torch.load(tmp_path / 'critic-full/checkpoint.pt', weights_only=False)

    original_save = train_qrflow_critic.save_checkpoint
    def interrupt_critic(path, state):
        original_save(path, state)
        if state['step'] == 1:
            raise InterruptedError
    monkeypatch.setattr(train_qrflow_critic, 'save_checkpoint', interrupt_critic)
    with pytest.raises(InterruptedError):
        train_qrflow_critic.train(cfg, tmp_path / 'critic-part', 'cpu')
    partial_q = tmp_path / 'critic-part/checkpoint.pt'
    cfg.runner.critic_checkpoint = str(partial_q)
    with pytest.raises(AssertionError, match='completed critic'):
        train_qrflow_actor.train(cfg, tmp_path / 'bad-actor', device='cpu')
    monkeypatch.setattr(train_qrflow_critic, 'save_checkpoint', original_save)
    cfg.runner.critic_checkpoint = None
    cfg.runner.resume_from = str(partial_q)
    train_qrflow_critic.train(cfg, tmp_path / 'critic-resume', 'cpu')
    actual_q = torch.load(tmp_path / 'critic-resume/checkpoint.pt', weights_only=False)
    for key, value in expected_q['q_state_dict'].items():
        torch.testing.assert_close(actual_q['q_state_dict'][key], value, rtol=0, atol=0)
    for key, value in expected_q['target_q_state_dict'].items():
        torch.testing.assert_close(actual_q['target_q_state_dict'][key], value, rtol=0, atol=0)

    cfg.runner.resume_from = None
    cfg.runner.critic_checkpoint = str(tmp_path / 'critic-full/checkpoint.pt')
    train_qrflow_actor.train(cfg, tmp_path / 'actor-full', device='cpu')
    expected_actor = torch.load(tmp_path / 'actor-full/checkpoint.pt', weights_only=False)
    original_save = train_qrflow_actor.save_checkpoint
    def interrupt_actor(path, state):
        original_save(path, state)
        if state['step'] == 1:
            raise InterruptedError
    monkeypatch.setattr(train_qrflow_actor, 'save_checkpoint', interrupt_actor)
    with pytest.raises(InterruptedError):
        train_qrflow_actor.train(cfg, tmp_path / 'actor-part', device='cpu')
    monkeypatch.setattr(train_qrflow_actor, 'save_checkpoint', original_save)
    cfg.runner.resume_from = str(tmp_path / 'actor-part/checkpoint.pt')
    train_qrflow_actor.train(cfg, tmp_path / 'actor-resume', device='cpu')
    actual_actor = torch.load(tmp_path / 'actor-resume/checkpoint.pt', weights_only=False)
    for key, value in expected_actor['actor_state_dict'].items():
        torch.testing.assert_close(actual_actor['actor_state_dict'][key], value, rtol=0, atol=0)
    assert actual_actor['step'] == expected_actor['step'] == 2
    cfg.algorithm.beta = 0.5
    with pytest.raises(AssertionError, match='same QRFlow objective'):
        train_qrflow_actor.train(cfg, tmp_path / 'bad-resume', device='cpu')


@pytest.mark.skipif(shutil.which('setsid') is None, reason='Launcher targets Linux cluster nodes')
@pytest.mark.parametrize('fail_critic', [False, True])
def test_shell_stage_order_inputs_and_environment_cleanup(tmp_path, fail_critic):
    stub = tmp_path / 'python-stub'
    stub.write_text('#!' + sys.executable + '\n' + '''
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
root = Path(os.environ['STUB_ROOT'])
if 'env_clients.client' in args:
    (root / ('env-' + str(os.getpid()))).write_text(str(os.getpid()))
    while True:
        time.sleep(1)
elif '--cfg' in args:
    print('env: {}')
else:
    stage = next(arg for arg in args if arg.startswith('examples/'))
    with (root / 'calls.jsonl').open('a') as f:
        f.write(json.dumps(args) + '\\n')
    directory = Path(next(arg.split('=', 1)[1] for arg in args if arg.startswith('hydra.run.dir=')))
    directory.mkdir(parents=True)
    if 'collect_qrflow' in stage:
        time.sleep(0.2)
        (directory / 'replay.json').write_text('{}')
    else:
        for file in root.glob('env-*'):
            try:
                os.kill(int(file.read_text()), 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('Environment server survived collection cleanup')
        replay = next(arg.split('=', 1)[1] for arg in args if arg.startswith('runner.replay_path='))
        assert Path(replay).exists()
        if 'critic' in stage and os.environ['FAIL_CRITIC'] == '1':
            sys.exit(7)
        (directory / 'checkpoint.pt').write_text('checkpoint')
''')
    stub.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, MODEL_PYTHON=str(stub), ENV_PYTHON=str(stub), STUB_ROOT=str(tmp_path),
               FAIL_CRITIC='1' if fail_critic else '0', OUT_DIR=str(tmp_path / 'output'), NPROC='2', ITERATIONS='2')
    result = subprocess.run(['bash', str(root / 'examples/run_iterative_rl.sh')], env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == (7 if fail_critic else 0), result.stderr
    calls = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    stages = [next(arg for arg in call if arg.startswith('examples/')) for call in calls]
    expected = ['examples/collect_qrflow.py', 'examples/train_qrflow_critic.py', 'examples/train_qrflow_actor.py']
    assert stages == (expected[:2] if fail_critic else expected * 2)
    if not fail_critic:
        assert f'runner.previous_replay={tmp_path}/output/iteration01/collection/replay.json' in calls[3]
        assert f'runner.critic_checkpoint={tmp_path}/output/iteration01/critic/checkpoint.pt' in calls[4]
        assert f'runner.actor_checkpoint={tmp_path}/output/iteration01/actor/checkpoint.pt' in calls[5]
