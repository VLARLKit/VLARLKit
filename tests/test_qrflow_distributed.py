"""Run with torchrun on a GPU node to exercise the actual FSDP checkpoint path."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn

from openpi import transforms
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.shared import normalize
from vlarlkit.models.modules.qrflow_critic import ActionInjectedEnsembleQ
from vlarlkit.policies.qrflow import QRFlowPolicy
from vlarlkit.utils.qrflow import save_checkpoint, save_json
from vlarlkit_openpi.dataconfigs.libero_dataconfig import LiberoInputs, LiberoOutputs
from vlarlkit_openpi.openpi_for_qrflow import OpenPi0ForQRFlow, OpenPi0QRFlowConfig

from test_qrflow_openpi import TinyPair, Tokenize, _obs


def main(output):
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group('nccl')
    world_size = dist.get_world_size()
    torch.manual_seed(7)

    def initialize(model, config):
        nn.Module.__init__(model)
        model.config, model.pi05 = config, True
        model.gradient_checkpointing_enabled = False
        model.paligemma_with_expert = TinyPair()
        model.action_in_proj, model.action_out_proj = nn.Linear(32, 16), nn.Linear(16, 32)
        model.time_mlp_in, model.time_mlp_out = nn.Linear(16, 16), nn.Linear(16, 16)

    original_init = PI0Pytorch.__init__
    PI0Pytorch.__init__ = initialize
    try:
        config = OpenPi0QRFlowConfig(pi05=True, action_horizon=10, action_chunk=10,
                                     num_steps=5, discrete_state_input=False)
        model = OpenPi0ForQRFlow(config)
    finally:
        PI0Pytorch.__init__ = original_init
    model.initialize_qrflow()
    stats = normalize.load('/network/scratch/s/sunyi/rlinf_sft_models/RLinf-Pi05-LIBERO-SFT/physical-intelligence/libero')
    normalizer = transforms.Normalize(stats, use_quantiles=True)
    bounds = normalizer({'actions': np.stack([-np.ones(7), np.ones(7)])})['actions']
    model.set_action_bounds(bounds[0, :6], bounds[1, :6])
    model.setup_wrappers(
        transforms=[LiberoInputs(model_type=config.model_type), normalizer, Tokenize(), transforms.PadStatesAndActions(32)],
        output_transforms=[transforms.Unnormalize(stats, use_quantiles=True), LiberoOutputs()],
    )
    model.to(rank).eval()
    raw = _obs()
    raw['states'] += rank * 0.01
    _, info = model.predict_action_batch(raw)
    batch = {key: value.cpu().numpy() for key, value in info['forward_inputs'].items() if key.startswith('obs/')}
    observation, _ = model.prepare_observation(raw)
    context = model.encode_prefix(observation)
    batch['reference_actions'] = model.reference_actions(context).cpu().numpy()
    critic = ActionInjectedEnsembleQ(feature_dim=24, action_dim=70, hidden_dims=[16]).to(rank)
    frozen = {key: value.detach().cpu().clone() for key, value in model.named_parameters()
              if not key.startswith('fast_actor.')}
    critic_before = {key: value.clone() for key, value in critic.state_dict().items()}
    cfg = OmegaConf.create(dict(
        algorithm=dict(actor_global_batch_size=world_size * 2, actor_grad_micro_batch_size=1,
                       qrflow_num_paths=32, beta=0.3, qrflow_cost_clip=10,
                       single_sde_node_seed=11, single_sde_noise_seed=12),
        training=dict(optim=dict(lr=1e-3, clip_grad=1), fsdp_config=dict(sharding_strategy='no_shard')),
        runner=dict(iteration=1),
    ))
    policy = QRFlowPolicy(cfg, model, critic, rank)
    first = policy.run_update(batch)
    scales = [None] * world_size
    dist.all_gather_object(scales, first['advantage_scale'])
    assert len(set(scales)) == 1 and scales[0] > 0
    state = policy.state_dict()
    if rank == 0:
        save_checkpoint(output / 'step1.pt', state)
    dist.barrier()
    policy.run_update(batch)
    expected = policy.state_dict()
    state = torch.load(output / 'step1.pt', map_location='cpu', weights_only=False)
    policy.load_state_dict(state)
    policy.run_update(batch)
    actual = policy.state_dict()
    if rank == 0:
        for key, value in expected['actor_state_dict'].items():
            torch.testing.assert_close(actual['actor_state_dict'][key], value, rtol=0, atol=0)
        assert {int(item['step']) for item in actual['optimizer']['state'].values()} == {2}
        save_json(output / 'passed.json', dict(result='passed', world_size=world_size,
                  fsdp='no_shard', updates=2, exact_resume=True, global_advantage_scale=scales[0]))
    for key, value in model.named_parameters():
        if key in frozen:
            torch.testing.assert_close(value.cpu(), frozen[key], rtol=0, atol=0)
    for key, value in critic_before.items():
        torch.testing.assert_close(critic.state_dict()[key], value, rtol=0, atol=0)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
