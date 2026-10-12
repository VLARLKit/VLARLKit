import math
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullOptimStateDictConfig, FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType,
)

from vlarlkit.models.modules.flow_sde import noise_plan, node_plan, prepare_actor_targets, qrflow_loss
from vlarlkit.utils.fsdp_utils import wrap_model_with_fsdp


def return_moments(values):
    values = values.detach().double()
    return torch.stack((values.sum(), values.square().sum(), values.new_tensor(values.numel())))


def return_statistics(moments, eps=1e-8):
    mean = moments[0] / moments[2]
    scale = (moments[1] / moments[2] - mean.square()).clamp_min(0).sqrt()
    return mean.float(), scale.float().clamp_min(eps)


class QRFlowPolicy:
    """Update only the actor against a frozen reference actor and TD critic."""

    def __init__(self, cfg, model, critic, rank):
        self._cfg = cfg
        self._rank = rank
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._device = next(model.parameters()).device
        self._model = model.requires_grad_(False).eval()
        self._critic = critic.to(self._device).requires_grad_(False).eval()
        self._num_paths = int(cfg.algorithm.qrflow_num_paths)
        self._beta = float(cfg.algorithm.beta)
        self._micro_batch_size = int(cfg.algorithm.actor_grad_micro_batch_size)
        self._cost_clip = float(cfg.algorithm.qrflow_cost_clip)
        self._node_seed = int(cfg.algorithm.single_sde_node_seed) + int(cfg.runner.iteration) * 100000
        self._noise_seed = int(cfg.algorithm.single_sde_noise_seed) + int(cfg.runner.iteration) * 100000
        self._update_step = 0
        self._signature = dict(beta=self._beta, paths=self._num_paths, microbatch=self._micro_batch_size,
                               global_batch=int(cfg.algorithm.actor_global_batch_size), cost_clip=self._cost_clip,
                               node_seed=self._node_seed, noise_seed=self._noise_seed,
                               lr=float(cfg.training.optim.lr), grad_clip=float(cfg.training.optim.clip_grad))
        assert model.config.num_steps == 5 and self._num_paths > 1
        assert math.isfinite(self._beta) and self._beta > 0
        assert self._micro_batch_size > 0 and self.replay_batch_size() % self._micro_batch_size == 0
        assert int(cfg.algorithm.actor_global_batch_size) % self._world_size == 0

        model.fast_actor.requires_grad_(True)
        fsdp_cfg = cfg.training.get('fsdp_config', {})
        if dist.is_initialized() and self._device.type == 'cuda':
            model.fast_actor = wrap_model_with_fsdp(model.fast_actor, fsdp_cfg, rank)
        elif fsdp_cfg.get('gradient_checkpointing', False):
            model.fast_actor.gradient_checkpointing_enable()
        self._optimizer = torch.optim.Adam(model.fast_actor.parameters(), lr=float(cfg.training.optim.lr))

    @property
    def update_step(self):
        return self._update_step

    def replay_batch_size(self):
        return int(self._cfg.algorithm.actor_global_batch_size) // self._world_size

    @torch.no_grad()
    def _actor_targets(self, context, references, nodes, initial, innovation):
        batch = len(context.features)
        paths = self._num_paths
        expanded = context.select(torch.arange(batch, device=self._device).repeat_interleave(paths))
        reference_features = context.features.repeat_interleave(references.shape[1], dim=0)
        reference_q = self._critic(reference_features, references.flatten(0, 1))
        reference_q = reference_q.reshape(batch, references.shape[1], -1).mean((1, 2))

        def velocity(positions, times, reference):
            value = self._model.velocity(positions.flatten(0, 1), times.flatten(), expanded, reference)
            return value.reshape_as(positions)

        def utility(actions):
            actions = self._model.canonical_actions(actions).flatten(0, 1)
            values = self._critic(expanded.features, actions).mean(-1).reshape(batch, paths)
            return (values - reference_q[:, None]).clamp(-10, 10)

        return prepare_actor_targets(velocity, utility, initial, innovation, nodes, self._beta)

    def _context(self, batch, start, stop):
        obs = {key[4:]: torch.as_tensor(value[start:stop], device=self._device)
               for key, value in batch.items() if key.startswith('obs/')}
        return self._model.prefix_from_replay(obs)

    def run_update(self, batch):
        local_batch = len(batch['reference_actions'])
        assert local_batch == self.replay_batch_size()
        size = self._micro_batch_size
        accumulations = local_batch // size
        actor = self._model.fast_actor
        self._model.eval()
        self._optimizer.zero_grad(set_to_none=True)
        nodes, _ = node_plan(self._node_seed, self._update_step, self._rank,
                             self._world_size, local_batch, 5)
        initial, innovation, _ = noise_plan(
            self._noise_seed, self._update_step, self._rank, self._world_size,
            local_batch, self._num_paths, self._model.config.action_horizon,
            self._model.config.action_dim, self._device,
        )
        moments = torch.zeros(3, dtype=torch.float64, device=self._device)
        cached, metrics = [], {}
        # Sample the complete batch before updating; normalization spans every task and rank.
        for start in range(0, local_batch, size):
            context = self._context(batch, start, start + size)
            references = torch.as_tensor(batch['reference_actions'][start:start + size], device=self._device)
            targets, info = self._actor_targets(context, references, nodes[start:start + size],
                                                 initial[start:start + size], innovation[start:start + size])
            moments += return_moments(targets['cost_to_go'])
            cached.append({key: value.cpu() for key, value in targets.items()})
            for key, value in info.items():
                metrics[key] = metrics.get(key, 0.0) + value / accumulations
            del context, targets
        if dist.is_initialized():
            dist.all_reduce(moments)
        if not bool(torch.isfinite(moments).all()):
            raise FloatingPointError('Nonfinite QRFlow returns')
        mean, scale = return_statistics(moments)

        for index, start in enumerate(range(0, local_batch, size)):
            targets = {key: value.to(self._device) for key, value in cached[index].items()}
            targets['cost_to_go'] = ((targets['cost_to_go'] - mean) / scale).clamp(-self._cost_clip, self._cost_clip)
            context = self._context(batch, start, start + size)
            context = context.select(torch.arange(size, device=self._device).repeat_interleave(self._num_paths))
            sync = actor.no_sync() if isinstance(actor, FSDP) and start + size < local_batch else nullcontext()
            with sync:
                velocity = self._model.velocity(targets['positions'].flatten(0, 2), targets['times'].flatten(), context)
                loss = qrflow_loss(velocity.reshape_as(targets['positions']), targets, 5)
                (loss / accumulations).backward()
            metrics['actor_loss'] = metrics.get('actor_loss', 0.0) + float(loss.detach()) / accumulations
            del context, targets, velocity, loss

        if isinstance(actor, FSDP):
            norm = actor.clip_grad_norm_(float(self._cfg.training.optim.clip_grad))
        else:
            if dist.is_initialized():
                for param in actor.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad)
                        param.grad.div_(self._world_size)
            norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), float(self._cfg.training.optim.clip_grad))
        finite = torch.isfinite(torch.as_tensor(norm, device=self._device)).int()
        if dist.is_initialized():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise FloatingPointError('Nonfinite QRFlow gradient; optimizer was not updated')
        self._optimizer.step()
        self._update_step += 1
        metrics.update(actor_grad_norm=float(norm), advantage_scale=float(scale))
        return metrics

    def state_dict(self):
        """All ranks participate when the actor is wrapped with FSDP."""
        actor = self._model.fast_actor
        if isinstance(actor, FSDP):
            with FSDP.state_dict_type(actor, StateDictType.FULL_STATE_DICT,
                                      FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                                      FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                weights = actor.state_dict()
                optimizer = FSDP.optim_state_dict(actor, self._optimizer)
        else:
            weights = {key: value.detach().cpu() for key, value in actor.state_dict().items()}
            optimizer = self._optimizer.state_dict()
        return dict(actor_state_dict=weights, reference_state_dict=self._model.reference_state_dict(),
                    optimizer=optimizer, step=self._update_step, policy_config=self._signature)

    def load_state_dict(self, state):
        assert state['policy_config'] == self._signature, 'Resume requires the same QRFlow objective and batching'
        actor = self._model.fast_actor
        if isinstance(actor, FSDP):
            with FSDP.state_dict_type(actor, StateDictType.FULL_STATE_DICT):
                actor.load_state_dict(state['actor_state_dict'])
            optimizer = FSDP.optim_state_dict_to_load(actor, self._optimizer, state['optimizer'])
        else:
            actor.load_state_dict(state['actor_state_dict'])
            optimizer = state['optimizer']
        self._model.load_reference_state_dict(state['reference_state_dict'])
        self._optimizer.load_state_dict(optimizer)
        self._update_step = int(state['step'])

    def get_model(self):
        return self._model
