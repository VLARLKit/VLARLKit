import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import (
    PI0Pytorch, create_sinusoidal_pos_embedding, make_att_2d_masks,
)
from vlarlkit.models.modules.flow_sde import sample_single_sde
from vlarlkit_openpi.openpi_for_rl import OpenPi0ForRL


def qrflow_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    # The installed HF SDPA implementation repeats KV over query heads and
    # calls contiguous(), retaining 8 copies in Pi0.5 backward. Fold grouped
    # query heads into the query-token axis instead: exactly the same attention,
    # with one KV head and ordinary masked SDPA kernels (no GQA-only kernel).
    b, heads, length, dim = query.shape
    kv_heads = key.shape[1]
    groups = heads // kv_heads
    query = query.reshape(b, kv_heads, groups * length, dim)
    if attention_mask is not None and attention_mask.shape[-2] != 1:
        attention_mask = attention_mask.repeat(1, 1, groups, 1)
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.to(query.dtype)
    output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout,
        scale=scaling, is_causal=False,
    )
    return output.reshape(b, heads, length, dim).transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS.register("qrflow_sdpa", qrflow_attention)


@dataclass(frozen=True)
class OpenPi0QRFlowConfig(Pi0Config):
    config_name: str = "pi05_libero"
    num_images_in_input: int = 2
    action_chunk: int = 10
    action_env_dim: int = 7
    num_steps: int = 5
    train_expert_only: bool = True
    critic_state_dim: int = 8


@dataclass
class PrefixContext:
    past_key_values: DynamicCache
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    features: torch.Tensor

    def select(self, state_indices):
        """Expand an immutable prefix, never re-encode images or text."""
        def select_tensor(value):
            if value.shape[0] == 1:
                return value.expand(len(state_indices), *value.shape[1:])
            return value.index_select(0, state_indices)

        cache = DynamicCache.from_legacy_cache(tuple(
            (select_tensor(key), select_tensor(value))
            for key, value in self.past_key_values
        ))
        return PrefixContext(cache, select_tensor(self.attention_mask),
                             select_tensor(self.position_ids), select_tensor(self.features))


class ActionExpert(nn.Module):
    """Only the Pi0.5 decoder and action/time projections; no VLM or LM head."""

    _no_split_modules = ["GemmaDecoderLayer", "GemmaRMSNorm", "GemmaRotaryEmbedding"]

    def __init__(self, decoder, action_in, action_out, time_in, time_out):
        super().__init__()
        self._decoder = decoder
        self._action_in = action_in
        self._action_out = action_out
        self._time_in = time_in
        self._time_out = time_out
        self._width = action_in.out_features
        self._gradient_checkpointing = False
        self._decoder.config._attn_implementation = "qrflow_sdpa"

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def _decode(self, action_emb, time_emb, context):
        return self._decoder(
            inputs_embeds=action_emb,
            attention_mask=context.attention_mask,
            position_ids=context.position_ids,
            past_key_values=context.past_key_values,
            use_cache=False,
            adarms_cond=time_emb,
        ).last_hidden_state

    def forward(self, positions, times, context):
        # Pi0 learns noise-action, with time decreasing from 1 to 0.
        # QRFlow learns action-noise, with time increasing from 0 to 1.
        native_time = 1.0 - times.float()
        time_emb = create_sinusoidal_pos_embedding(
            native_time, self._width, min_period=4e-3, max_period=4.0,
            device=positions.device,
        ).float()
        time_emb = F.silu(self._time_out(F.silu(self._time_in(time_emb))))
        action_emb = self._action_in(positions.float())
        if self._gradient_checkpointing and torch.is_grad_enabled():
            hidden = checkpoint(self._decode, action_emb, time_emb, context,
                                use_reentrant=False, preserve_rng_state=False)
        else:
            hidden = self._decode(action_emb, time_emb, context)
        return -self._action_out(hidden.float())


class OpenPi0ForQRFlow(OpenPi0ForRL):
    config: OpenPi0QRFlowConfig

    def __init__(self, config):
        assert config.pi05 and config.train_expert_only
        assert config.action_dim == 32 and 0 < config.action_chunk <= config.action_horizon
        sample_actions = self.sample_actions
        PI0Pytorch.__init__(self, config)
        self.sample_actions = sample_actions

    def initialize_qrflow(self):
        """Called once by the loader AFTER loading the initial SFT weights."""
        assert not hasattr(self, "fast_actor")
        self.fast_actor = ActionExpert(
            self.paligemma_with_expert.gemma_expert.model,
            self.action_in_proj, self.action_out_proj, self.time_mlp_in, self.time_mlp_out,
        )
        del self.paligemma_with_expert.gemma_expert
        del self.action_in_proj, self.action_out_proj, self.time_mlp_in, self.time_mlp_out
        self._reference_actor = copy.deepcopy(self.fast_actor).requires_grad_(False)
        self.freeze_vlm()
        self.eval()

    def load_actor_state_dict(self, state, update_reference=True):
        self.fast_actor.load_state_dict(state)
        if update_reference:
            self._reference_actor.load_state_dict(state)

    def reference_state_dict(self):
        return {key: value.detach().cpu() for key, value in self._reference_actor.state_dict().items()}

    def load_reference_state_dict(self, state):
        self._reference_actor.load_state_dict(state)

    @torch.no_grad()
    def reference_actions(self, context, candidates=8):
        batch = len(context.features)
        expanded = context.select(torch.arange(batch, device=context.features.device).repeat_interleave(candidates))
        value = torch.randn(batch * candidates, self.config.action_horizon, self.config.action_dim,
                            device=context.features.device)
        for node in range(self.config.num_steps):
            times = torch.full((batch * candidates,), node / self.config.num_steps, device=value.device)
            value = value + self.velocity(value, times, expanded, reference=True).float() / self.config.num_steps
        return self.critic_actions(value).reshape(batch, candidates, self.config.action_chunk, self.config.action_env_dim)

    def normalize_actions(self, env_obs, actions):
        raw = self.obs_processor(env_obs)
        raw["actions"] = actions
        return self.critic_actions(self.input_transform(raw, transpose=False)["actions"])

    def freeze_vlm(self):
        self.paligemma_with_expert.paligemma.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.paligemma_with_expert.paligemma.eval()
        if hasattr(self, "fast_actor"):
            # Keep path sampling and gradient recomputation identical (no dropout).
            self.fast_actor.eval()
            self._reference_actor.eval()
        return self

    @torch.no_grad()
    def encode_prefix(self, observation):
        images, masks, tokens, token_masks, state = self._preprocess_observation(observation, train=False)
        embeddings, padding, attention = self.embed_prefix(images, masks, tokens, token_masks)
        mask = self._prepare_attention_masks_4d(make_att_2d_masks(padding, attention))
        # SDPA additive masks must match the Gemma attention projection dtype.
        mask = mask.to(self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype)
        positions = padding.cumsum(1) - 1
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "sdpa"
        output = self.paligemma_with_expert.paligemma.language_model(
            attention_mask=mask, position_ids=positions, inputs_embeds=embeddings,
            use_cache=True,
        )
        hidden, cache = output.last_hidden_state, output.past_key_values
        pooled = (hidden.float() * padding[..., None]).sum(1) / padding.sum(1, keepdim=True).clamp_min(1)
        features = torch.cat([pooled, state[:, :self.config.critic_state_dim].float()], dim=-1)
        b = padding.shape[0]
        horizon = self.config.action_horizon
        suffix_mask = torch.ones(b, 1, horizon, dtype=torch.bool, device=padding.device)
        mask = torch.cat([padding[:, None, :], suffix_mask], dim=-1)
        mask = self._prepare_attention_masks_4d(mask)
        positions = padding.sum(-1, keepdim=True) + torch.arange(horizon, device=padding.device)[None]
        return PrefixContext(cache, mask, positions, features)

    def velocity(self, positions, times, context, reference=False):
        actor = self._reference_actor if reference else self.fast_actor
        return actor(positions, times, context)

    def critic_actions(self, actions):
        return actions[..., :self.config.action_chunk, :self.config.action_env_dim]

    def set_action_bounds(self, lower, upper):
        self.register_buffer('_action_lower', torch.as_tensor(lower, dtype=torch.float32))
        self.register_buffer('_action_upper', torch.as_tensor(upper, dtype=torch.float32))

    def canonical_actions(self, actions):
        actions = self.critic_actions(actions).float()
        # These bounds represent physical [-1, 1], not the quantile interval.
        motion = actions[..., :6].clamp(min=self._action_lower, max=self._action_upper)
        return torch.cat((motion, actions[..., 6:7]), dim=-1)

    @torch.no_grad()
    def prefix_from_replay(self, obs):
        # Replay state is ALREADY normalized; do not run Normalize again.
        processed = {"image": {}, "image_mask": {}}
        for key, value in obs.items():
            if "/" in key:
                group, name = key.split("/", 1)
                processed[group][name] = value
            else:
                processed[key] = value
        for name in processed["image_mask"]:
            if name not in processed["image"]:
                processed["image"][name] = torch.zeros_like(next(iter(processed["image"].values())))
        return self.encode_prefix(_model.Observation.from_dict(processed))

    def prepare_observation(self, env_obs):
        processed = self.input_transform(self.obs_processor(env_obs), transpose=False)
        # Save uint8 images before Observation.from_dict converts its image dict.
        replay_obs = {}
        for key, value in processed.items():
            if isinstance(value, dict):
                for name, tensor in value.items():
                    # Reconstruct absent camera padding instead of storing a
                    # third all-zero image for every LIBERO transition.
                    if key == "image" and not processed["image_mask"][name].any():
                        continue
                    replay_obs[f"{key}/{name}"] = tensor
            else:
                replay_obs[key] = value
        processed = self.precision_processor(processed)
        return _model.Observation.from_dict(processed), replay_obs

    @torch.no_grad()
    def sample_actions(self, observation, noise=None, context=None, stochastic=False, **kwargs):
        if context is None:
            context = self.encode_prefix(observation)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = torch.randn(batch_size, self.config.action_horizon, self.config.action_dim,
                                device=observation.state.device)
        if stochastic:
            nodes = torch.randint(5, (batch_size,))
            innovation = torch.randn_like(noise)[:, None]
            def velocity(value, times, reference):
                return self.velocity(value[:, 0], times[:, 0], context, reference)[:, None]
            x = sample_single_sde(velocity, noise[:, None], innovation, nodes)[:, 0]
            return {"actions": x, "features": context.features}
        x = noise.float()
        for i in range(self.config.num_steps):
            time = torch.full((batch_size,), i / self.config.num_steps, device=x.device)
            x = x + self.velocity(x, time, context).float() / self.config.num_steps
        return {"actions": x, "features": context.features}

    @torch.no_grad()
    def predict_action_batch(self, env_obs, mode="train", **kwargs):
        observation, replay_obs = self.prepare_observation(env_obs)
        outputs = self.sample_actions(observation)
        actions = self.output_transform({"actions": outputs["actions"], "state": observation.state})["actions"]
        forward_inputs = {f"obs/{key}": value for key, value in replay_obs.items()}
        forward_inputs.update(features=outputs["features"],
                              normalized_actions=self.critic_actions(outputs["actions"]))
        return actions.numpy(), {"forward_inputs": forward_inputs,
                                 "prev_logprobs": None, "prev_values": None}

    @torch.no_grad()
    def encode_next_observation(self, env_obs):
        observation, _ = self.prepare_observation(env_obs)
        return {"features": self.encode_prefix(observation).features}
