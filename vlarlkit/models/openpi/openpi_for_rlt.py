from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

from vlarlkit.models.openpi.openpi_for_rl import OpenPi0ForRL


@dataclass(frozen=True)
class OpenPi0RLTConfig(Pi0Config):
    config_name: str = "pi0_libero"
    num_images_in_input: int = 2
    action_chunk: int = 5
    action_env_dim: int = 7
    num_steps: int = 10
    train_expert_only: bool = True
    # RLT-specific
    rlt_state_dim: int = 8
    rlt_hidden_dims: tuple = field(default_factory=lambda: (256, 256))
    rlt_num_q_heads: int = 2
    rlt_ref_action_dropout: float = 0.5
    rlt_action_magnitude: float = 1.0


class OpenPi0ForRLT(OpenPi0ForRL):

    config: OpenPi0RLTConfig

    @property
    def _no_split_modules(self) -> list[str]:
        if self.config.train_expert_only:
            return [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        return [
            "GemmaMLP",
            "SiglipVisionEmbeddings",
            "GemmaRMSNorm",
            "GemmaRotaryEmbedding",
        ]

    def __init__(self, config: OpenPi0RLTConfig):
        from vlarlkit.models.modules.compact_encoders import CompactMultiQHead
        from vlarlkit.models.modules.gaussian_policy import GaussianPolicy

        # Bypass OpenPi0ForRL.__init__, call PI0Pytorch directly
        sample_actions_func = self.sample_actions
        PI0Pytorch.__init__(self, config)
        self.sample_actions = sample_actions_func
        self.global_step = 0

        # RL token dimension: 2048 for pi05, 1024 for pi0
        if "pi05_" in config.config_name:
            self._rl_token_dim = 2048
        else:
            self._rl_token_dim = 1024

        action_dim = config.action_chunk * config.action_env_dim

        _rlt_dtype = torch.bfloat16
        mag = config.rlt_action_magnitude

        # Actor: maps (rl_token, state, ref_actions) -> refined actions
        self.rlt_actor = GaussianPolicy(
            input_dim=self._rl_token_dim + config.rlt_state_dim + action_dim,
            output_dim=action_dim,
            hidden_dims=config.rlt_hidden_dims,
            low=-mag if mag != 1.0 else None,
            high=mag if mag != 1.0 else None,
            action_horizon=1,
        ).to(dtype=_rlt_dtype)

        # Critic: double-Q on (rl_token, state) + action
        self.q_head = CompactMultiQHead(
            state_dim=self._rl_token_dim + config.rlt_state_dim,
            image_dim=0,
            action_dim=action_dim,
            hidden_dims=config.rlt_hidden_dims,
            num_q_heads=config.rlt_num_q_heads,
            output_dim=1,
        ).to(dtype=_rlt_dtype)

        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    def extract_rl_token(self, prefix_output):
        """Mean-pool VLM prefix output over valid tokens → [B, rl_token_dim].

        Reuses the masking logic from get_value_from_vlm.
        """
        if "pi05_" in self.config.config_name:
            lang_token_len = 200
        elif "pi0_" in self.config.config_name:
            lang_token_len = 48

        prefix_mask = (
            [True] * 256 * self.config.num_images_in_input
            + [False] * 256 * (3 - self.config.num_images_in_input)
            + [True] * lang_token_len
        )
        rl_token = prefix_output[:, prefix_mask, :]
        rl_token = rl_token.mean(dim=1, keepdim=False)
        rl_token = rl_token.to(dtype=torch.float32)
        return rl_token

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        observation = _model.Observation.from_dict(processed_obs)

        # VLM forward → prefix_output + ODE denoising → ref_actions
        outputs = self.sample_actions(observation)
        ref_actions_raw = self.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )["actions"]  # [B, C, d]

        rl_token = outputs["rl_token"]  # [B, rl_token_dim], float32

        # Flatten ref_actions: [B, C, d] -> [B, C*d]
        B = ref_actions_raw.shape[0]
        ref_actions_flat = ref_actions_raw.reshape(B, -1)

        # Actor refinement
        states = env_obs["states"]
        if not isinstance(states, torch.Tensor):
            states = torch.from_numpy(states)
        states = states.to(device=rl_token.device, dtype=torch.bfloat16)
        if states.dim() > 2:
            states = states.reshape(B, -1)

        rl_token_bf16 = rl_token.to(torch.bfloat16)
        ref_actions_bf16 = ref_actions_flat.to(device=rl_token.device, dtype=torch.bfloat16)
        actor_input = torch.cat([rl_token_bf16, states, ref_actions_bf16], dim=-1)

        deterministic = mode == "eval"
        refined_actions, _ = self.rlt_actor.sample(actor_input, deterministic=deterministic)
        # [B, 1, C*d] -> [B, C, d]
        refined_actions = refined_actions.squeeze(1).reshape(B, self.config.action_chunk, self.config.action_env_dim)
        actions = refined_actions.float().detach().cpu().numpy()

        forward_inputs = {
            "rl_token": rl_token.detach().cpu().numpy(),
            "ref_actions": ref_actions_flat.detach().cpu().numpy(),
        }
        result = {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": forward_inputs,
        }
        return actions, result

    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Pure ODE denoising — same as DSRL. Returns actions + rl_token."""
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps

        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Extract RL token from VLM output
        rl_token = self.extract_rl_token(prefix_output)

        # Pure Euler ODE steps
        timesteps = torch.linspace(1, 1 / num_steps, num_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        x_t = noise

        for idx in range(num_steps):
            t_input = timesteps[idx].expand(bsize)
            delta = timesteps[idx] - timesteps[idx + 1]
            suffix_out = self.get_suffix_out(
                state, prefix_pad_masks, past_key_values, x_t, t_input,
            )
            v_t = self.action_out_proj(suffix_out)
            x_t = x_t - v_t * delta

        return {"actions": x_t, "rl_token": rl_token}

    def forward(self, **kwargs):
        return self.actor_forward(**kwargs)

    def actor_forward(self, obs=None, train=False, **kwargs):
        """Actor forward: concat(rl_token, states, ref_actions) -> actions.

        During training, applies ref_action dropout (zero out ref_actions with probability p).
        """
        rl_token = obs["rl_token"]
        states = obs["states"]
        ref_actions = obs["ref_actions"]

        device = next(self.rlt_actor.parameters()).device
        rl_token = self._to_device(rl_token, device, torch.bfloat16)
        states = self._to_device(states, device, torch.bfloat16)
        ref_actions = self._to_device(ref_actions, device, torch.bfloat16)

        # Ref action dropout during training
        if train and self.config.rlt_ref_action_dropout > 0:
            B = ref_actions.shape[0]
            mask = (torch.rand(B, 1, device=device) > self.config.rlt_ref_action_dropout).float()
            ref_actions = ref_actions * mask

        features = torch.cat([rl_token, states, ref_actions], dim=-1)
        deterministic = not train
        actions, log_probs = self.rlt_actor.sample(features, deterministic=deterministic)
        return actions, log_probs, None

    def critic_forward(self, obs=None, actions=None, **kwargs):
        """Critic forward: Q(concat(rl_token, states), actions) -> [B, num_q_heads]."""
        rl_token = obs["rl_token"]
        states = obs["states"]

        device = next(self.q_head.parameters()).device
        rl_token = self._to_device(rl_token, device, torch.bfloat16)
        states = self._to_device(states, device, torch.bfloat16)
        actions = self._to_device(actions, device, torch.bfloat16)

        if actions.dim() == 3:
            actions = actions[:, 0, :]

        state_features = torch.cat([rl_token, states], dim=-1)
        image_features = torch.zeros(state_features.shape[0], 0, device=device, dtype=torch.bfloat16)
        q_values = self.q_head(state_features, image_features, actions)
        return q_values

    def freeze_vlm(self):
        super().freeze_vlm()
        # RLT: also freeze gemma_expert and projection layers (same as DSRL)
        if self.config.train_expert_only:
            self.paligemma_with_expert.gemma_expert.eval()
            for params in self.paligemma_with_expert.gemma_expert.parameters():
                params.requires_grad = False

            if self.pi05:
                projection_names = [
                    "action_in_proj", "action_out_proj",
                    "time_mlp_in", "time_mlp_out",
                ]
            else:
                projection_names = [
                    "action_in_proj", "action_out_proj",
                    "state_proj", "action_time_mlp",
                ]
            for name, param in self.named_parameters():
                if any(proj_name in name for proj_name in projection_names):
                    param.requires_grad = False

    @staticmethod
    def _to_device(t, device, dtype):
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
        return t.to(device=device, dtype=dtype)
