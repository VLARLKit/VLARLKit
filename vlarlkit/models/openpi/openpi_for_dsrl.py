from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

from vlarlkit.models.openpi.openpi_for_rl import OpenPi0ForRL


@dataclass(frozen=True)
class OpenPi0DSRLConfig(Pi0Config):
    config_name: str = "pi0_libero"
    num_images_in_input: int = 2
    action_chunk: int = 5
    action_env_dim: int = 7
    num_steps: int = 10
    train_expert_only: bool = False
    # DSRL-specific
    dsrl_state_dim: int = 8
    dsrl_action_noise_dim: int = 32
    dsrl_num_q_heads: int = 10
    dsrl_image_latent_dim: int = 64
    dsrl_state_latent_dim: int = 64
    dsrl_hidden_dims: tuple = field(default_factory=lambda: (128, 128, 128))
    dsrl_action_magnitude: float = 1.0


class OpenPi0ForDSRL(OpenPi0ForRL):

    config: OpenPi0DSRLConfig

    @property
    def _no_split_modules(self) -> list[str]:
        # DSRL-specific float32 modules must be their own FSDP units
        # to avoid mixed-dtype flattening with the bfloat16 base model.
        dsrl_modules = [
            "GaussianPolicy", "CompactMultiQHead",
            "LightweightImageEncoder64", "CompactStateEncoder",
        ]
        if self.config.train_expert_only:
            return [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ] + dsrl_modules
        return [
            "GemmaMLP",
            "SiglipVisionEmbeddings",
            "GemmaRMSNorm",
            "GemmaRotaryEmbedding",
        ] + dsrl_modules

    def __init__(self, config: OpenPi0DSRLConfig):
        from vlarlkit.models.modules.compact_encoders import (
            CompactMultiQHead,
            CompactStateEncoder,
            LightweightImageEncoder64,
        )
        from vlarlkit.models.modules.gaussian_policy import GaussianPolicy

        # Bypass OpenPi0ForRL.__init__, call PI0Pytorch directly
        sample_actions_func = self.sample_actions
        PI0Pytorch.__init__(self, config)
        self.sample_actions = sample_actions_func
        self.global_step = 0

        # DSRL-specific modules
        _dsrl_dtype = torch.float32
        dsrl_input_dim = config.dsrl_state_latent_dim + config.dsrl_image_latent_dim

        mag = config.dsrl_action_magnitude
        self.dsrl_action_noise_net = GaussianPolicy(
            input_dim=dsrl_input_dim,
            output_dim=config.dsrl_action_noise_dim,
            hidden_dims=config.dsrl_hidden_dims,
            low=-mag if mag != 1.0 else None,
            high=mag if mag != 1.0 else None,
            action_horizon=config.action_horizon,
        ).to(dtype=_dsrl_dtype)

        self.actor_image_encoder = LightweightImageEncoder64(
            num_images=1,
            latent_dim=config.dsrl_image_latent_dim,
            image_size=64,
        ).to(dtype=_dsrl_dtype)
        self.actor_state_encoder = CompactStateEncoder(
            state_dim=config.dsrl_state_dim,
            hidden_dim=config.dsrl_state_latent_dim,
        ).to(dtype=_dsrl_dtype)
        self.critic_image_encoder = LightweightImageEncoder64(
            num_images=1,
            latent_dim=config.dsrl_image_latent_dim,
            image_size=64,
        ).to(dtype=_dsrl_dtype)
        self.critic_state_encoder = CompactStateEncoder(
            state_dim=config.dsrl_state_dim,
            hidden_dim=config.dsrl_state_latent_dim,
        ).to(dtype=_dsrl_dtype)
        self.q_head = CompactMultiQHead(
            state_dim=config.dsrl_state_latent_dim,
            image_dim=config.dsrl_image_latent_dim,
            action_dim=config.dsrl_action_noise_dim,
            hidden_dims=config.dsrl_hidden_dims,
            num_q_heads=config.dsrl_num_q_heads,
            output_dim=1,
        ).to(dtype=_dsrl_dtype)

        from vlarlkit.models.modules.image_augmentation import ColorJitter, RandomCrop
        self._augmentations = [RandomCrop(pad=4), ColorJitter()]

        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    def forward(self, forward_type="actor", **kwargs):
        if forward_type == "actor":
            return self.actor_forward(**kwargs)
        elif forward_type == "critic":
            return self.critic_forward(**kwargs)
        elif forward_type == "actor_critic":
            actions, log_probs, _ = self.actor_forward(**kwargs)
            q_values = self.critic_forward(
                obs=kwargs.get("obs"), actions=actions,
                detach_encoder=kwargs.get("detach_encoder", False),
                aug_img=kwargs.get("aug_img", False),
            )
            return actions, log_probs, q_values
        else:
            raise ValueError(f"Unknown forward_type: {forward_type}")

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

        # SAC agent outputs noise
        dsrl_obs = {"images": [env_obs["main_images"]], "states": env_obs["states"]}
        noise_actions, noise_logprob, _ = self.actor_forward(
            obs=dsrl_obs, aug_img=False, mode=mode
        )

        # Use noise to steer frozen Pi0 denoising
        outputs = self.sample_actions(observation, noise=noise_actions)

        # Extract real actions for env
        actions = self.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )["actions"].numpy()

        forward_inputs = {
            "action": noise_actions,
        }
        result = {
            "prev_logprobs": noise_logprob,
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
        """Pure ODE denoising — no SDE, no chains/logprobs tracking."""
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps

        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        else:
            noise = noise.to(self.action_in_proj.weight.dtype)

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

        return {"actions": x_t}

    def freeze_vlm(self):
        super().freeze_vlm()
        # DSRL: also freeze gemma_expert and projection layers
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

    def actor_forward(self, obs=None, aug_img=False, **kwargs):
        if obs is None:
            obs = kwargs.get("obs", {})

        if "images" not in obs:
            if "main_images" in obs:
                obs = {"images": [obs["main_images"]], "states": obs["states"]}
            else:
                raise ValueError(f"Invalid obs format: {obs.keys()}")

        images = self._preprocess_dsrl_images(obs["images"], aug_img=aug_img)
        states = self._preprocess_states(obs["states"])

        _p = next(self.actor_image_encoder.parameters())
        device, _dtype = _p.device, _p.dtype
        images = images.to(device=device, dtype=_dtype)
        states = states.to(device=device, dtype=_dtype)

        image_features = self.actor_image_encoder(images)
        state_features = self.actor_state_encoder(states)
        features = torch.cat([state_features, image_features], dim=-1)

        mode = kwargs.get("mode", "train")
        deterministic = mode == "eval"
        action_noise, logprobs = self.dsrl_action_noise_net(
            features, deterministic=deterministic
        )

        return action_noise, logprobs, None

    def critic_forward(self, obs=None, actions=None, detach_encoder=False, aug_img=False, **kwargs):
        if obs is None:
            obs = kwargs.get("obs", {})
        if actions is None:
            actions = kwargs.get("actions")

        if "images" not in obs:
            if "main_images" in obs:
                obs = {"images": [obs["main_images"]], "states": obs["states"]}
            else:
                raise ValueError(f"Invalid obs format: {obs.keys()}")

        images = self._preprocess_dsrl_images(obs["images"], aug_img=aug_img)
        states = self._preprocess_states(obs["states"])

        _p = next(self.critic_image_encoder.parameters())
        device, _dtype = _p.device, _p.dtype
        images = images.to(device=device, dtype=_dtype)
        states = states.to(device=device, dtype=_dtype)
        actions = actions.to(device=device, dtype=_dtype)

        image_features = self.critic_image_encoder(images)
        state_features = self.critic_state_encoder(states)

        if detach_encoder:
            image_features = image_features.detach()
            state_features = state_features.detach()

        if actions.dim() == 3:
            actions = actions[:, 0, :]

        q_values = self.q_head(state_features, image_features, actions)
        return q_values

    def _preprocess_dsrl_images(self, images, aug_img=False):
        import torch.nn.functional as F

        agentview_img = images[0] if isinstance(images, list) else images

        if not isinstance(agentview_img, torch.Tensor):
            agentview_img = torch.from_numpy(agentview_img)

        # NHWC -> NCHW
        if agentview_img.shape[-1] == 3:
            agentview_img = agentview_img.permute(0, 3, 1, 2)

        # Normalize to [0, 1]
        if agentview_img.dtype == torch.uint8:
            agentview_img = agentview_img.float() / 255.0
        else:
            if agentview_img.min() < 0:
                agentview_img = (agentview_img + 1.0) / 2.0
        agentview_img = agentview_img.clamp(0.0, 1.0)

        resized_img = F.interpolate(
            agentview_img, size=(64, 64), mode="bilinear", align_corners=False,
        )

        # Data augmentation (operates on [0, 1] range)
        if aug_img:
            for aug in self._augmentations:
                resized_img = aug(resized_img)

        # [0, 1] -> [-1, 1]
        resized_img = resized_img * 2.0 - 1.0
        # [B, C, 64, 64] -> [B, 1, C, 64, 64]
        resized_img = resized_img.unsqueeze(1)
        return resized_img

    def _preprocess_states(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.from_numpy(states)
        if states.dim() > 2:
            states = states.reshape(states.shape[0], -1)
        return states.float()
