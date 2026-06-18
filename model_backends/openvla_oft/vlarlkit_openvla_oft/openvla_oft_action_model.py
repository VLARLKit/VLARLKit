# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0.
#
# --------------------------------------------------------------------
# Modifications:
#   Modified by VLARLKit Authors on 2026-06-07.
# --------------------------------------------------------------------

from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from prismatic.extern.hf.configuration_prismatic import (
    OpenVLAConfig as OpenVLAOFTConfig,
)
from prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction as OpenVLAOFTForActionPrediction,
)
from prismatic.vla.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    STOP_INDEX,
    NormalizationType,
)
from transformers.generation import TopKLogitsWarper

from vlarlkit.models.base import BaseModel
from vlarlkit.models.modules.value_head import ValueHead


def _logprobs_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch_shape = logits.shape[:-1]
    logits = logits.reshape(-1, logits.shape[-1])
    target = target.reshape(-1)
    logprobs = -F.cross_entropy(logits, target, reduction="none")
    return logprobs.view(*batch_shape).float()


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -torch.where(p > 0, p * logp, 0.0).sum(dim=-1)


class OpenVLAOFTForRLActionPrediction(OpenVLAOFTForActionPrediction, BaseModel):
    def __init__(
        self,
        config: OpenVLAOFTConfig,
        action_dim: int,
        num_action_chunks: int,
        add_value_head: bool,
        max_prompt_length: int,
    ) -> None:
        super().__init__(config)

        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.unnorm_key = config.unnorm_key
        if (
            self.unnorm_key not in self.norm_stats
            and f"{self.unnorm_key}_no_noops" in self.norm_stats
        ):
            self.unnorm_key = f"{self.unnorm_key}_no_noops"
        assert self.unnorm_key in self.norm_stats, (
            f"Action un-norm key {self.unnorm_key} not found in VLA norm_stats."
        )

        if add_value_head:
            output_dim = (
                1 if self.config.value_type == "chunk_level" else self.num_action_chunks
            )
            self.value_head = ValueHead(
                input_dim=self.config.hidden_size,
                hidden_sizes=(512, 128),
                output_dim=output_dim,
                activation="gelu",
                bias_last=False,
            )

        self.max_prompt_length = max_prompt_length
        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    @property
    def _no_split_modules(self) -> list[str]:
        names = [
            "LlamaDecoderLayer",
            "MistralDecoderLayer",
            "GemmaDecoderLayer",
            "VisionTransformer",
            "PrismaticProjector",
        ]
        if hasattr(self, "value_head"):
            names.append("ValueHead")
        return names

    @property
    def _no_split_names(self) -> list[str]:
        return ["projector", "lm_head"]

    def _build_embedding(self, input_ids, attention_mask, pixel_values):
        assert torch.all(input_ids[:, -1] == STOP_INDEX)
        assert input_ids.shape == attention_mask.shape

        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]
        n_patch_tokens = (
            self.vision_backbone.get_num_patches()
            * self.vision_backbone.get_num_images_in_input()
        )

        all_actions_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        all_actions_mask[:, -self.action_dim * self.num_action_chunks :] = True

        input_embeddings = self.get_input_embeddings()(input_ids)
        input_embeddings = input_embeddings * (~all_actions_mask.unsqueeze(-1))

        projected_patch_embeddings = self._process_vision_features(
            pixel_values,
            None,
            use_film=False,
        )
        assert projected_patch_embeddings.shape[1] == n_patch_tokens
        projected_patch_embeddings = projected_patch_embeddings.reshape(
            input_embeddings.shape[0],
            -1,
            *projected_patch_embeddings.shape[2:],
        )
        multimodal_embeddings, multimodal_attention_mask = (
            self._build_multimodal_attention(
                input_embeddings,
                projected_patch_embeddings,
                attention_mask,
            )
        )
        return multimodal_embeddings, multimodal_attention_mask

    def _get_action_stats(self) -> dict[str, Any]:
        unnorm_key = self._check_unnorm_key(self.norm_stats, self.unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        placeholder_action_token_ids = torch.ones(
            (input_ids.shape[0], self.action_dim * self.num_action_chunks),
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        stop_token_id = torch.ones(
            (input_ids.shape[0], 1),
            device=input_ids.device,
            dtype=input_ids.dtype,
        ) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        mask_extension = torch.ones(
            (attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)
        return input_ids, attention_mask

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        action_norm_stats = self.get_action_stats(unnorm_key)
        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get(
                "mask",
                np.ones_like(action_norm_stats["min"], dtype=bool),
            )
            action_high = np.array(action_norm_stats["max"])
            action_low = np.array(action_norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get(
                "mask",
                np.ones_like(action_norm_stats["q01"], dtype=bool),
            )
            action_high = np.array(action_norm_stats["q99"])
            action_low = np.array(action_norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type.")

        repeat_factor = normalized_actions.shape[-1] // action_high.shape[0]
        action_high = np.tile(action_high, repeat_factor)
        action_low = np.tile(action_low, repeat_factor)
        mask = np.tile(mask, repeat_factor)

        return np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8)
            + action_low,
            normalized_actions,
        )

    def _prepare_env_inputs(self, env_obs):
        task_descriptions = [
            f"In: What action should the robot take to {task.lower()}?\nOut: "
            for task in env_obs["task_descriptions"]
        ]

        main_images = torch.as_tensor(env_obs["main_images"])
        if main_images.ndim == 4:
            main_images = main_images.unsqueeze(1)
        assert main_images.ndim == 5
        all_images = [main_images.permute(0, 1, 4, 2, 3)]

        wrist_images = env_obs.get("wrist_images")
        if self.vision_backbone.get_num_images_in_input() > 1:
            assert wrist_images is not None
            wrist_images = torch.as_tensor(wrist_images)
            if wrist_images.ndim == 4:
                wrist_images = wrist_images.unsqueeze(1)
            assert wrist_images.ndim == 5
            wrist_images = wrist_images.permute(0, 1, 4, 2, 3)
            all_images.extend([wrist_images[:, i] for i in range(wrist_images.shape[1])])

        states = torch.as_tensor(env_obs["states"])
        device = next(self.parameters()).device
        precision = next(self.parameters()).dtype

        primary_image = all_images.pop(0)
        inputs = self.input_processor(
            text=task_descriptions,
            images={"images": primary_image},
            proprio_states=states,
            padding="max_length",
            max_length=self.max_prompt_length,
        )
        if all_images:
            wrist_inputs = [
                self.input_processor(
                    text=task_descriptions,
                    images={"images": wrist_image.unsqueeze(1)},
                    proprio_states=states,
                    padding="max_length",
                    max_length=self.max_prompt_length,
                )
                for wrist_image in all_images
            ]
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"]]
                + [wrist_input["pixel_values"] for wrist_input in wrist_inputs],
                dim=1,
            )

        input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
        attention_mask = inputs["attention_mask"].to(device=device, dtype=torch.bool)
        pixel_values = inputs["pixel_values"].to(device=device, dtype=precision)
        batch_size, num_images, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(
            batch_size,
            num_images * channels,
            height,
            width,
        )
        return input_ids, attention_mask, pixel_values

    @torch.no_grad()
    def predict_action_batch(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        env_obs=None,
        calculate_logprobs=True,
        calculate_values=True,
        mode: str = "train",
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        do_sample = bool(kwargs.pop("do_sample", mode == "train"))
        temperature = float(kwargs.pop("temperature", 1.0))
        top_k = int(kwargs.pop("top_k", -1))

        if env_obs is None and isinstance(input_ids, dict):
            env_obs = input_ids
            input_ids = None
            attention_mask = None
            pixel_values = None
        elif env_obs is None and input_ids is None:
            env_obs = kwargs.pop("obs", None)
        if env_obs is not None:
            input_ids, attention_mask, pixel_values = self._prepare_env_inputs(env_obs)

        # NumPy cannot store bf16 tensors directly; fp16 keeps this image cache compact.
        cached_pixel_values = pixel_values
        if cached_pixel_values.dtype == torch.bfloat16:
            cached_pixel_values = cached_pixel_values.to(torch.float16)

        forward_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": cached_pixel_values,
        }

        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        n_prompt_tokens = input_ids.shape[-1] - 1
        n_patches = (
            self.vision_backbone.get_num_patches()
            * self.vision_backbone.get_num_images_in_input()
        )

        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids,
            attention_mask,
        )
        mm_embeddings, mm_attention_mask = self._build_embedding(
            input_ids,
            attention_mask,
            pixel_values,
        )
        multimodal_position_ids = mm_attention_mask.cumsum(dim=1) - 1

        outputs = self.language_model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        action_token_count = self.action_dim * self.num_action_chunks
        logits = outputs.logits[
            :,
            n_patches + n_prompt_tokens : n_patches
            + n_prompt_tokens
            + action_token_count,
            :,
        ]
        last_hidden_states = outputs.hidden_states[-1][:, -action_token_count - 1 : -1]
        logits[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
        logits[..., self.vocab_size :] = -torch.inf

        if do_sample:
            processed_logits = logits / temperature
            top_k = min(top_k, processed_logits.size(-1))
            if top_k > 0:
                processed_logits = TopKLogitsWarper(top_k)(None, processed_logits)
            probs = F.softmax(processed_logits, dim=-1)
            idxs = torch.multinomial(
                probs.view(-1, probs.shape[-1]),
                num_samples=1,
                replacement=True,
            ).view(probs.shape[0], probs.shape[1])
        else:
            processed_logits = logits
            idxs = processed_logits.argmax(dim=-1)

        assert torch.all(idxs >= self.vocab_size - self.config.n_action_bins)
        assert torch.all(idxs < self.vocab_size)

        action_tokens = idxs.reshape(-1, self.action_dim)
        discretized_actions = self.vocab_size - action_tokens.cpu().numpy()
        discretized_actions = np.clip(
            discretized_actions - 1,
            a_min=0,
            a_max=self.bin_centers.shape[0] - 1,
        )
        normalized_actions = np.asarray(
            [self.bin_centers[discretized_action] for discretized_action in discretized_actions]
        ).reshape(-1, self.action_dim)
        actions = self._unnormalize_actions(normalized_actions, self.unnorm_key)
        actions = actions.reshape(idxs.shape)

        action_logits = processed_logits
        action_logits[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
        action_logits[..., self.vocab_size :] = -torch.inf
        chunk_logprobs = _logprobs_from_logits(action_logits, idxs)

        if hasattr(self, "value_head") and calculate_values:
            hidden_features = last_hidden_states[:, -action_token_count]
            chunk_values = self.value_head(hidden_features)
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        chunk_actions = torch.as_tensor(
            actions.reshape(-1, self.num_action_chunks, self.action_dim),
            dtype=torch.float32,
        )
        forward_inputs["action_tokens"] = idxs.reshape(
            -1,
            self.num_action_chunks,
            self.action_dim,
        )
        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result

    def preprocess_for_train(self, data):
        data = dict(data)
        action_tokens = data["action_tokens"]
        data["action_tokens"] = action_tokens.reshape(
            action_tokens.shape[0],
            self.action_dim * self.num_action_chunks,
            *action_tokens.shape[3:],
        )
        return data

    def setup_config_and_processor(self, model_config, input_processor):
        self.vocab_size = (
            model_config.text_config.vocab_size - model_config.pad_to_multiple_of
        )
        self.bins = np.linspace(-1, 1, model_config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        action_norm_stats = self._get_action_stats()
        self.min_action = np.array(action_norm_stats["q01"])
        self.max_action = np.array(action_norm_stats["q99"])
        self.action_scale = 1.0
        self.input_processor = input_processor

    def forward(self, **kwargs):
        return self.default_forward(**kwargs)

    def default_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: bool = False,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        temperature = float(kwargs.pop("temperature", 1.0))
        top_k = int(kwargs.pop("top_k", -1))

        if forward_inputs is not None:
            forward_inputs = self.preprocess_for_train(forward_inputs)
            input_ids = forward_inputs["input_ids"]
            attention_mask = forward_inputs["attention_mask"]
            pixel_values = forward_inputs["pixel_values"]
            action_tokens = forward_inputs["action_tokens"]

        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        model_param = next(self.parameters())
        attention_mask = attention_mask.to(torch.long)
        pixel_values = pixel_values.to(device=model_param.device, dtype=model_param.dtype)
        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids,
            attention_mask,
        )
        mm_embeddings, mm_attention_mask = self._build_embedding(
            input_ids,
            attention_mask,
            pixel_values,
        )
        multimodal_position_ids = mm_attention_mask.cumsum(dim=1) - 1

        if compute_values:
            output_hidden_states = True

        outputs = self.language_model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if not compute_logprobs and not compute_values:
            return outputs

        logprobs = None
        entropy = None
        action_token_count = self.action_dim * self.num_action_chunks
        if compute_logprobs:
            logits = outputs.logits[:, -action_token_count - 1 : -1]
            action_logits = logits / temperature
            top_k = min(top_k, action_logits.size(-1))
            if top_k > 0:
                action_logits = TopKLogitsWarper(top_k)(None, action_logits)

            action_logits[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
            action_logits[..., self.vocab_size :] = -torch.inf
            logprobs = _logprobs_from_logits(action_logits, action_tokens)
            if compute_entropy:
                entropy = _entropy_from_logits(action_logits)

        if hasattr(self, "value_head") and compute_values:
            last_hidden_state = outputs.hidden_states[-1]
            hidden_features = last_hidden_state[:, -action_token_count - 1]
            values = self.value_head(hidden_features)
        else:
            values = None

        return {
            "logprobs": logprobs,
            "entropy": entropy,
            "values": values,
        }
