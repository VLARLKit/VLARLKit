# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# openpi model configs
#
# --------------------------------------------------------------------
# Modifications:
#   Modified by VLARLKit Authors on 2026-02-11.
# --------------------------------------------------------------------

import os

from omegaconf import DictConfig


_MODEL_REGISTRY = None


def _get_registry():
    global _MODEL_REGISTRY
    if _MODEL_REGISTRY is None:
        from vlarlkit.models.openpi.openpi_for_rl import OpenPi0RLConfig, OpenPi0ForRL
        from vlarlkit.models.openpi.openpi_for_dsrl import OpenPi0DSRLConfig, OpenPi0ForDSRL

        _MODEL_REGISTRY = {
            "OpenPi0ForRL": (OpenPi0RLConfig, OpenPi0ForRL),
            "OpenPi0ForDSRL": (OpenPi0DSRLConfig, OpenPi0ForDSRL),
        }
    return _MODEL_REGISTRY


def get_model(cfg: DictConfig):
    import glob

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints
    from openpi.training.config import AssetsConfig, DataConfig

    from vlarlkit.models.openpi.dataconfigs import get_data_config

    # Route to the correct model class
    model_config_dict = dict(cfg.openpi)
    model_class_name = model_config_dict.pop("model_class_name", "OpenPi0ForRL")

    registry = _get_registry()
    config_cls, model_cls = registry[model_class_name]
    model_config = config_cls(**model_config_dict)

    # load model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))
    weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not weight_paths:
        weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]

    model = model_cls(model_config)
    # train expert only
    if model_config.train_expert_only:
        model.freeze_vlm()

    for weight_path in weight_paths:
        safetensors.torch.load_model(model, weight_path, strict=False)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    # load data stats
    data_config_cls = get_data_config(cfg.data.name)(
        repo_id=cfg.data.repo_id,
        base_config=DataConfig(prompt_from_task=cfg.data.prompt_from_task),
        assets=AssetsConfig(assets_dir=cfg.data.assets_dir),
        extra_delta_transform=cfg.data.extra_delta_transform,
    )
    data_config = data_config_cls.create(
        cfg.data.assets_dir, model_config
    )
    # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
    # that the policy is using the same normalization stats as the original training process.
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)

    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
