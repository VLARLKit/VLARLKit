

# VLARLKit: An elegant PyTorch VLA-RL library

An elegant and researcher-friendly RL library for Vision-Language-Action (VLA) models.



## ✨ Features

- **Simple and clear implementation** — cleanly separated policy, rollout, runner, and model layers with minimal abstraction; easy to read, modify, and extend for research purposes
- **Dependency-decoupled architecture** — model backends use separate uv projects, while benchmark environments run as independent ZMQ processes; this keeps base-model and simulator dependency conflicts out of the core library
- **Async off-policy training** — supports asynchronous off-policy training, enabling non-blocking data collection alongside model updates

## 🧩 Supported Algorithms, Base Models, and Benchmarks (Work in Progress)


| Category          | Type               | Supported                                                                                                                                                                  |
| ----------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RL Algorithms** | On-policy RL       | [PPO](https://arxiv.org/abs/1707.06347), [GRPO](https://arxiv.org/abs/2402.03300)                                                                                          |
|                   | Off-policy RL      | [DSRL](https://arxiv.org/pdf/2506.15799), [RLT](https://www.pi.website/download/rlt.pdf)                                                                                   |
|                   | Model-based RL     | [VLA-MBPO](https://rhx11111.github.io/VLA-MBPO/)                                                                                                                           |
| **Base Models**   | Flow-based VLA     | [π₀.₅](https://github.com/Physical-Intelligence/openpi)                                                                                                                    |
|                   | Autoregressive VLA | [OpenVLA-OFT](https://github.com/moojink/openvla-oft)                                                                                                                      |
| **Benchmarks**    | Simulation         | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [ManiSkill](https://github.com/haosulab/ManiSkill), [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) |


## 📦 Installation

### 1. Core Library

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
git clone https://github.com/VLARLKit/VLARLKit.git
cd VLARLKit
git checkout compute-canada
module load git-lfs
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv pip install -e .
```

The core package intentionally does not depend on any base-model repository.
Install the model backend you need for each experiment.

### 2. Model Backends (choose one you need)

Each model backend uses its own uv project with its own dependencies. This
keeps base-model repositories separate from the core package while still
training the model in the main `torchrun` process.

Install and prepare the backend you need:

```bash
# OpenPI
GIT_LFS_SKIP_SMUDGE=1 uv sync --project model_backends/openpi
uv run --project model_backends/openpi \
    bash model_backends/openpi/scripts/apply_transformers_patch.sh

# OpenVLA-OFT
uv sync --project model_backends/openvla_oft
uv run --project model_backends/openvla_oft \
    bash model_backends/openvla_oft/scripts/install_flash_attn.sh
```

### 3. Benchmarks (choose one/more you need)

The environment client runs in a **separate** Python environment with its own dependencies. This avoids dependency conflicts between the simulator and the training stack.

Install scripts for each benchmark are located in the `third_party/` directory. Run the one you need:

```bash
# LIBERO
bash third_party/install_libero.sh

# ManiSkill
bash third_party/install_maniskill.sh

# RoboTwin
bash third_party/new_install_robotwin.sh
```

### 🚀 Quick Start

RL process is typically performing on a SFT model. So you need to download such an SFT model first.
We highly recommend you to use models from RLinf community.
For the full benchmark-by-benchmark SFT and RL setup, see [SFT Checkpoints and RL Settings](docs/sft_rl_settings.md).

```bash
# download sft openpi model
hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir <your local path>

# For ManiSkill SFT model:
# hf download RLinf/RLinf-Pi05-ManiSkill-25Main-SFT --local-dir <your local path>

# For RoboTwin SFT model:
# hf download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle --local-dir <your local path>

# download tokenizer of openpi model
mkdir -p $HOME/.cache/openpi/big_vision
wget -O $HOME/.cache/openpi/big_vision/paligemma_tokenizer.model \
  "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
```

Then, change the `model_path` in config file (examples/configs/libero_spatial_ppo_pi05.yaml) to your path.
For example:

```yaml
model:
  model_path: "<your download path>/RLinf-Pi05-LIBERO-SFT"
```

Now, you can lanuch the script to run!

```bash
bash examples/run_onpolicy_rl.sh
```

If you want to have a try with our MBRL method (VLA-MBPO), please follow [BAGEL-WM](https://github.com/VLARLKit/BAGEL/tree/main) to setup envs and artifacts.

## 📋 TODO

- [x] Add ManiSkill benchmark support
- [x] Add RoboTwin benchmark support
- [x] Add GRPO algorithm support
- [x] Add off-policy asynchronous training support
- [x] Add OpenVLA base model support
- [x] Add offline and model-based VLA methods support

## 🙏 Acknowledgements

We borrow some good designs from [RLinf](https://github.com/RLinf/RLinf). The model integration and environment module implementations are primarily adapted from RLinf. We thank the RLinf team for their foundational work.

# 📄 License

This project is licensed under the MIT License (see LICENSE file).

Some source files are derived from Apache-2.0 licensed projects. The original copyright notices are preserved in those files.

## 📚 Citation

If you find VLARLKit useful in your research, please consider citing it:

```bibtex
@misc{vlarlkit2026,
  title = {VLARLKit: An Elegant PyTorch VLA-RL Library},
  author = {Yihao Sun},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {https://github.com/VLARLKit/VLARLKit}
}
```
