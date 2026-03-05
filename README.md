# VLARLKit: An elegant PyTorch VLA-RL library

An elegant and researcher-friendly RL library for Vision-Language-Action (VLA) models.

## Features

- **Simple and clear implementation** — cleanly separated policy, rollout, runner, and model layers with minimal abstraction; easy to read, modify, and extend for research purposes
- **Environment-decoupled architecture** — environments run as independent processes via ZMQ, eliminating dependency conflicts between different benchmark simulators
- **Async off-policy training (on building)** — supports asynchronous off-policy training, enabling non-blocking data collection alongside model updates

## Supported Algorithms, Base Models, and Benchmarks (Keeping progress)

| Category | Supported |
|---|---|
| **RL Algorithms** | PPO |
| **Base Models** | $\pi_{\text{0.5}}$ |
| **Benchmarks** | LIBERO |

## Installation

### 1. Main Library

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
git clone https://github.com/VLARLKit/VLARLKit.git
cd VLARLKit
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 2. Benchmarks (Optional)

The environment client runs in a **separate** Python environment with its own dependencies. This avoids dependency conflicts between the simulator and the training stack.

You can choose install any one you need.

#### Install LIBERO
```bash
conda create -n libero python=3.8
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
touch libero/__init__.py
pip install cmake==3.24.3
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

### Quick Start

RL process is typically performing on a SFT model. So you need to download such an SFT model first.
We highly recommend you to use models from RLinf community.

```bash
huggingface-cli download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir <your local path>
```

Then, change the ``model_path`` and ``assets_dir`` in config file (examples/configs/libero_spatial_ppo_pi05.yaml) to your path.
For example:
```yaml
model:
  model_path: "<your download path>/RLinf-Pi05-LIBERO-SFT"
  data:
    assets_dir: "<your download path>/RLinf-Pi05-LIBERO-SFT/physical-intelligence/libero"
```

Now, you can lanuch the script to run!

```bash
bash examples/run_onpolicy_rl.sh
```

## TODO

- [ ] Add CALVIN and ManiSkill benchmark support
- [ ] Add GRPO algorithm support
- [ ] Add off-policy asynchronous training support
- [ ] Add OpenVLA base model support

## Acknowledgements
We borrow some good designs from [RLinf](https://github.com/RLinf/RLinf). The model integration and environment module implementations are primarily adapted from RLinf. We thank the RLinf team for their foundational work.

# License
This project is licensed under the MIT License (see LICENSE file).

Some source files are derived from Apache-2.0 licensed projects. The original copyright notices are preserved in those files.