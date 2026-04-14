#!/bin/bash
# Install ManiSkill benchmark environment

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir "$SCRIPT_DIR/maniskill" && cd "$SCRIPT_DIR/maniskill"

uv init --no-workspace .
uv venv
source .venv/bin/activate

uv pip install --no-cache "mani_skill==3.0.0b22" zmq omegaconf hydra-core huggingface_hub[cli]
uv pip uninstall torch
uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-cache "numpy==1.26.4"

# Download ManiSkill built-in assets
uv run --no-sync python -m mani_skill.utils.download_asset bridge_v2_real2sim
uv run --no-sync python -m mani_skill.utils.download_asset widowx250s

# Download RLinf custom task assets (carrot, plate, table, etc.)
uv run --no-sync hf download --repo-type dataset RLinf/maniskill_assets --local-dir "$SCRIPT_DIR/../env_clients/maniskill/assets"

# Download PhysX precompiled library
wget https://github.com/sapien-sim/physx-precompiled/releases/download/105.1-physx-5.3.1.patch0/linux-so.zip -O /tmp/linux-so.zip
mkdir -p $HOME/.sapien/physx/105.1-physx-5.3.1.patch0
unzip /tmp/linux-so.zip -d $HOME/.sapien/physx/105.1-physx-5.3.1.patch0

deactivate
