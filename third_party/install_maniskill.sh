#!/bin/bash
# Install ManiSkill benchmark environment

conda create -n maniskill python=3.10 -y
conda activate maniskill

pip install --upgrade mani_skill torch zmq omegaconf huggingface_hub[cli]
sudo apt-get install -y libvulkan1 vulkan-tools

# conda install conda-forge::vulkan-tools conda-forge::vulkan-headers
# export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# Download ManiSkill built-in assets
python -m mani_skill.utils.download_asset bridge_v2_real2sim
python -m mani_skill.utils.download_asset widowx250s

# Download RLinf custom task assets (carrot, plate, table, etc.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hf download --repo-type dataset RLinf/maniskill_assets --local-dir "$SCRIPT_DIR/../env_clients/maniskill/assets"