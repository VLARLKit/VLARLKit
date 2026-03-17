#!/bin/bash
# Install LIBERO benchmark environment
# This runs in a separate conda environment to avoid dependency conflicts.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Initialize submodule if not already cloned
git submodule update --init "$SCRIPT_DIR/LIBERO"

conda create -n libero python=3.8 -y
conda activate libero
cd "$SCRIPT_DIR/LIBERO"
touch libero/__init__.py
pip install cmake==3.24.3
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install zmq
pip install -e .
