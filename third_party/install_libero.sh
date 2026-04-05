#!/usr/bin/env bash
set -euo pipefail

# Resolve the absolute path of the directory containing this script (VLARLKit/third_party).
# This ensures correct paths regardless of which directory the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Installing LIBERO into: $SCRIPT_DIR/LIBERO"

# Initialize the LIBERO git submodule
git submodule update --init "$SCRIPT_DIR/LIBERO"

# Enter the LIBERO directory
cd "$SCRIPT_DIR/LIBERO"

# Create an empty __init__.py so Python recognizes libero as a package
touch libero/__init__.py

# Write pyproject.toml directly via heredoc (no editor required)
# Single-quoted 'EOF' prevents the shell from expanding any special characters inside
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "libero"
version = "0.1.0"
description = "Dependencies for LIBERO project"
readme = "README.md"
requires-python = ">=3.8,<3.9"
dependencies = [
    "hydra-core==1.2.0",
    "numpy==1.22.4",
    "wandb==0.13.1",
    "easydict==1.9",
    "transformers==4.21.1",
    "opencv-python==4.6.0.66",
    "robomimic==0.2.0",
    "einops==0.4.1",
    "thop==0.1.1-2209072238",
    "robosuite==1.4.0",
    "bddl==1.0.1",
    "future==0.18.2",
    "matplotlib==3.5.3",
    "cloudpickle==2.1.0",
    "gym==0.25.2",
    "cmake==3.24.3",
    "zmq>=0.0.0"
]
EOF

echo "==> pyproject.toml written successfully"

# Sync dependencies declared in pyproject.toml
uv sync --no-cache

# Install PyTorch with CUDA 11.3 support
uv pip install \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install the LIBERO package itself in editable mode
uv pip install -e .

echo "==> LIBERO installation complete!"
