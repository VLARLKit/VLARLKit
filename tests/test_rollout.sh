#!/bin/bash
eval "$(conda shell.bash hook)"

# launch the environment client
conda activate libero
python -m env_clients.client \
    --config examples/configs/libero_10_ppo_pi05.yaml \
    --host 0.0.0.0 --port 5550 \
    --rank 0 --world_size 1 &
conda deactivate

# launch the rollout worker
source .venv/bin/activate
python tests/test_rollout.py