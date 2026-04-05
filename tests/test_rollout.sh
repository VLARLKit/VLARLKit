#!/bin/bash

module load gcc opencv
PROJECT_ROOT=$SCRATCH/VLARLKit
CONFIG_NAME="maniskill_ppo_pi05"
CONFIG="$PROJECT_ROOT/examples/configs/${CONFIG_NAME}.yaml"

# launch the environment client
cd $SCRATCH/VLARLKit/third_party/maniskill
PYTHONPATH=$PROJECT_ROOT uv run --no-sync python -m env_clients.client \
    --config $CONFIG \
    --host 0.0.0.0 --port 5550 \
    --rank 0 --world_size 1 &
ENV_PID=$!

# launch the rollout worker
cd $SCRATCH/VLARLKit
uv run --no-sync python tests/test_rollout.py --config-name $CONFIG_NAME

# kill the env client process
kill $ENV_PID 2>/dev/null
wait $ENV_PID 2>/dev/null