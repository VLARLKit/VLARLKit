#!/bin/bash
cd ~/codes/VLARLKit

NPROC=4
CONFIG_NAME="libero_spatial_rlt_pi05"
CONFIG="examples/configs/${CONFIG_NAME}.yaml"
BASE_PORT=5550

# launch environment clients (in the libero conda env)
CLIENT_PIDS=()
for ((i=0; i<NPROC; i++)); do
    CUDA_VISIBLE_DEVICES="$i" conda run -n libero \
        python -m env_clients.client \
        --config "$CONFIG" \
        --host 0.0.0.0 --port $((BASE_PORT + i)) \
        --rank "$i" --world_size "$NPROC" &
    CLIENT_PIDS+=($!)
done

# launch training with torchrun
uv run torchrun --nproc_per_node="$NPROC" \
    examples/train_offpolicy_rl.py \
    --config-name "$CONFIG_NAME"

# cleanup: kill all client processes
for pid in "${CLIENT_PIDS[@]}"; do
    kill "$pid" 2>/dev/null
done
wait
