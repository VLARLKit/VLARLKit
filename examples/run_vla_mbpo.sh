#!/bin/bash
set -euo pipefail

cd ~/codes/VLARLKit

NPROC=4
CONFIG_NAME="libero_goal_vla_mbpo"
CONFIG="examples/configs/${CONFIG_NAME}.yaml"

WM_BASE_PORT=8002
ENV_BASE_PORT=5550
WAIT_HOST="127.0.0.1"
PORT_WAIT_TIMEOUT=900
LOAD_MODEL_PATH="/home/mila/s/sunyi/scratch/Bagel-WM/Bagel-libero-goal"

WM_PIDS=()
ENV_PIDS=()

cleanup() {
    trap - EXIT INT TERM
    for pid in "${ENV_PIDS[@]}" "${WM_PIDS[@]}"; do
        kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}

wait_for_port() {
    local port="$1"
    local name="$2"
    local pid="$3"
    local start="$SECONDS"

    while true; do
        if python -c 'import socket, sys; s = socket.socket(); s.settimeout(1); s.connect((sys.argv[1], int(sys.argv[2]))); s.close()' "$WAIT_HOST" "$port" >/dev/null 2>&1; then
            echo "$name is ready on ${WAIT_HOST}:${port}"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$name process exited before ${WAIT_HOST}:${port} became ready" >&2
            return 1
        fi
        if (( SECONDS - start >= PORT_WAIT_TIMEOUT )); then
            echo "Timed out waiting for $name on ${WAIT_HOST}:${port}" >&2
            return 1
        fi
        sleep 2
    done
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# launch BAGEL world-model clients on gpu0-3
for ((i=0; i<NPROC; i++)); do
    CUDA_VISIBLE_DEVICES="$i" setsid third_party/BAGEL/.venv/bin/python \
        -m env_clients.world_models.bagel.client \
        --load-model-path "$LOAD_MODEL_PATH" \
        --host 0.0.0.0 --port $((WM_BASE_PORT + i)) &
    WM_PIDS+=($!)
done

for ((i=0; i<NPROC; i++)); do
    wait_for_port $((WM_BASE_PORT + i)) "world model client $i" "${WM_PIDS[$i]}"
done

# launch eval environment clients on gpu4-7
for ((i=0; i<NPROC; i++)); do
    gpu=$((i + 4))
    CUDA_VISIBLE_DEVICES="$gpu" setsid conda run -n libero \
        python -m env_clients.client \
        --config "$CONFIG" \
        --host 0.0.0.0 --port $((ENV_BASE_PORT + i)) \
        --rank "$i" --world_size "$NPROC" \
        --modes eval &
    ENV_PIDS+=($!)
done

for ((i=0; i<NPROC; i++)); do
    wait_for_port $((ENV_BASE_PORT + i)) "eval environment client $i" "${ENV_PIDS[$i]}"
done

# launch VLA-MBPO training on gpu4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run --project model_backends/openpi --no-sync torchrun --nproc_per_node="$NPROC" \
    examples/train_vla_mbpo.py \
    --config-name "$CONFIG_NAME" \
    world_model.base_port="$WM_BASE_PORT" \
    world_model.load_model_path="$LOAD_MODEL_PATH" \
    env.env_client_base_port="$ENV_BASE_PORT"
