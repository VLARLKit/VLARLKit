#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

NPROC=${NPROC:-4}
ITERATIONS=${ITERATIONS:-10}
START_ITERATION=${START_ITERATION:-1}
CONFIG_NAME=${CONFIG_NAME:-libero_10_qrflow_pi05}
BASE_PORT=${BASE_PORT:-5550}
MODEL_PYTHON=${MODEL_PYTHON:-model_backends/openpi/.venv/bin/python}
ENV_PYTHON=${ENV_PYTHON:-/network/scratch/s/sunyi/.conda/envs/libero/bin/python}
OUT_DIR=${OUT_DIR:-outputs/qrflow_iterative}
ACTOR=${ACTOR_CHECKPOINT:-null}
CRITIC=${CRITIC_CHECKPOINT:-null}
REPLAY=${REPLAY_PATH:-null}
mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)
export PYTHONPATH="$PWD:$PWD/model_backends/openpi${PYTHONPATH:+:$PYTHONPATH}"
ulimit -Sn 65536
ulimit -Hn 131072
CACHE_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/qrflow-${SLURM_JOB_ID:-$$}"
export CUDA_CACHE_PATH="$CACHE_DIR/cuda" TRITON_CACHE_DIR="$CACHE_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/inductor"
mkdir -p "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

CLIENT_PIDS=()
cleanup() {
    for pid in "${CLIENT_PIDS[@]}"; do
        kill -- -"$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    CLIENT_PIDS=()
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ((iteration=START_ITERATION; iteration<=ITERATIONS; iteration++)); do
    printf -v ROUND_DIR '%s/iteration%02d' "$OUT_DIR" "$iteration"
    mkdir "$ROUND_DIR"
    COMMON=(--config-name "$CONFIG_NAME" "$@"
        "runner.iteration=$iteration" "runner.actor_checkpoint=$ACTOR"
        "env.env_client_base_port=$BASE_PORT" "env.train.total_num_envs=$((NPROC * 5))")

    "$MODEL_PYTHON" examples/collect_qrflow.py \
        "${COMMON[@]}" "runner.previous_replay=$REPLAY" --cfg job --resolve > "$ROUND_DIR/config.yaml"
    for ((rank=0; rank<NPROC; rank++)); do
        CUDA_VISIBLE_DEVICES="$rank" setsid --wait "$ENV_PYTHON" \
            -m env_clients.client --config "$ROUND_DIR/config.yaml" \
            --host 127.0.0.1 --port "$((BASE_PORT + rank))" \
            --rank "$rank" --world_size "$NPROC" --modes train > "$ROUND_DIR/env-$rank.log" 2>&1 &
        CLIENT_PIDS+=($!)
    done
    "$MODEL_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
        examples/collect_qrflow.py "${COMMON[@]}" \
        "runner.previous_replay=$REPLAY" "hydra.run.dir=$ROUND_DIR/collection"
    cleanup

    REPLAY="$ROUND_DIR/collection/replay.json"
    "$MODEL_PYTHON" examples/train_qrflow_critic.py \
        "${COMMON[@]}" "runner.replay_path=$REPLAY" "runner.critic_checkpoint=$CRITIC" \
        "hydra.run.dir=$ROUND_DIR/critic"
    CRITIC="$ROUND_DIR/critic/checkpoint.pt"
    "$MODEL_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
        examples/train_qrflow_actor.py "${COMMON[@]}" "runner.replay_path=$REPLAY" \
        "runner.critic_checkpoint=$CRITIC" "runner.save_interval=100" \
        "hydra.run.dir=$ROUND_DIR/actor"
    ACTOR="$ROUND_DIR/actor/checkpoint.pt"
done
