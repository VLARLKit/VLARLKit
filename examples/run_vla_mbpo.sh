#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH --account=aip-plbacon

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH}/VLARLKit}"
SCRIPT_PATH="${PROJECT_ROOT}/examples/run_vla_mbpo.sh"

NPROC=4
CONFIG_NAME="libero_goal_vla_mbpo"
CONFIG="${PROJECT_ROOT}/examples/configs/${CONFIG_NAME}.yaml"

WM_BASE_PORT=8002
ENV_BASE_PORT=5550
LOAD_MODEL_PATH="/scratch/s/sunyh/Bagel-WM/Bagel-libero-goal"
BAGEL_PYTHON="${PROJECT_ROOT}/third_party/BAGEL/.venv/bin/python"

wait_port() {
    local host=$1
    local port=$2
    local retry=${3:-120}
    local interval=${4:-1}
    local i
    for ((i=1; i<=retry; i++)); do
        if timeout 1 bash -lc "echo > /dev/tcp/${host}/${port}" 2>/dev/null; then
            echo "Port ${port} on ${host} is ready."
            return 0
        fi
        sleep "${interval}"
    done
    echo "Port ${port} on ${host} not ready after $((retry * interval))s."
    return 1
}

run_wm_node() {
    module load gcc opencv
    cd "${PROJECT_ROOT}"
    export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/BAGEL:${PYTHONPATH:-}"

    WM_PIDS=()
    cleanup() {
        for pid in "${WM_PIDS[@]}"; do
            pkill -P "$pid" 2>/dev/null || true
            kill "$pid" 2>/dev/null || true
        done
        wait 2>/dev/null || true
    }
    trap cleanup EXIT INT TERM

    echo "Starting BAGEL world-model clients on $(hostname) (gpus 0-$((NPROC - 1)))"
    for ((i=0; i<NPROC; i++)); do
        CUDA_VISIBLE_DEVICES="$i" "${BAGEL_PYTHON}" \
            -m env_clients.world_models.bagel.client \
            --load-model-path "${LOAD_MODEL_PATH}" \
            --host 0.0.0.0 --port $((WM_BASE_PORT + i)) &
        WM_PIDS+=($!)
    done

    wait
}

run_train_node() {
    module load gcc opencv
    cd "${PROJECT_ROOT}"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    wandb offline

    ENV_PIDS=()
    cleanup() {
        for pid in "${ENV_PIDS[@]}"; do
            pkill -P "$pid" 2>/dev/null || true
            kill "$pid" 2>/dev/null || true
        done
        wait 2>/dev/null || true
    }
    trap cleanup EXIT INT TERM

    echo "Starting LIBERO env clients on $(hostname) (gpus 0-$((NPROC - 1)))"
    cd "${PROJECT_ROOT}/third_party/LIBERO"
    for ((i=0; i<NPROC; i++)); do
        CUDA_VISIBLE_DEVICES="$i" PYTHONPATH="${PROJECT_ROOT}" uv run --no-sync \
            python -m env_clients.client \
            --config "${CONFIG}" \
            --host 0.0.0.0 --port $((ENV_BASE_PORT + i)) \
            --rank "$i" --world_size "$NPROC" \
            --modes eval &
        ENV_PIDS+=($!)
    done

    for ((i=0; i<NPROC; i++)); do
        wait_port "127.0.0.1" $((ENV_BASE_PORT + i)) || exit 1
    done

    cd "${PROJECT_ROOT}"
    echo "Starting VLA-MBPO training (world model at ${WM_NODE})"
    uv run --project model_backends/openpi --no-sync torchrun --nproc_per_node="${NPROC}" \
        examples/train_vla_mbpo.py \
        --config-name "${CONFIG_NAME}" \
        world_model.host="${WM_NODE}" \
        world_model.base_port="${WM_BASE_PORT}" \
        world_model.load_model_path="${LOAD_MODEL_PATH}" \
        env.env_client_host="localhost" \
        env.env_client_base_port="${ENV_BASE_PORT}"
    cleanup
}

run_launch() {
    if [[ -z "${SLURM_NODELIST:-}" ]]; then
        echo "Error: submit with sbatch (needs SLURM_NODELIST)."
        exit 1
    fi

    HOSTS=($(scontrol show hostnames "$SLURM_NODELIST"))
    WM_NODE="${HOSTS[0]}"
    TRAIN_NODE="${HOSTS[1]}"
    export WM_NODE TRAIN_NODE

    echo "WM node: ${WM_NODE}, train node: ${TRAIN_NODE}"

    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="${WM_NODE}" \
        env RUN_ROLE=wm PROJECT_ROOT="${PROJECT_ROOT}" \
        WM_NODE="${WM_NODE}" TRAIN_NODE="${TRAIN_NODE}" \
        bash "${SCRIPT_PATH}" &
    WM_PID=$!

    stop_wm() {
        kill -TERM "${WM_PID}" 2>/dev/null || true
        wait "${WM_PID}" 2>/dev/null || true
    }
    trap stop_wm EXIT INT TERM

    for ((i=0; i<NPROC; i++)); do
        wait_port "${WM_NODE}" $((WM_BASE_PORT + i)) || exit 1
    done

    TRAIN_STATUS=0
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="${TRAIN_NODE}" \
        env RUN_ROLE=train PROJECT_ROOT="${PROJECT_ROOT}" \
        WM_NODE="${WM_NODE}" TRAIN_NODE="${TRAIN_NODE}" \
        bash "${SCRIPT_PATH}" || TRAIN_STATUS=$?

    stop_wm
    trap - EXIT INT TERM

    exit "${TRAIN_STATUS}"
}

[[ "${RUN_ROLE:-}" == "wm" ]] && { run_wm_node; exit 0; }
[[ "${RUN_ROLE:-}" == "train" ]] && { run_train_node; exit 0; }
run_launch