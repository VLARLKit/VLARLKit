#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --account=aip-plbacon

module load gcc opencv
source .venv/bin/activate
wandb offline # since compute nodes cannot access the internet, we need to run in offline mode

NPROC=4
PROJECT_ROOT=$SCRATCH/VLARLKit
CONFIG_NAME="libero_spatial_rlt_pi05"
CONFIG="$PROJECT_ROOT/examples/configs/${CONFIG_NAME}.yaml"
BASE_PORT=5550

# launch environment clients
cd $SCRATCH/VLARLKit/third_party/LIBERO
CLIENT_PIDS=()
for ((i=0; i<NPROC; i++)); do
    CUDA_VISIBLE_DEVICES="$i" PYTHONPATH=$PROJECT_ROOT uv run --no-sync \
        python -m env_clients.client \
        --config "$CONFIG" \
        --host 0.0.0.0 --port $((BASE_PORT + i)) \
        --rank "$i" --world_size "$NPROC" &
    CLIENT_PIDS+=($!)
done

# launch training with torchrun
cd $SCRATCH/VLARLKit
uv run --no-sync torchrun --nproc_per_node="$NPROC" \
    examples/train_offpolicy_rl.py \
    --config-name "$CONFIG_NAME"

# cleanup: kill all client processes
for pid in "${CLIENT_PIDS[@]}"; do
    kill "$pid" 2>/dev/null
done
wait
