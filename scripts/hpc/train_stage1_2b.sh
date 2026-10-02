#!/bin/bash -l
#
# Stage-1 training (PROD)
# -----------------------
# What:  Stage-1 perception pretraining (det+map+ego, no motion).
#        Trains 30 epochs on nuScenes trainval at batch_size=4/GPU.
# Where: 4 x V100-32GB (-C volta32), 4-day wall (rough — adjust after first run).
# Notes:
#   - Output: work_dirs/unidrivevla_stage1_2b/iter_XXXX/global_stepXXXX/
#     mp_rank_00_model_states.pt
#     That .pt is what train_stage2_2b.sh consumes as STAGE1_CHECKPOINT.
#   - Will OOM on V100 with current sequence lengths. Blocked until we move
#     to A100+ or enable knowledge insulation (training-objective change).
#     This script is left ready for the moment we have suitable hardware.
#
#SBATCH -J unidrivevla_train_s1
#SBATCH --mail-type=start,end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
#SBATCH -p gpu
#SBATCH -C volta32
#SBATCH --qos=normal
#SBATCH -o slurm-train-s1-%j.out

set -eo pipefail

eval "$(micromamba shell hook --shell bash)"
micromamba activate unidrivevla

PROJECT_ROOT=$HOME/UniDriveVLA
cd "$PROJECT_ROOT/nuScenes"

export VLM_PRETRAINED_PATH="$PROJECT_ROOT/checkpoints/UniDriveVLA_Nusc_Base_Stage1"
export OCCWORLD_VAE_PATH="$PROJECT_ROOT/checkpoints/occworld/occvae_latest.pth"
export DEEPSPEED_CONFIG="$PROJECT_ROOT/nuScenes/zero_configs/adam_zero1_bf16.json"
export NUM_GPUS=4
export TORCH_DIST_TIMEOUT=14400
export MLP_WORKER_0_PORT=28599

CFG=projects/configs/UniDriveVLA/unidrivevla_stage1_2b_no_cotraining.py
EXP_NAME=unidrivevla_stage1_2b
GPUS=4

echo "============================================"
echo "  Host:        $(hostname)"
echo "  GPUs:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  Config:      $CFG"
echo "  Exp name:    $EXP_NAME"
echo "  VLM weights: $VLM_PRETRAINED_PATH"
echo "============================================"

bash tools/dist_train.sh "$CFG" "$GPUS" "$EXP_NAME"
