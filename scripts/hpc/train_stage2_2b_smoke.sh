#!/bin/bash -l
#
# Stage-2 training (SMOKE)
# ------------------------
# What:  Stage-2 motion + planning + AR — 1-GPU.
#        Same purpose as train_stage1_2b_smoke.sh: confirm pipeline launches.
# How:   Run INTERACTIVELY inside a 1-GPU V100-32GB allocation. Do NOT sbatch.
#
#   salloc -p gpu --qos=normal --gres=gpu:1 -C volta32 -t 0-02:00:00 \
#          --mem=100G --cpus-per-task=7
#   bash scripts/hpc/train_stage2_2b_smoke.sh
#
# Notes:
#   - Stage-2 *requires* a Stage-1 DeepSpeed-saved checkpoint at
#     STAGE1_CHECKPOINT below. The path currently points at the published
#     HF VLM weights, which are NOT a valid STAGE1_CHECKPOINT (wrong
#     format + wrong state-dict shape). Stage-2 is BLOCKED until we run
#     Stage-1 training successfully and produce mp_rank_00_model_states.pt.
#   - dist_train.sh clobbers MASTER_PORT; set MLP_WORKER_0_PORT instead.

set -eo pipefail

eval "$(micromamba shell hook --shell bash)"
micromamba activate unidrivevla

PROJECT_ROOT=$HOME/UniDriveVLA
cd "$PROJECT_ROOT/nuScenes"

export VLM_PRETRAINED_PATH="$PROJECT_ROOT/checkpoints/UniDriveVLA_Nusc_Base_Stage1"
export OCCWORLD_VAE_PATH="$PROJECT_ROOT/checkpoints/occworld/occvae_latest.pth"
# TODO: replace with real Stage-1 deepspeed ckpt once Stage-1 training has run.
export STAGE1_CHECKPOINT="$PROJECT_ROOT/checkpoints/UniDriveVLA_Nusc_Base_Stage1/UniDriveVLA_Stage1_Nuscenes_2B.pt"
export DEEPSPEED_CONFIG="$PROJECT_ROOT/nuScenes/zero_configs/adam_zero1_bf16.json"
export NUM_GPUS=1
export TORCH_DIST_TIMEOUT=14400
export MLP_WORKER_0_PORT=28597

CFG=projects/configs/UniDriveVLA/unidrivevla_stage2_2b.py
EXP_NAME=unidrivevla_stage2_2b_smoke
GPUS=1

echo "============================================"
echo "  Host:        $(hostname)"
echo "  GPUs:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  Config:      $CFG"
echo "  Exp name:    $EXP_NAME"
echo "  VLM weights: $VLM_PRETRAINED_PATH"
echo "  Stage1 ckpt: $STAGE1_CHECKPOINT"
echo "============================================"

bash tools/dist_train.sh "$CFG" "$GPUS" "$EXP_NAME"
