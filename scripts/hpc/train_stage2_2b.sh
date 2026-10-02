#!/bin/bash -l
#
# Stage-2 training (PROD)
# -----------------------
# What:  Stage-2 motion + planning + AR — 4 GPUs, long walltime.
#        Loads STAGE1_CHECKPOINT and finetunes the full driving stack.
# Where: 4 x V100-32GB (-C volta32), 4-day wall (rough — adjust after first run).
# Notes:
#   - BLOCKED until Stage-1 training produces a real DeepSpeed checkpoint at
#     STAGE1_CHECKPOINT. The published HF Stage-1 release is *VLM weights*,
#     not a STAGE1_CHECKPOINT.
#   - Currently constrained by the same SDPA-on-V100 limitation as Stage-1.
#
#SBATCH -J unidrivevla_train_s2
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
#SBATCH -o slurm-train-s2-%j.out

set -eo pipefail

eval "$(micromamba shell hook --shell bash)"
micromamba activate unidrivevla

PROJECT_ROOT=$HOME/UniDriveVLA
cd "$PROJECT_ROOT/nuScenes"

export VLM_PRETRAINED_PATH="$PROJECT_ROOT/checkpoints/UniDriveVLA_Nusc_Base_Stage1"
export OCCWORLD_VAE_PATH="$PROJECT_ROOT/checkpoints/occworld/occvae_latest.pth"
# TODO: point at the real Stage-1 deepspeed checkpoint produced by train_stage1_2b.sh:
#   work_dirs/unidrivevla_stage1_2b/iter_XXXX/global_stepXXXX/mp_rank_00_model_states.pt
export STAGE1_CHECKPOINT="$PROJECT_ROOT/nuScenes/projects/work_dirs/unidrivevla_stage1_2b/iter_XXXX/global_stepXXXX/mp_rank_00_model_states.pt"
export DEEPSPEED_CONFIG="$PROJECT_ROOT/nuScenes/zero_configs/adam_zero1_bf16.json"
export NUM_GPUS=4
export TORCH_DIST_TIMEOUT=14400
export MLP_WORKER_0_PORT=28599

CFG=projects/configs/UniDriveVLA/unidrivevla_stage2_2b.py
EXP_NAME=unidrivevla_stage2_2b
GPUS=4

if [ ! -f "$STAGE1_CHECKPOINT" ]; then
  echo "ERROR: STAGE1_CHECKPOINT does not exist: $STAGE1_CHECKPOINT" >&2
  echo "       Run train_stage1_2b.sh first and update the path above." >&2
  exit 1
fi

echo "============================================"
echo "  Host:        $(hostname)"
echo "  GPUs:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  Config:      $CFG"
echo "  Exp name:    $EXP_NAME"
echo "  VLM weights: $VLM_PRETRAINED_PATH"
echo "  Stage1 ckpt: $STAGE1_CHECKPOINT"
echo "============================================"

bash tools/dist_train.sh "$CFG" "$GPUS" "$EXP_NAME"
