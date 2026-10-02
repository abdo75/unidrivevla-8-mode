#!/bin/bash -l
#
# Stage-1 training (SMOKE)
# ------------------------
# What:  Stage-1 perception pretraining (det+map+ego, no motion) — 1-GPU.
#        Confirms the training pipeline launches and captures the diagnostic
#        prints in planning_head.py / vlm_qwenvl3.py ([diag/seq] and
#        [diag/sdpa] lines).
# How:   1 GPU, interactive. Works on both clusters:
#   - iris (V100-32GB): inside an salloc allocation (do NOT sbatch):
#       salloc -p gpu --qos=normal --gres=gpu:1 -C volta32 -t 0-01:00:00 \
#              --mem=100G --cpus-per-task=7
#       bash scripts/hpc/train_stage1_2b_smoke.sh
#   - ubix (Blackwell, 96 GB): GPU always available; just run it — off-SLURM the
#     script sets NUSC_VERSION=mini so the config uses the v1.0-mini split:
#       bash scripts/hpc/train_stage1_2b_smoke.sh
#
# Notes:
#   - On V100 this OOMs at the first SDPA forward (~30 s in, once the diag lines
#     fire) — expected; the diag tells us why. On Blackwell (flash-attn + 96 GB)
#     it should train past iter 0.
#   - VLM weights come from the published Stage-1 HF release (VLM init weights,
#     not a STAGE1_CHECKPOINT). Stage-1 trains from this VLM init.
#   - dist_train.sh exports MASTER_PORT=$MLP_WORKER_0_PORT (clobbering any
#     caller-set MASTER_PORT). Set MLP_WORKER_0_PORT instead.
#   - CFG and EXP_NAME are overridable via env vars.

set -eo pipefail

# micromamba: on PATH (iris) or rootless install (ubix); MAMBA_ROOT_PREFIX is
# inherited from the shell if set, else defaults to the rootless location.
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
eval "$("$MM" shell hook --shell bash)"
micromamba activate unidrivevla

PROJECT_ROOT=$HOME/UniDriveVLA
cd "$PROJECT_ROOT/nuScenes"

export VLM_PRETRAINED_PATH="$PROJECT_ROOT/checkpoints/UniDriveVLA_Nusc_Base_Stage1"
export OCCWORLD_VAE_PATH="$PROJECT_ROOT/checkpoints/occworld/occvae_latest.pth"
export DEEPSPEED_CONFIG="$PROJECT_ROOT/nuScenes/zero_configs/adam_zero1_bf16.json"
GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
export NUM_GPUS=$GPUS
export TORCH_DIST_TIMEOUT=14400
export MLP_WORKER_0_PORT=28598

# Dataset version drives which nuScenes split the (single) config uses — the
# config reads NUSC_VERSION (see unidrivevla_stage1_2b_no_cotraining.py). Default
# to mini off-SLURM (ubix Blackwell) and trainval under SLURM (iris V100).
if [ -z "${NUSC_VERSION:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then NUSC_VERSION=trainval; else NUSC_VERSION=mini; fi
fi
export NUSC_VERSION
CFG="${CFG:-projects/configs/UniDriveVLA/unidrivevla_stage1_2b_no_cotraining.py}"
EXP_NAME="${EXP_NAME:-unidrivevla_stage1_2b_${NUSC_VERSION}}"

echo "============================================"
echo "  Host:        $(hostname)"
echo "  GPUs:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  Config:      $CFG"
echo "  NUSC_VERSION: $NUSC_VERSION"
echo "  Exp name:    $EXP_NAME"
echo "  VLM weights: $VLM_PRETRAINED_PATH"
echo "============================================"

bash tools/dist_train.sh "$CFG" "$GPUS" "$EXP_NAME"
