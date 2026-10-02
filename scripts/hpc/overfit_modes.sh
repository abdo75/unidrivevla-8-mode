#!/bin/bash -l
#
# Increment 2 — OVERFIT divergence test (DrivoR thesis). NOT for results.
# ----------------------------------------------------------------------
# What: overfit a FIXED 4 mini samples for 150 iters with strong mode levers
#       (mode_init_std=0.2, mode_query lr_mult=50x, pure WTA eps=0). Validates
#       that the N modes actually SPECIALIZE — the 20-iter smoke proved the
#       plumbing but ran too few distinct-sample steps to show divergence.
# How:  1 GPU on ubix (Blackwell). Mirrors finetune_stage2_smoke.sh.
# Watch in the train log:
#   planning.mode_endpoint_std            -> should climb above ~0.1 m
#   planning.mode_mean_all - mode_winner_min -> per-sample loss spread; should grow.
#   Both growing => mechanism PROVEN. Both flat even here => mechanism too weak.
#
set -eo pipefail
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
eval "$("$MM" shell hook --shell bash)"
micromamba activate unidrivevla

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT/nuScenes"

# checkpoints: existence probe (same convention as finetune_stage2_smoke.sh)
CKPT_ROOT="${CKPT_ROOT:-}"
if [ -z "$CKPT_ROOT" ]; then
    for c in "/mnt/shared/$USER/UniDriveVLA/checkpoints" "/mnt/shared/UniDriveVLA/checkpoints" "$HOME/UniDriveVLA/checkpoints" "$PROJECT_ROOT/checkpoints"; do
        [ -d "$c/UniDriveVLA_Nusc_Base_Stage1" ] && { CKPT_ROOT="$c"; break; }
    done
fi
[ -n "$CKPT_ROOT" ] || { echo "!! CKPT_ROOT not found (UniDriveVLA_Nusc_Base_Stage1)"; exit 1; }

export VLM_PRETRAINED_PATH="$CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage1"
export OCCWORLD_VAE_PATH="$CKPT_ROOT/occworld/occvae_latest.pth"
export DEEPSPEED_CONFIG="$PROJECT_ROOT/nuScenes/zero_configs/adam_zero1_bf16.json"
export FT_LOAD_FROM="$CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage2/UniDriveVLA_Stage2_Nuscenes_2B.pt"
export NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
export TORCH_DIST_TIMEOUT=14400
export MLP_WORKER_0_PORT="${MLP_WORKER_0_PORT:-28599}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NUSC_VERSION="${NUSC_VERSION:-mini}"

CFG="projects/configs/UniDriveVLA/unidrivevla_stage2_2b_overfit_modes.py"
EXP_NAME="${EXP_NAME:-overfit_modes_${NUSC_VERSION}}"
TRAIN_LOG="work_dirs/${EXP_NAME}/logs/baseline/train-baseline-all-train.txt"

echo "=== Increment 2 OVERFIT divergence test — $(date -Is) on $(hostname -s) ==="
echo "  CKPT_ROOT=$CKPT_ROOT  NUSC_VERSION=$NUSC_VERSION  GPUS=$NUM_GPUS"
echo "  load_from=$FT_LOAD_FROM"
echo "  TRAINING LOG (watch this): $PROJECT_ROOT/nuScenes/$TRAIN_LOG"
echo "  expect: 150 iters; watch mode_endpoint_std climb and (mode_mean_all - mode_winner_min) grow"
echo "  tail it from another shell:  tail -f $PROJECT_ROOT/nuScenes/$TRAIN_LOG"

# Fresh start: throwaway overfit run. Avoid mmdet_train.py auto-resume reloading a
# stale deepspeed ckpt (useless + crashes under torch-2.7 weights_only). Guarded to
# a path containing "overfit" so a mistyped EXP_NAME can't nuke a real run.
WORK_DIR="work_dirs/${EXP_NAME}"
if [ -d "$WORK_DIR" ] && [[ "$WORK_DIR" == *overfit* ]]; then
    echo "  [fresh-start] removing stale overfit work_dir: $PROJECT_ROOT/nuScenes/$WORK_DIR"
    rm -rf "$WORK_DIR"
fi

# --no-validate: skip building the val dataset (saves RAM + time).
bash tools/dist_train.sh "$CFG" "$NUM_GPUS" "$EXP_NAME" --no-validate
echo "=== launcher returned; training log: $PROJECT_ROOT/nuScenes/$TRAIN_LOG ==="