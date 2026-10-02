#!/bin/bash -l
#
# Stage-2 head-only fine-tune SMOKE (DrivoR thesis, Increment 1).
# ---------------------------------------------------------------
# What: freeze the backbone + perception, train ONLY the action-side planning
#       head for 20 iterations. Validates that the fine-tune harness runs
#       end-to-end (freeze + data + deepspeed + loss + no crash) BEFORE we add
#       the N-mode + winner-take-all change. NOT for results.
# How:  1 GPU on ubix (Blackwell). Mirrors train_stage1_2b_smoke.sh.
# Note: dist_train.sh clobbers MASTER_PORT with MLP_WORKER_0_PORT; set that.
#
set -eo pipefail
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
eval "$("$MM" shell hook --shell bash)"
micromamba activate unidrivevla

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT/nuScenes"

# checkpoints: existence probe (same convention as eval_stage2_2b_smoke.sh)
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
# reduce fragmentation on a shared GPU (does NOT create free VRAM — you still need
# ~20-30 GB free; this smoke OOMs if other users are occupying the card)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Default to v1.0-mini: low system RAM (~1-2 GB annotation load vs 20-40 GB for
# trainval). Needs mini infos in data/infos_mini/ (run gen_mini_infos.sh once).
export NUSC_VERSION="${NUSC_VERSION:-mini}"

CFG="projects/configs/UniDriveVLA/unidrivevla_stage2_2b_ftsmoke.py"
EXP_NAME="${EXP_NAME:-ftsmoke_stage2_${NUSC_VERSION}}"
# dist_train.sh redirects ALL training output here (not to this script's stdout):
TRAIN_LOG="work_dirs/${EXP_NAME}/logs/baseline/train-baseline-all-train.txt"

echo "=== Stage-2 head-only fine-tune SMOKE — $(date -Is) on $(hostname -s) ==="
echo "  CKPT_ROOT=$CKPT_ROOT  NUSC_VERSION=$NUSC_VERSION  GPUS=$NUM_GPUS"
echo "  load_from=$FT_LOAD_FROM"
echo "  TRAINING LOG (watch this): $PROJECT_ROOT/nuScenes/$TRAIN_LOG"
echo "  expect: a [freeze_except=...] line (trainable<<frozen), then ~20 iters of finite loss"
echo "  tail it from another shell:  tail -f $PROJECT_ROOT/nuScenes/$TRAIN_LOG"

# Fresh start: this smoke is throwaway (20 iters, NOT for results). If a previous
# smoke left a checkpoint behind, mmdet_train.py auto-resume (cfg.resume_from filled
# from the latest deepspeed ckpt in work_dir) would (a) reload a *completed* 20-iter
# run -> 0 new iters, useless as a go/no-go, and (b) crash loading the DeepSpeed
# optimizer state under torch-2.7 (torch.load now defaults to weights_only=True,
# which refuses the pickled deepspeed LossScaler). So we wipe the smoke work_dir.
# Guarded to a path containing "smoke" so a mistyped EXP_NAME can't nuke a real run.
WORK_DIR="work_dirs/${EXP_NAME}"
if [ -d "$WORK_DIR" ] && [[ "$WORK_DIR" == *smoke* ]]; then
    echo "  [fresh-start] removing stale smoke work_dir: $PROJECT_ROOT/nuScenes/$WORK_DIR"
    rm -rf "$WORK_DIR"
fi

# --no-validate: skip building the val dataset (saves RAM + time for a smoke).
bash tools/dist_train.sh "$CFG" "$NUM_GPUS" "$EXP_NAME" --no-validate
echo "=== launcher returned; training log: $PROJECT_ROOT/nuScenes/$TRAIN_LOG ==="
