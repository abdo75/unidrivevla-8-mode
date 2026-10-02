#!/bin/bash -l
#
# Increment 2 diagnostic: dump the N candidate trajectories from the overfit
# checkpoint and report per-mode spread (behaviorally diverse vs degenerate).
# Runs forward_test(return_all_modes=True) on a few mini-val samples. No training.
#
set -eo pipefail
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
eval "$("$MM" shell hook --shell bash)"
micromamba activate unidrivevla

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT/nuScenes"

CKPT_ROOT="${CKPT_ROOT:-}"
if [ -z "$CKPT_ROOT" ]; then
    for c in "/mnt/shared/$USER/UniDriveVLA/checkpoints" "/mnt/shared/UniDriveVLA/checkpoints" "$HOME/UniDriveVLA/checkpoints" "$PROJECT_ROOT/checkpoints"; do
        [ -d "$c/UniDriveVLA_Nusc_Base_Stage1" ] && { CKPT_ROOT="$c"; break; }
    done
fi
[ -n "$CKPT_ROOT" ] || { echo "!! CKPT_ROOT not found (UniDriveVLA_Nusc_Base_Stage1)"; exit 1; }

export VLM_PRETRAINED_PATH="$CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage1"
export OCCWORLD_VAE_PATH="$CKPT_ROOT/occworld/occvae_latest.pth"
# FT_LOAD_FROM is referenced by the config but the weights actually come from the
# overfit checkpoint below; point it at the released .pt so config build succeeds.
export FT_LOAD_FROM="$CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage2/UniDriveVLA_Stage2_Nuscenes_2B.pt"
export NUSC_VERSION="${NUSC_VERSION:-mini}"

CFG="projects/configs/UniDriveVLA/unidrivevla_stage2_2b_overfit_modes.py"
CKPT="${OVERFIT_CKPT:-work_dirs/overfit_modes_mini/iter_150}"
[ -e "$CKPT" ] || { echo "!! overfit checkpoint not found at $CKPT (set OVERFIT_CKPT=...)"; exit 1; }

# PYTHONPATH so `projects...` imports resolve (dump_modes.py also self-inserts it).
export PYTHONPATH="$PROJECT_ROOT/nuScenes:$PYTHONPATH"

LOG="$PROJECT_ROOT/nuScenes/work_dirs/overfit_modes_mini/dump_modes_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
echo "=== dump modes — $(date -Is) on $(hostname -s) ==="
echo "  config=$CFG  ckpt=$CKPT"
echo "  logging to: $LOG"
python tools/dump_modes.py "$CFG" "$CKPT" \
    --num-samples "${NUM_SAMPLES:-4}" \
    --out "work_dirs/overfit_modes_mini/mode_candidates.pkl" 2>&1 | tee "$LOG"
echo "=== done; log: $LOG ==="