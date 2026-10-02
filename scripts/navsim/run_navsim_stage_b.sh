#!/bin/bash
#
# Stage B runner (unidrivevla env) — run UniDriveVLA on the Stage-A dumps.
# Full run, zero flags. Smoke (a few tokens) = run_navsim_stage_b_smoke.sh.
# Pre: Stage A produced .npz under $NAVSIM_WS/exp/uni_inputs/<split>.
#
set -eo pipefail
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$(dirname "${BASH_SOURCE[0]}")/navsim_paths.sh" 2>/dev/null || true
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"
SPLIT="${TRAIN_TEST_SPLIT:-warmup_two_stage}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# checkpoints: same existence-probe convention as the eval script
CKPT_ROOT="${CKPT_ROOT:-}"
if [ -z "$CKPT_ROOT" ]; then
    for c in "/mnt/shared/$USER/UniDriveVLA/checkpoints" "/mnt/shared/UniDriveVLA/checkpoints" "$HOME/UniDriveVLA/checkpoints" "$PROJECT_ROOT/checkpoints"; do
        [ -d "$c/UniDriveVLA_Nusc_Base_Stage1" ] && { CKPT_ROOT="$c"; break; }
    done
fi
[ -n "$CKPT_ROOT" ] || { echo "!! CKPT_ROOT not found (UniDriveVLA_Nusc_Base_Stage1)"; exit 1; }

export VLM_PRETRAINED_PATH="$CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage1"
export OCCWORLD_VAE_PATH="$CKPT_ROOT/occworld/occvae_latest.pth"
export NUSC_VERSION="${NUSC_VERSION:-trainval}"
# Overridable so a wrapper can point at a multi-mode config + checkpoint (Increment 2).
export CONFIG="${CONFIG:-$PROJECT_ROOT/nuScenes/projects/configs/UniDriveVLA/unidrivevla_stage2_2b.py}"
export CKPT="${CKPT:-$CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage2/UniDriveVLA_Stage2_Nuscenes_2B.pt}"
export STAGE_A_IN="${STAGE_A_IN:-$NAVSIM_WS/exp/uni_inputs/$SPLIT}"
export STAGE_B_OUT="${STAGE_B_OUT:-$NAVSIM_WS/exp/uni_trajectories/$SPLIT}"

LOG="./navsim_stage_b_${SPLIT}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "=== Stage B ($SPLIT, LIMIT=${LIMIT:-0}) — $(date -Is) ==="
echo "  CKPT_ROOT=$CKPT_ROOT"
echo "  in=$STAGE_A_IN  out=$STAGE_B_OUT"

eval "$("$MM" shell hook --shell bash)"
micromamba activate "${UNI_ENV:-unidrivevla}"
cd "$PROJECT_ROOT/nuScenes"   # so `import projects.mmdet3d_plugin` resolves
export PYTHONPATH="$PROJECT_ROOT/nuScenes:${PYTHONPATH:-}"
python "$HERE/stage_b_run_uni.py"
echo "=== DONE — log: $LOG ==="
