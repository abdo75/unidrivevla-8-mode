#!/bin/bash -l
#
# BEV candidate-trajectory visualisation for the multi-mode planner.
# -----------------------------------------------------------------------------
# Runs render_candidates.py: for a handful of nuScenes-val samples it draws a
# BEV of the N candidate ego trajectories with the DEPLOYED (mode 0) and the
# ORACLE-BEST candidate highlighted in distinct colours, plus the GT future.
# Produces one PNG per sample (+ candidates.mp4 with --video) for paper figures.
#
# Usage:  bash scripts/hpc/viz_candidates.sh <CKPT> [extra render_candidates.py args]
#   <CKPT> = a training checkpoint's model-states file, e.g.
#     projects/work_dirs/modes8_ft_8gpu/iter_5000/global_step1250/mp_rank_00_model_states.pt
# Env overrides:
#   CFG        config with num_modes=8 (default: the 8-mode fine-tune config)
#   OUT_DIR    where PNGs/mp4 land (default: viz_candidates in the caller's dir)
#   NUM        samples to render (default 40)   START  first sample idx (default 0)
#   VIDEO      stitch the PNGs into candidates.mp4 (default 1; set 0 to skip)  FPS (default 2)
#   AGENT_MOTION=1  also draw other agents' GT future paths (more context, more clutter)
# Env mirrors eval_minade.sh (checkpoint-root probe, VLM/OccVAE paths, NCCL fix).
set -eo pipefail
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
eval "$("$MM" shell hook --shell bash)"; micromamba activate unidrivevla

CKPT_ARG="${1:?usage: viz_candidates.sh <CKPT: .../mp_rank_00_model_states.pt> [args]}"
shift || true
CKPT="$(readlink -f "$CKPT_ARG" 2>/dev/null || echo "$CKPT_ARG")"
START_DIR="$(pwd)"
OUT_DIR="${OUT_DIR:-$START_DIR/viz_candidates}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT/nuScenes"
[ -f "$CKPT" ] || { echo "!! checkpoint not found: $CKPT (from arg: $CKPT_ARG)"; exit 1; }

# checkpoint-root probe (same convention as eval_minade.sh / finetune scripts)
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

# GCP A3-Ultra gIB-plugin neutralisation (harmless off-GCP); single-GPU render.
export NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_TUNER_PLUGIN=none
unset NCCL_TUNER_CONFIG_PATH NCCL_NET_GDR_LEVEL
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CFG="${CFG:-projects/configs/UniDriveVLA/unidrivevla_stage2_2b_modes8_ft.py}"
NUM="${NUM:-40}"; START="${START:-0}"; FPS="${FPS:-2}"
[ "${VIDEO:-1}" = "1" ] && VID_FLAG="--video" || VID_FLAG=""
[ "${AGENT_MOTION:-0}" = "1" ] && MOTION_FLAG="--agent-motion" || MOTION_FLAG=""

LOG="$START_DIR/viz_candidates_$(date +%Y%m%d_%H%M%S).log"
echo "=== BEV candidate viz — ckpt=$CKPT  CFG=$CFG  out=$OUT_DIR  num=$NUM start=$START video=${VIDEO:-1} ==="
echo "    log: $LOG"
python tools/visualization/render_candidates.py "$CFG" "$CKPT" \
    --out-dir "$OUT_DIR" --num-samples "$NUM" --start "$START" --fps "$FPS" \
    $VID_FLAG $MOTION_FLAG "$@" 2>&1 | tee "$LOG"
echo "=== done -> $OUT_DIR  (log: $LOG) ==="