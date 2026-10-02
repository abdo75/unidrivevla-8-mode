#!/bin/bash -l
#
# minADE_k / minFDE_k convergence eval for the multi-mode fine-tune.
# -----------------------------------------------------------------------------
# Runs the standard nuScenes-val planning eval with EVAL_RETURN_MODES=1, so the
# model emits all N candidate trajectories and planning_eval.py reports:
#     [Multi-mode]  minADE_k  minFDE_k  mode0_ADE  diversity_gap(mode0-min)
# for ONE checkpoint. Run it per training checkpoint to build the convergence
# curve that answers "is the mode training converged / is training length the
# bottleneck?" -- an in-domain (nuScenes) signal, independent of NAVSIM/scorer.
#   - minADE_k dropping  -> still improving (train longer)
#   - minADE_k flat      -> converged (stop)
#   - mode0_ADE - minADE_k (the gap) -> in-domain diversity/oracle headroom
#
# Usage:  bash scripts/hpc/eval_minade.sh <CKPT>
#   <CKPT> = a training checkpoint's model-states file, e.g.
#     projects/work_dirs/modes8_ft_8gpu/iter_5000/global_step1250/mp_rank_00_model_states.pt
#   (that file loads directly -- no DeepSpeed->.pt conversion needed; dump_modes
#    loads exactly this file.)
# Env overrides:
#   CFG          config with num_modes=8 (default: the 8-mode fine-tune config)
#   GPUS         eval GPUs (default: all visible)
#   NUSC_VERSION trainval = full val split (default) | mini = quick smoke
set -eo pipefail
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
eval "$("$MM" shell hook --shell bash)"; micromamba activate unidrivevla

# Resolve the checkpoint to an ABSOLUTE path from the caller's cwd BEFORE we cd
# into nuScenes below -- otherwise a relative path (e.g. nuScenes/work_dirs/...)
# breaks once we've changed directory.
CKPT_ARG="${1:?usage: eval_minade.sh <CKPT: .../mp_rank_00_model_states.pt>}"
CKPT="$(readlink -f "$CKPT_ARG" 2>/dev/null || echo "$CKPT_ARG")"
START_DIR="$(pwd)"   # capture the caller's dir so the log lands where they ran it

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT/nuScenes"

[ -f "$CKPT" ] || { echo "!! checkpoint not found: $CKPT (from arg: $CKPT_ARG)"; exit 1; }

# checkpoint root probe (same convention as finetune/eval scripts)
CKPT_ROOT="${CKPT_ROOT:-}"
if [ -z "$CKPT_ROOT" ]; then
    for c in "/mnt/shared/$USER/UniDriveVLA/checkpoints" "/mnt/shared/UniDriveVLA/checkpoints" "$HOME/UniDriveVLA/checkpoints" "$PROJECT_ROOT/checkpoints"; do
        [ -d "$c/UniDriveVLA_Nusc_Base_Stage1" ] && { CKPT_ROOT="$c"; break; }
    done
fi
[ -n "$CKPT_ROOT" ] || { echo "!! CKPT_ROOT not found (UniDriveVLA_Nusc_Base_Stage1)"; exit 1; }
export VLM_PRETRAINED_PATH="$CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage1"
export OCCWORLD_VAE_PATH="$CKPT_ROOT/occworld/occvae_latest.pth"
export TORCH_DIST_TIMEOUT=14400
export NUSC_VERSION="${NUSC_VERSION:-trainval}"
export EVAL_RETURN_MODES=1            # <-- makes the model emit all N candidates
# EVAL_MAX_SAMPLES>0 = eval only the first N val samples (fast subset for the curve;
# NUSC_VERSION does NOT shrink val -- the val ann-file is fixed to the trainval pkl).
export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
# Distinct master port so a concurrent eval doesn't clash with a live training run.
export MASTER_PORT="${MASTER_PORT:-29555}"

# Same GCP A3-Ultra gIB-plugin neutralisation as the finetune launcher.
export NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_TUNER_PLUGIN=none
unset NCCL_TUNER_CONFIG_PATH NCCL_NET_GDR_LEVEL

CFG="${CFG:-projects/configs/UniDriveVLA/unidrivevla_stage2_2b_modes8_ft.py}"
GPUS="${GPUS:-$(nvidia-smi -L | wc -l)}"
CKPT_TAG="$(basename "$(dirname "$(dirname "$CKPT")")")"   # e.g. iter_5000
WORK_DIR="projects/work_dirs/eval_minade_${CKPT_TAG}"
RESUME_DIR="$WORK_DIR/resume_state"
rm -rf "$RESUME_DIR"; mkdir -p "$RESUME_DIR"   # fresh each run (no stale partials)

# Run ONLY the planning (minADE_k) eval: disable the detection mAP eval, which loads
# the full-DB GT and asserts pred-tokens==GT-tokens -> "Samples in split doesn't match
# predictions" on any subset. planning_eval is self-contained (doesn't need detection).
# Also cap the inference dataset so a subset eval is fast.
CFG_OPTS=(--cfg-options "evaluation.eval_mode.with_det=False")
[ "$EVAL_MAX_SAMPLES" -gt 0 ] && CFG_OPTS+=("data.test.max_samples=$EVAL_MAX_SAMPLES")

# Timestamped log in the dir the user ran from (same convention as install_*).
LOG="$START_DIR/eval_minade_${CKPT_TAG}_$(date +%Y%m%d_%H%M%S).log"
echo "=== minADE_k eval — ckpt=$CKPT_TAG  CFG=$CFG  GPUS=$GPUS  max_samples=$EVAL_MAX_SAMPLES ==="
echo "    log: $LOG"
bash tools/dist_eval.sh "$CFG" "$CKPT" "$GPUS" --resume-dir "$RESUME_DIR" "${CFG_OPTS[@]}" 2>&1 | tee "$LOG"
echo "=== $CKPT_TAG result (full log: $LOG) ==="
grep -E "\[Multi-mode\]|L2:|Traceback|Error|Exception" "$LOG" | tail -8
