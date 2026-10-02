#!/bin/bash
#
# Stage C runner (navsim env) — convert UniDriveVLA trajectories -> NavSim
# submission pickle, then score it (run_pdm_score_from_submission) = first
# UniDriveVLA PDMS on NavSim. Zero flags. Pre: Stage B FULL produced
# trajectories.pkl (all tokens), and the metric cache exists (run_navsim_metric_cache.sh).
#
set -eo pipefail
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

source "$(dirname "${BASH_SOURCE[0]}")/navsim_paths.sh" 2>/dev/null || true
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"
ENV_NAME="${NAVSIM_ENV:-navsim}"
ENV_FILE="$NAVSIM_WS/navsim_env.sh"
SPLIT="${TRAIN_TEST_SPLIT:-warmup_two_stage}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG="./navsim_stage_c_${SPLIT}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "=== Stage C ($SPLIT) — $(date -Is) ==="

[ -f "$ENV_FILE" ] || { echo "!! $ENV_FILE not found — run setup_navsim_env.sh first"; exit 1; }
# shellcheck source=/dev/null
source "$ENV_FILE"
eval "$("$MM" shell hook --shell bash)"
micromamba activate "$ENV_NAME"

export TRAIN_TEST_SPLIT="$SPLIT"
TS="$OPENSCENE_DATA_ROOT/$SPLIT"   # v2.2: split is top-level, not nested
export STAGE_B_TRAJ="${STAGE_B_TRAJ:-$NAVSIM_WS/exp/uni_trajectories/$SPLIT/trajectories.pkl}"
export SUBMISSION_OUT="${SUBMISSION_OUT:-$NAVSIM_WS/exp/uni_submission/$SPLIT/submission.pkl}"
CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache/$SPLIT}"

[ -f "$STAGE_B_TRAJ" ] || { echo "!! $STAGE_B_TRAJ not found — run Stage B (full) first"; exit 1; }
[ -d "$CACHE_PATH/metadata" ] || { echo "!! metric cache missing at $CACHE_PATH — run run_navsim_metric_cache.sh"; exit 1; }

echo "--- build submission ---"
python "$HERE/stage_c_build_submission.py"

echo "--- score submission (PDMS) ---"
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_from_submission.py" \
    train_test_split="$SPLIT" \
    experiment_name="uni_${SPLIT}" \
    submission_file_path="$SUBMISSION_OUT" \
    metric_cache_path="$CACHE_PATH" \
    synthetic_sensor_path="$TS/sensor_blobs" \
    synthetic_scenes_path="$TS/synthetic_scene_pickles"

echo "=== DONE — log: $LOG ==="
