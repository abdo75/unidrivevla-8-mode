#!/bin/bash
#
# Conversion audit (truthfulness check): is our low navhard EPDMS real, or a Stage-C
# coordinate bug? Re-score the SAME cand0 trajectories under both lateral-sign conventions
# and report the official two-stage "Final extended pdm score" for each.
#   LATERAL_SIGN=+1  -> our reported baseline (cand0 = 0.1212)
#   LATERAL_SIGN=-1  -> if turns were sign-flipped, trajectories veer off the drivable
#                       area; flipping should jump the score. If both stay ~0.12, the sign
#                       is correct and the low drivable-area compliance is genuine.
# Pure reuse of stage_c_build_submission.py + run_pdm_score_from_submission (no new logic).
# navsim env; GPU-free (~45 min per sign). tmux.
#
set -eo pipefail
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$(dirname "${BASH_SOURCE[0]}")/../navsim/navsim_paths.sh" 2>/dev/null || true
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"
SPLIT="${TRAIN_TEST_SPLIT:-navhard_two_stage}"
ENV_FILE="$NAVSIM_WS/navsim_env.sh"
STAGE_B_TRAJ="${STAGE_B_TRAJ:-$NAVSIM_WS/exp/uni_trajectories/${SPLIT}_scorer/trajectories.pkl}"
LOG="./audit_conversion_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "=== conversion audit ($SPLIT) — $(date -Is) ==="
echo "  trajectories=$STAGE_B_TRAJ"

# shellcheck source=/dev/null
source "$ENV_FILE"
eval "$("$MM" shell hook --shell bash)"
micromamba activate "${NAVSIM_ENV:-navsim}"
export TRAIN_TEST_SPLIT="$SPLIT"
TS="$OPENSCENE_DATA_ROOT/$SPLIT"   # v2.2: split is top-level, not nested
CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache/$SPLIT}"

for SIGN in 1 -1; do
    tag="sign${SIGN}"
    SUB="$NAVSIM_WS/exp/uni_scorer/audit_cand0_${tag}.pkl"
    EXP="uni_audit_${SPLIT}_${tag//-/m}"
    echo; echo "##### LATERAL_SIGN=$SIGN #####"
    CAND_IDX=0 LATERAL_SIGN="$SIGN" STAGE_B_TRAJ="$STAGE_B_TRAJ" SUBMISSION_OUT="$SUB" \
        python "$PROJECT_ROOT/scripts/navsim/stage_c_build_submission.py"
    python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_from_submission.py" \
        train_test_split="$SPLIT" experiment_name="$EXP" submission_file_path="$SUB" \
        metric_cache_path="$CACHE_PATH" \
        synthetic_sensor_path="$TS/sensor_blobs" \
        synthetic_scenes_path="$TS/synthetic_scene_pickles" 2>&1 \
      | grep -iE "Final extended pdm score|successful scenarios|failed scenarios" || true
done
echo; echo "=== DONE — compare the two 'Final extended pdm score' lines. log: $LOG ==="
