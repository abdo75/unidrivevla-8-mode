#!/bin/bash
#
# Learned-scorer navhard evaluation (Increment 3, fills the thesis table row). Chains:
#   1. ensure navhard phi_s + candidates are dumped (Stage B w/ DUMP_SCENE_TOKENS)
#   2. (unidrivevla env) scorer picks one candidate per token -> selection.json
#   3. (navsim env) build a NAVSIM submission of the selected trajectories
#   4. (navsim env) score that submission with the real EPDMS metric
# Zero flags; override SCORER to point at a different trained scorer.pt.
#
# Pre: a trained scorer.pt (run_train_scorer.sh) and the navhard metric cache.
#
set -eo pipefail
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$(dirname "${BASH_SOURCE[0]}")/../navsim/navsim_paths.sh" 2>/dev/null || true
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"
SPLIT="${TRAIN_TEST_SPLIT:-navhard_two_stage}"
ENV_FILE="$NAVSIM_WS/navsim_env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCORER="${SCORER:-$NAVSIM_WS/exp/uni_scorer/scorer.pt}"
STAGE_B_DIR="${STAGE_B_DIR:-$NAVSIM_WS/exp/uni_trajectories/${SPLIT}_scorer}"
SEL="${SEL:-$NAVSIM_WS/exp/uni_scorer/selection_${SPLIT}.json}"
SUB="${SUB:-$NAVSIM_WS/exp/uni_scorer/submission_${SPLIT}_scorer.pkl}"

LOG="./scorer_navhard_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "=== learned-scorer navhard eval ($SPLIT) — $(date -Is) ==="
echo "  scorer=$SCORER  stage_b=$STAGE_B_DIR"

[ -f "$SCORER" ] || { echo "!! scorer not found: $SCORER (run run_train_scorer.sh first)"; exit 1; }
[ -f "$ENV_FILE" ] || { echo "!! $ENV_FILE not found — run setup_navsim_env.sh first"; exit 1; }

# 1) phi_s + candidates for navhard (skip if already dumped)
if [ ! -d "$STAGE_B_DIR/phi_s" ] || [ ! -f "$STAGE_B_DIR/trajectories.pkl" ]; then
    echo "-- dumping navhard phi_s + candidates (Stage B)"
    TRAIN_TEST_SPLIT="$SPLIT" STAGE_B_OUT="$STAGE_B_DIR" \
        bash "$PROJECT_ROOT/scripts/navsim/run_navsim_stage_b_scorer.sh"
else
    echo "-- reusing existing dump at $STAGE_B_DIR"
fi

# 2) scorer selection (unidrivevla env, torch)
echo "-- selecting candidates with the learned scorer"
mkdir -p "$(dirname "$SEL")"
( cd "$HERE" && "$MM" run -n "${UNI_ENV:-unidrivevla}" python select_candidates.py "$SCORER" "$STAGE_B_DIR" "$SEL" )

# 3) + 4) build the selected submission and score it (navsim env)
# shellcheck source=/dev/null
source "$ENV_FILE"
eval "$("$MM" shell hook --shell bash)"
micromamba activate "${NAVSIM_ENV:-navsim}"
export TRAIN_TEST_SPLIT="$SPLIT"
TS="$OPENSCENE_DATA_ROOT/$SPLIT"   # v2.2: split is top-level, not nested
CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache/$SPLIT}"
EXP="uni_scorer_${SPLIT}"

echo "-- building selected submission"
SELECTION="$SEL" STAGE_B_TRAJ="$STAGE_B_DIR/trajectories.pkl" SUBMISSION_OUT="$SUB" \
    python "$PROJECT_ROOT/scripts/navsim/stage_c_build_submission.py"

echo "-- scoring selected submission (EPDMS)"
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_from_submission.py" \
    train_test_split="$SPLIT" \
    experiment_name="$EXP" \
    submission_file_path="$SUB" \
    metric_cache_path="$CACHE_PATH" \
    synthetic_sensor_path="$TS/sensor_blobs" \
    synthetic_scenes_path="$TS/synthetic_scene_pickles"

CSV=$(find "$NAVSIM_EXP_ROOT/$EXP" -name "*.csv" -printf '%T+ %p\n' 2>/dev/null | sort | tail -1 | cut -d' ' -f2-)
echo "  scorer-selected score CSV: $CSV"
python - "$CSV" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
df = df[df["token"] != "average"]
s = df["score"] if "score" in df.columns else df["score_stage_one"].combine_first(df["score_stage_two"])
print(f"=== LEARNED-SCORER EPDMS on {len(s)} tokens: {s.mean():.4f} ===")
PY
echo "=== DONE — log: $LOG ==="