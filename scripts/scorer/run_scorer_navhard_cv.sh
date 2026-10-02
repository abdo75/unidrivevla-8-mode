#!/bin/bash
#
# Learned-scorer navhard HEADLINE via k-fold cross-validation (Increment 3). Chains:
#   1. ensure navhard phi_s + candidates are dumped (Stage B w/ DUMP_SCENE_TOKENS)
#   2. (unidrivevla env) build dataset from phi_s + the existing oracle sub-score CSVs
#   3. (unidrivevla env) k-fold CV: each scene selected by a scorer that never saw it
#      -> pooled out-of-fold selection.json (+ printed CV recovery summary)
#   4. (navsim env) build a NAVSIM submission of the selected trajectories
#   5. (navsim env) score that submission with the real EPDMS -> the table number
# Zero flags. K / EPOCHS overridable. Hyperparameters are frozen from the warmup
# ablations; navhard is only ever evaluated here, never tuned on.
#
# Pre: navhard oracle labels exist (uni_oracle_navhard_two_stage*_cand*), metric cache.
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
K="${K:-5}"
EPOCHS="${EPOCHS:-200}"

NAVSIM_EXP_ROOT_DEFAULT="$NAVSIM_WS/exp"
STAGE_B_DIR="${STAGE_B_DIR:-$NAVSIM_WS/exp/uni_trajectories/${SPLIT}_scorer}"
DS="${DS:-$NAVSIM_WS/exp/uni_scorer/dataset_${SPLIT}.pt}"
SEL="${SEL:-$NAVSIM_WS/exp/uni_scorer/selection_${SPLIT}_cv.json}"
SUB="${SUB:-$NAVSIM_WS/exp/uni_scorer/submission_${SPLIT}_cv.pkl}"

LOG="./scorer_navhard_cv_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "=== learned-scorer navhard CV ($SPLIT, K=$K) — $(date -Is) ==="
echo "  stage_b=$STAGE_B_DIR  ds=$DS"

[ -f "$ENV_FILE" ] || { echo "!! $ENV_FILE not found — run setup_navsim_env.sh first"; exit 1; }

# 1) phi_s + candidates for navhard (skip if already dumped)
if [ ! -d "$STAGE_B_DIR/phi_s" ] || [ ! -f "$STAGE_B_DIR/trajectories.pkl" ]; then
    echo "-- dumping navhard phi_s + candidates (Stage B)"
    TRAIN_TEST_SPLIT="$SPLIT" STAGE_B_OUT="$STAGE_B_DIR" \
        bash "$PROJECT_ROOT/scripts/navsim/run_navsim_stage_b_scorer.sh"
else
    echo "-- reusing existing dump at $STAGE_B_DIR"
fi

# the oracle sub-score CSVs (labels) for this split
# shellcheck source=/dev/null
source "$ENV_FILE"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$NAVSIM_EXP_ROOT_DEFAULT}"
CAND_GLOB="${CAND_GLOB:-$NAVSIM_EXP_ROOT/uni_oracle_${SPLIT}*_cand*}"

# 2)+3) build dataset and k-fold CV selection (unidrivevla env, torch)
mkdir -p "$(dirname "$DS")"
( cd "$HERE" && "$MM" run -n "${UNI_ENV:-unidrivevla}" bash -c "
    set -e
    [ -f '$DS' ] || python build_dataset.py '$STAGE_B_DIR' '$CAND_GLOB' '$DS'
    python cv_select.py '$DS' '$SEL' '$K' '$EPOCHS'
" )

# 4)+5) build the selected submission and score it (navsim env)
eval "$("$MM" shell hook --shell bash)"
micromamba activate "${NAVSIM_ENV:-navsim}"
export TRAIN_TEST_SPLIT="$SPLIT"
TS="$OPENSCENE_DATA_ROOT/$SPLIT"   # v2.2: split is top-level, not nested
CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache/$SPLIT}"
EXP="uni_scorer_cv_${SPLIT}"

echo "-- building CV-selected submission"
SELECTION="$SEL" STAGE_B_TRAJ="$STAGE_B_DIR/trajectories.pkl" SUBMISSION_OUT="$SUB" \
    python "$PROJECT_ROOT/scripts/navsim/stage_c_build_submission.py"

echo "-- scoring CV-selected submission (EPDMS)"
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_from_submission.py" \
    train_test_split="$SPLIT" \
    experiment_name="$EXP" \
    submission_file_path="$SUB" \
    metric_cache_path="$CACHE_PATH" \
    synthetic_sensor_path="$TS/sensor_blobs" \
    synthetic_scenes_path="$TS/synthetic_scene_pickles"

CSV=$(find "$NAVSIM_EXP_ROOT/$EXP" -name "*.csv" -printf '%T+ %p\n' 2>/dev/null | sort | tail -1 | cut -d' ' -f2-)
echo "  CV-selected score CSV: $CSV"
python - "$CSV" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1]); df = df[df["token"] != "average"]
s = df["score"] if "score" in df.columns else df["score_stage_one"].combine_first(df["score_stage_two"])
print(f"=== LEARNED-SCORER (CV) EPDMS on {len(s)} tokens: {s.mean():.4f} ===")
PY
echo "=== DONE — log: $LOG ==="