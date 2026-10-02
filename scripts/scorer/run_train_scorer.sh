#!/bin/bash
#
# Train the learned PDMS scorer (unidrivevla env). Zero flags: builds the dataset
# from the Stage B dump + Stage C sub-score CSVs if DS does not exist, then trains.
# Defaults target the navtrain data-gen output; override DS/OUT/STAGE_B_DIR/CSV_DIR
# only to point at a different slice (e.g. the 4-scene smoke for the overfit go/no-go).
#
set -eo pipefail
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$(dirname "${BASH_SOURCE[0]}")/../navsim/navsim_paths.sh" 2>/dev/null || true
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stage B dump (phi_s/ + trajectories.pkl) and the per-candidate sub-score CSV dirs.
# CAND_GLOB expands to ONE split's uni_oracle_<split>_cand<k> dirs (the oracle output),
# so cand0 from another split cannot overwrite this split's cand0.
SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$NAVSIM_WS/exp}"
STAGE_B_DIR="${STAGE_B_DIR:-$NAVSIM_WS/exp/uni_trajectories/${SPLIT}_modes}"
CAND_GLOB="${CAND_GLOB:-$NAVSIM_EXP_ROOT/uni_oracle_${SPLIT}*_cand*}"
DS="${DS:-$NAVSIM_WS/exp/uni_scorer/dataset.pt}"
OUT="${OUT:-$NAVSIM_WS/exp/uni_scorer/scorer.pt}"

mkdir -p "$(dirname "$DS")"
LOG="./scorer_train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "=== train scorer — $(date -Is) ==="
echo "  stage_b=$STAGE_B_DIR  cand_glob=$CAND_GLOB"
echo "  ds=$DS  out=$OUT"

eval "$("$MM" shell hook --shell bash)"
micromamba activate "${UNI_ENV:-unidrivevla}"
cd "$HERE"

if [ ! -f "$DS" ]; then
    echo "-- building dataset"
    python build_dataset.py "$STAGE_B_DIR" "$CAND_GLOB" "$DS"
fi
python train_scorer.py "$DS" "$OUT"
echo "=== DONE — log: $LOG ==="