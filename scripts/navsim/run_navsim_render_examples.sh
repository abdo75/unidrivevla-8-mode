#!/bin/bash
#
# Render good/bad navhard BEV example reels (map + agents + our N candidates).
# ---------------------------------------------------------------------------
# Chains find_examples.py (pick tokens from the per-candidate EPDMS CSVs) then
# render_navhard_examples.py (devkit BEV + our candidates, per-category mp4s).
# Zero flags for the common case; everything overridable.
#
# Pre: navhard oracle CSVs (uni_oracle_navhard_two_stage*_cand*), Stage B _modes
#      trajectories.pkl, and (optional) the scorer selection.json all exist.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/navsim_paths.sh" 2>/dev/null || true
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SPLIT="${TRAIN_TEST_SPLIT:-navhard_two_stage}"
ENV_FILE="$NAVSIM_WS/navsim_env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$ENV_FILE" ] || { echo "!! $ENV_FILE not found — run setup_navsim_env.sh first"; exit 1; }
# shellcheck source=/dev/null
source "$ENV_FILE"

MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$NAVSIM_WS/exp}"
export CAND_GLOB="${CAND_GLOB:-$NAVSIM_EXP_ROOT/uni_oracle_${SPLIT}*_cand*}"
export SELECTION="${SELECTION:-$NAVSIM_EXP_ROOT/uni_scorer/selection_${SPLIT}_cv.json}"
export STAGE_B_TRAJ="${STAGE_B_TRAJ:-$NAVSIM_EXP_ROOT/uni_trajectories/${SPLIT}_modes/trajectories.pkl}"
export OUT_DIR="${OUT_DIR:-./navhard_examples}"
EXAMPLES_JSON="${EXAMPLES_JSON:-$NAVSIM_EXP_ROOT/uni_scorer/examples_${SPLIT}.json}"
export EXAMPLES_JSON

LOG="./render_examples_${SPLIT}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "=== navhard example reels ($SPLIT) — $(date -Is) ==="

# 1) pick bad/good example tokens (unidrivevla env: pandas/torch) — prints single-quality stats
"$MM" run -n "${UNI_ENV:-unidrivevla}" python "$HERE/../scorer/find_examples.py" \
    "$CAND_GLOB" "$SELECTION" "$EXAMPLES_JSON"

# 2) render + stitch reels (navsim env: has the devkit visualization + hydra)
eval "$("$MM" shell hook --shell bash)"
micromamba activate "${NAVSIM_ENV:-navsim}"
python "$HERE/render_navhard_examples.py"
echo "=== DONE — reels in $OUT_DIR, log: $LOG ==="
