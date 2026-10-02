#!/bin/bash
#
# NavSim metric caching — build the per-scene GT cache that PDMS scoring loads.
# ----------------------------------------------------------------------------
# NavSim eval is TWO steps: (1) this script builds the metric cache, then
# (2) run_navsim_warmup_cv.sh loads it and scores. run_pdm_score.py is a pure
# LOADER — it will not build the cache, hence this must run first.
# Pre:   setup_navsim_env.sh (creates 'navsim' env + navsim_env.sh).
# Usage: bash scripts/hpc/run_navsim_metric_cache.sh
#        TRAIN_TEST_SPLIT=navhard_two_stage   # the real split (default: warmup_two_stage)
#        METRIC_CACHE_PATH=/path              # default: $NAVSIM_EXP_ROOT/metric_cache/<split>
# NOTE:  keep METRIC_CACHE_PATH identical between this and the score script, or
#        the scorer won't find the cache.
#
set -eo pipefail

MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

source "$(dirname "${BASH_SOURCE[0]}")/../navsim/navsim_paths.sh" 2>/dev/null || true
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"
ENV_NAME="${NAVSIM_ENV:-navsim}"
ENV_FILE="$NAVSIM_WS/navsim_env.sh"
SPLIT="${TRAIN_TEST_SPLIT:-warmup_two_stage}"

LOG="./run_navsim_metric_cache_${SPLIT}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "=== NavSim metric caching — $(date -Is) on $(hostname -s) ==="

[ -f "$ENV_FILE" ] || { echo "!! $ENV_FILE not found — run setup_navsim_env.sh first"; exit 1; }
# shellcheck source=/dev/null
source "$ENV_FILE"
eval "$("$MM" shell hook --shell bash)"
micromamba activate "$ENV_NAME"

# v2.2 official layout: each two-stage split is top-level ($DATA/<split>/...). Metric
# caching needs the ORIGINAL stage-1 logs and the stage-2 synthetic scenes:
#   - stage-1 logs = the standard base split (navhard's data_split=test), i.e.
#     $DATA/navsim_logs/test, filtered to navhard's 87 logs. This is the hydra DEFAULT
#     (navsim_logs/<data_split>), so we do NOT override navsim_log_path -- the split
#     config already points there. (Fetch it via fetch_navsim_data.sh, metadata only.)
#   - stage-2 synthetic scenes = $DATA/<split>/synthetic_scene_pickles (the v2.2 top-level
#     layout), which we DO override. Sensors are unused for caching (no-sensors loader).
TWO_STAGE_DIR="$OPENSCENE_DATA_ROOT/$SPLIT"
SYNTHETIC_SCENES_PATH="$TWO_STAGE_DIR/synthetic_scene_pickles"
CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache/$SPLIT}"
mkdir -p "$CACHE_PATH"

echo "  split:    $SPLIT"
echo "  cache ->  $CACHE_PATH   (writes \$cache/metadata)"
echo "  scenes:   $SYNTHETIC_SCENES_PATH   (synthetic stage-2 scenes)"
echo "  logs:     navsim_logs/<data_split> (hydra default; base split for stage-1)"

[ -e "$SYNTHETIC_SCENES_PATH" ] || { echo "!! path missing: $SYNTHETIC_SCENES_PATH — check layout under $TWO_STAGE_DIR"; exit 1; }
if [ ! -d "$OPENSCENE_DATA_ROOT/navsim_logs" ]; then
    echo "!! $OPENSCENE_DATA_ROOT/navsim_logs missing — stage-1 logs not fetched."
    echo "   run: bash scripts/hpc/fetch_navsim_data.sh   (it fetches the test-split logs, metadata only)"
    exit 1
fi

echo "=== running run_metric_caching.py ==="
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split="$SPLIT" \
    metric_cache_path="$CACHE_PATH" \
    synthetic_scenes_path="$SYNTHETIC_SCENES_PATH"

echo "=== DONE — cache at $CACHE_PATH/metadata ==="
echo "Next: bash scripts/hpc/run_navsim_warmup_cv.sh   (same split → loads this cache)"
