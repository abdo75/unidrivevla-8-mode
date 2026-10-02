#!/bin/bash
#
# NavSim PDMS smoke — constant-velocity agent on warmup_two_stage (DrivoR thesis).
# ------------------------------------------------------------------------------
# What:  run NavSim's built-in PDMS eval with a NON-UniDriveVLA baseline agent on
#        the tiny warmup split. PURPOSE = prove the harness works (data paths +
#        PDMS oracle) and reveal the two-stage query pattern, BEFORE any
#        UniDriveVLA wiring. The CV agent needs no model and no GPU.
# Pre:   run setup_navsim_env.sh first (creates the 'navsim' env + navsim_env.sh).
# Usage: bash scripts/hpc/run_navsim_warmup_cv.sh
#        Switches (env overrides):
#          TRAIN_TEST_SPLIT=navhard_two_stage   # the real 33G eval (default: warmup_two_stage)
#          AGENT=ego_status_mlp_agent           # default: constant_velocity_agent
#          METRIC_CACHE_PATH=/path              # default: $NAVSIM_EXP_ROOT/metric_cache/<split>
#
set -eo pipefail

MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

source "$(dirname "${BASH_SOURCE[0]}")/../navsim/navsim_paths.sh" 2>/dev/null || true
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"
ENV_NAME="${NAVSIM_ENV:-navsim}"
ENV_FILE="$NAVSIM_WS/navsim_env.sh"
SPLIT="${TRAIN_TEST_SPLIT:-warmup_two_stage}"
AGENT="${AGENT:-constant_velocity_agent}"

LOG="./run_navsim_${AGENT}_${SPLIT}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "=== NavSim PDMS run — $(date -Is) on $(hostname -s) ==="

# --- env: source the vars emitted by setup, activate the navsim env ----------
[ -f "$ENV_FILE" ] || { echo "!! $ENV_FILE not found — run setup_navsim_env.sh first"; exit 1; }
# shellcheck source=/dev/null
source "$ENV_FILE"
eval "$("$MM" shell hook --shell bash)"
micromamba activate "$ENV_NAME"

# --- paths: two-stage data is nested under navhard_two_stage/<split>/ here ----
TWO_STAGE_DIR="$OPENSCENE_DATA_ROOT/$SPLIT"   # v2.2: split is top-level, not nested
SYNTHETIC_SENSOR_PATH="$TWO_STAGE_DIR/sensor_blobs"
SYNTHETIC_SCENES_PATH="$TWO_STAGE_DIR/synthetic_scene_pickles"
CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache/$SPLIT}"
mkdir -p "$CACHE_PATH"

echo "  split:    $SPLIT"
echo "  agent:    $AGENT"
echo "  sensors:  $SYNTHETIC_SENSOR_PATH"
echo "  scenes:   $SYNTHETIC_SCENES_PATH"
echo "  cache:    $CACHE_PATH"
echo "  exp_root: $NAVSIM_EXP_ROOT"

# --- preflight: the nested paths must exist (this is the real path check) -----
for p in "$SYNTHETIC_SENSOR_PATH" "$SYNTHETIC_SCENES_PATH"; do
    [ -e "$p" ] || { echo "!! path missing: $p"; echo "   check the nested two-stage layout under $OPENSCENE_DATA_ROOT/navhard_two_stage/"; exit 1; }
done

# --- run --------------------------------------------------------------------
echo "=== running run_pdm_score.py ==="
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
    train_test_split="$SPLIT" \
    agent="$AGENT" \
    experiment_name="${AGENT}_${SPLIT}" \
    metric_cache_path="$CACHE_PATH" \
    synthetic_sensor_path="$SYNTHETIC_SENSOR_PATH" \
    synthetic_scenes_path="$SYNTHETIC_SCENES_PATH"

echo "=== DONE — log: $LOG ==="
echo "Results (PDMS csv + score) under: $NAVSIM_EXP_ROOT/${AGENT}_${SPLIT}/"
