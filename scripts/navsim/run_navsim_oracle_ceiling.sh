#!/bin/bash
#
# Oracle ceiling (DrivoR thesis, Increment 2 -> motivates Increment 3).
# For each candidate index k=0..N_MODES-1: build a NavSim submission from Stage B's
# multi-candidate trajectories (CAND_IDX=k) and score it with the real PDMS. Then
# aggregate best-of-N per token. Runs in the `navsim` env. Zero flags.
#
# Pre: Stage B with MODE_CANDIDATES=1 produced trajectories.pkl of {token:(N,6,2)}
#      (run_navsim_stage_b_modes_smoke.sh for a few, or a full modes run), and the
#      metric cache exists (run_navsim_metric_cache.sh).
#
set -eo pipefail
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

source "$(dirname "${BASH_SOURCE[0]}")/navsim_paths.sh" 2>/dev/null || true
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"
ENV_NAME="${NAVSIM_ENV:-navsim}"
ENV_FILE="$NAVSIM_WS/navsim_env.sh"
SPLIT="${TRAIN_TEST_SPLIT:-warmup_two_stage}"
N_MODES="${N_MODES:-8}"   # converged FT is N=8 (was 6 for the old overfit model)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG="./navsim_oracle_${SPLIT}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "=== Oracle ceiling ($SPLIT, N_MODES=$N_MODES) — $(date -Is) ==="

[ -f "$ENV_FILE" ] || { echo "!! $ENV_FILE not found — run setup_navsim_env.sh first"; exit 1; }
# shellcheck source=/dev/null
source "$ENV_FILE"
eval "$("$MM" shell hook --shell bash)"
micromamba activate "$ENV_NAME"

export TRAIN_TEST_SPLIT="$SPLIT"
TS="$OPENSCENE_DATA_ROOT/$SPLIT"   # v2.2: split is top-level, not nested
# Reads the FULL Stage B modes output (run_navsim_stage_b_modes.sh) by default.
export STAGE_B_TRAJ="${STAGE_B_TRAJ:-$NAVSIM_WS/exp/uni_trajectories/${SPLIT}_modes/trajectories.pkl}"
CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache/$SPLIT}"
OUT_DIR="${ORACLE_OUT:-$NAVSIM_WS/exp/uni_oracle/$SPLIT}"
mkdir -p "$OUT_DIR"

[ -f "$STAGE_B_TRAJ" ] || { echo "!! $STAGE_B_TRAJ not found — run Stage B with MODE_CANDIDATES=1 first"; exit 1; }
[ -d "$CACHE_PATH/metadata" ] || { echo "!! metric cache missing at $CACHE_PATH — run run_navsim_metric_cache.sh"; exit 1; }

CSVS=()
for k in $(seq 0 $((N_MODES - 1))); do
    echo; echo "######## candidate $k / $((N_MODES - 1)) ########"
    export CAND_IDX="$k"
    export SUBMISSION_OUT="$OUT_DIR/submission_cand${k}.pkl"
    EXP="uni_oracle_${SPLIT}_cand${k}"

    echo "--- build submission (CAND_IDX=$k) ---"
    python "$HERE/stage_c_build_submission.py"

    echo "--- score submission (PDMS) ---"
    python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_from_submission.py" \
        train_test_split="$SPLIT" \
        experiment_name="$EXP" \
        submission_file_path="$SUBMISSION_OUT" \
        metric_cache_path="$CACHE_PATH" \
        synthetic_sensor_path="$TS/sensor_blobs" \
        synthetic_scenes_path="$TS/synthetic_scene_pickles"

    # newest score CSV produced under this experiment
    CSV=$(find "$NAVSIM_EXP_ROOT/$EXP" -name "*.csv" -printf '%T+ %p\n' 2>/dev/null | sort | tail -1 | cut -d' ' -f2-)
    [ -n "$CSV" ] || { echo "!! no score CSV found under $NAVSIM_EXP_ROOT/$EXP"; exit 1; }
    echo "  candidate $k score CSV: $CSV"
    CSVS+=("$CSV")
done

echo; echo "######## aggregate best-of-$N_MODES ########"
python "$HERE/oracle_aggregate.py" "${CSVS[@]}"
echo "=== DONE — log: $LOG ==="