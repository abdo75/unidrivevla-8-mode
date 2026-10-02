#!/bin/bash
#
# Stage A runner (navsim env) — dump UniDriveVLA-ready inputs per NavSim token.
# Pre: setup_navsim_env.sh. Smoke first:  LIMIT=4 bash scripts/navsim/run_navsim_stage_a.sh
#
set -eo pipefail
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

source "$(dirname "${BASH_SOURCE[0]}")/navsim_paths.sh" 2>/dev/null || true
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"
ENV_NAME="${NAVSIM_ENV:-navsim}"
ENV_FILE="$NAVSIM_WS/navsim_env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG="./navsim_stage_a_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

[ -f "$ENV_FILE" ] || { echo "!! $ENV_FILE not found — run setup_navsim_env.sh first"; exit 1; }
# shellcheck source=/dev/null
source "$ENV_FILE"
eval "$("$MM" shell hook --shell bash)"
micromamba activate "$ENV_NAME"

export TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-warmup_two_stage}"
echo "=== Stage A ($TRAIN_TEST_SPLIT, LIMIT=${LIMIT:-0}) — $(date -Is) ==="
python "$HERE/stage_a_dump_inputs.py"
echo "=== DONE — log: $LOG ==="
