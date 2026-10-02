#!/bin/bash
#
# Run the NavSim AgentInput inspector in the navsim env (DrivoR thesis, Stage A prep).
# Read-only: loads one warmup token and prints tensor formats for the adapter.
# Pre: setup_navsim_env.sh (navsim env + navsim_env.sh).
#
set -eo pipefail
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

source "$(dirname "${BASH_SOURCE[0]}")/navsim_paths.sh" 2>/dev/null || true
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"
ENV_NAME="${NAVSIM_ENV:-navsim}"
ENV_FILE="$NAVSIM_WS/navsim_env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG="./navsim_inspect_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

[ -f "$ENV_FILE" ] || { echo "!! $ENV_FILE not found — run setup_navsim_env.sh first"; exit 1; }
# shellcheck source=/dev/null
source "$ENV_FILE"
eval "$("$MM" shell hook --shell bash)"
micromamba activate "$ENV_NAME"

export TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-warmup_two_stage}"
echo "=== inspecting $TRAIN_TEST_SPLIT — $(date -Is) ==="
python "$HERE/inspect_agent_input.py"
echo "=== DONE — log: $LOG ==="
