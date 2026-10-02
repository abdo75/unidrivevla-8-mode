#!/bin/bash
#
# Stage B MODES smoke (Increment 2) — same as run_navsim_stage_b_modes.sh but only
# 5 tokens, into a separate _smoke output dir, to eyeball the candidate spread
# before a full run. Zero flags. Full run = run_navsim_stage_b_modes.sh.
#
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "${BASH_SOURCE[0]}")/navsim_paths.sh" 2>/dev/null || true
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"
SPLIT="${TRAIN_TEST_SPLIT:-warmup_two_stage}"

export LIMIT="${LIMIT:-5}"
export STAGE_B_OUT="${STAGE_B_OUT:-$NAVSIM_WS/exp/uni_trajectories/${SPLIT}_modes_smoke}"

exec bash "$HERE/run_navsim_stage_b_modes.sh"