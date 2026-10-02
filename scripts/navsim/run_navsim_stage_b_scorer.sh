#!/bin/bash
#
# Stage B MODES + phi_s dump FULL (Increment 3 / learned scorer data-gen). Same as
# run_navsim_stage_b_modes.sh but dumps the per-scene perception tokens
# (DUMP_SCENE_TOKENS=1) into a dedicated _scorer dir, for the whole split. This is the
# scorer data run (default split navhard_two_stage). Zero flags. Smoke (a few tokens)
# = run_navsim_stage_b_scorer_smoke.sh.
#
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "${BASH_SOURCE[0]}")/navsim_paths.sh" 2>/dev/null || true
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"
SPLIT="${TRAIN_TEST_SPLIT:-navhard_two_stage}"

export DUMP_SCENE_TOKENS=1
export STAGE_B_OUT="${STAGE_B_OUT:-$NAVSIM_WS/exp/uni_trajectories/${SPLIT}_scorer}"

exec bash "$HERE/run_navsim_stage_b_modes.sh"