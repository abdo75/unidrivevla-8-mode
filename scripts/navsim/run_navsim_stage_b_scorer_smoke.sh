#!/bin/bash
#
# Stage B MODES + phi_s dump SMOKE (Increment 3 / learned scorer). Same as
# run_navsim_stage_b_modes_smoke.sh but also dumps the per-scene perception tokens
# (DUMP_SCENE_TOKENS=1) the scorer attends over, into a dedicated _scorer_smoke dir.
# A few tokens only, to de-risk the dump + dataset build + overfit go/no-go before
# the full navtrain data-gen. Zero flags. Full data-gen = run_navsim_stage_b_scorer.sh.
#
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "${BASH_SOURCE[0]}")/navsim_paths.sh" 2>/dev/null || true
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"
SPLIT="${TRAIN_TEST_SPLIT:-warmup_two_stage}"

export DUMP_SCENE_TOKENS=1
export LIMIT="${LIMIT:-5}"
export STAGE_B_OUT="${STAGE_B_OUT:-$NAVSIM_WS/exp/uni_trajectories/${SPLIT}_scorer_smoke}"

exec bash "$HERE/run_navsim_stage_b_modes.sh"