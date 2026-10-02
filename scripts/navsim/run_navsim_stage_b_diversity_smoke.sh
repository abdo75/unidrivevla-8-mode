#!/bin/bash
#
# Stage B DIVERSITY smoke — tests the core DrivoR premise: does re-seeding the
# flow-matching noise produce DIFFERENT trajectories? Generates 6 candidates for
# 5 tokens and prints the endpoint spread. Limits baked in (no flags).
# Writes to a separate dir so it doesn't overwrite the production trajectories.
#
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export N_CANDIDATES=6
export LIMIT=5
export STAGE_B_OUT="${STAGE_B_OUT:-/mnt/shared/UniDriveVLA/navsim/exp/uni_trajectories/warmup_diversity}"
exec bash "$HERE/run_navsim_stage_b.sh"
