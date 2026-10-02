#!/bin/bash
#
# Stage B COMMAND-SWEEP smoke — does the command (right/left/straight) bend the
# trajectory? For 5 scenes, runs the model 3x (one per command) with the noise
# held FIXED, so the only variable is the command. Prints the 3 endpoints.
# Limits baked in (no flags). Separate output dir.
#
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CMD_SWEEP=1
export LIMIT=5
export STAGE_B_OUT="${STAGE_B_OUT:-/mnt/shared/UniDriveVLA/navsim/exp/uni_trajectories/warmup_cmdsweep}"
exec bash "$HERE/run_navsim_stage_b.sh"
