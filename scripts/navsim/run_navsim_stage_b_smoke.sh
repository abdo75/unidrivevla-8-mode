#!/bin/bash
#
# Stage B SMOKE — run UniDriveVLA on the first 3 Stage-A dumps and print the
# trajectories (frame check). Limit baked in; no flags to remember.
#
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIMIT=3
exec bash "$HERE/run_navsim_stage_b.sh"
