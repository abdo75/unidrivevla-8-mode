#!/bin/bash
#
# Stage A SMOKE — dumps just 4 tokens to validate the adapter. No flags to
# remember: the limit is baked in. Full run = run_navsim_stage_a.sh (all tokens).
#
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIMIT=4
exec bash "$HERE/run_navsim_stage_a.sh"
