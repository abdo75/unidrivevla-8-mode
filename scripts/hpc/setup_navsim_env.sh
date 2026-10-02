#!/bin/bash
#
# NavSim devkit + micromamba env setup (DrivoR thesis) — ubix.
# ------------------------------------------------------------
# What:  clone the NavSim devkit, create its OWN micromamba env, install the
#        devkit editable, and emit navsim_env.sh with env vars pointed at the
#        shared (read-only) dataset + a writable exp dir.
# Why:   NavSim/nuplan-devkit pins clash with the torch-2.7 unidrivevla env, so
#        it gets a separate env. PDMS scoring is CPU-only — this env's torch does
#        NOT need to support the Blackwell sm_120 GPU.
# Usage: bash scripts/hpc/setup_navsim_env.sh
#        Overrides: NAVSIM_WS=/writable/ws  OPENSCENE_DATA_ROOT=/data  NAVSIM_ENV=navsim
# Idempotent: skips the clone / env-create / editable-install if already present.
#
set -eo pipefail

MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

# --- paths (all overridable via env) ----------------------------------------
# central path config (edit scripts/navsim/navsim_paths.sh to relocate) + fallback
source "$(dirname "${BASH_SOURCE[0]}")/../navsim/navsim_paths.sh" 2>/dev/null || true
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"                       # writable workspace
DATASET="${OPENSCENE_DATA_ROOT:-$HOME/navsim_dataset}"         # data source (now fetched, not read-only)
NAVSIM_REPO="${NAVSIM_REPO:-https://github.com/autonomousvision/navsim.git}"
DEVKIT="$NAVSIM_WS/navsim"
EXP="$NAVSIM_WS/exp"

LOG="./setup_navsim_env_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "=== NavSim env setup — $(date -Is) on $(hostname -s) ==="
echo "NAVSIM_WS=$NAVSIM_WS"
echo "DATASET  =$DATASET  (read-only data source)"
echo "log=$LOG"

# --- preflight: writable workspace, readable dataset ------------------------
mkdir -p "$NAVSIM_WS" 2>/dev/null || true
[ -w "$NAVSIM_WS" ] || { echo "!! NAVSIM_WS not writable: $NAVSIM_WS — set NAVSIM_WS=/writable/path"; exit 1; }
{ [ -d "$DATASET" ] && [ -r "$DATASET" ]; } || { echo "!! dataset not readable: $DATASET — set OPENSCENE_DATA_ROOT"; exit 1; }
[ -d "$DATASET/maps" ] || echo "  (warning: $DATASET/maps missing — check NUPLAN_MAPS_ROOT later)"

# --- A. clone devkit (idempotent) -------------------------------------------
echo "=== A. NavSim devkit -> $DEVKIT ==="
if [ -d "$DEVKIT/.git" ]; then
    echo "  already cloned — skipping (update manually with: git -C $DEVKIT pull)"
else
    git clone "$NAVSIM_REPO" "$DEVKIT"
fi
[ -f "$DEVKIT/environment.yml" ] || { echo "!! $DEVKIT/environment.yml not found after clone"; exit 1; }

# env name comes from the yml's `name:` (override with NAVSIM_ENV=)
ENV_NAME="${NAVSIM_ENV:-$(awk '/^name:/{print $2; exit}' "$DEVKIT/environment.yml")}"
ENV_NAME="${ENV_NAME:-navsim}"
echo "  env name: $ENV_NAME"

# --- B. micromamba env (idempotent) -----------------------------------------
echo "=== B. micromamba env '$ENV_NAME' ==="
if [ -d "$MAMBA_ROOT_PREFIX/envs/$ENV_NAME" ]; then
    echo "  env already exists at $MAMBA_ROOT_PREFIX/envs/$ENV_NAME — skipping create"
else
    "$MM" create -y -f "$DEVKIT/environment.yml"
fi
eval "$("$MM" shell hook --shell bash)"
micromamba activate "$ENV_NAME"
echo "  active prefix: $(python -c 'import sys; print(sys.prefix)' 2>/dev/null || echo '?')"

# --- C. install devkit editable ---------------------------------------------
echo "=== C. pip install -e $DEVKIT ==="
python -m pip install -e "$DEVKIT"

# --- D. emit env-var file for the eval step ---------------------------------
echo "=== D. write $NAVSIM_WS/navsim_env.sh ==="
mkdir -p "$EXP"
cat > "$NAVSIM_WS/navsim_env.sh" <<EOF
# Source before running NavSim eval:  source $NAVSIM_WS/navsim_env.sh
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=$DATASET
export NUPLAN_MAPS_ROOT=$DATASET/maps
export NAVSIM_DEVKIT_ROOT=$DEVKIT
export NAVSIM_EXP_ROOT=$EXP
EOF
cat "$NAVSIM_WS/navsim_env.sh"

# --- E. smoke-check the import ----------------------------------------------
echo "=== E. import check ==="
python -c "import navsim; print('navsim import OK:', navsim.__file__)" \
    || { echo "!! navsim import failed — env built but package not importable"; exit 1; }

echo
echo "=== DONE — log: $LOG ==="
echo "Next:"
echo "  micromamba activate $ENV_NAME"
echo "  source $NAVSIM_WS/navsim_env.sh"
echo "  # then run the warmup constant-velocity PDMS eval (next script)."
