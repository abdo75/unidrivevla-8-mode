#!/bin/bash -l
#
# Generate v1.0-mini info pkls into a SEPARATE dir (data/infos_mini/) so the
# existing trainval infos in data/infos/ are left untouched. Lets us run the
# low-RAM mini fine-tune smoke without overwriting trainval. Idempotent.
#
# mini scenes are a subset of trainval, so the trainval-symlinked samples/sweeps/
# gts already contain the images these infos reference. Run once on ubix.
#
set -eo pipefail
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
eval "$("$MM" shell hook --shell bash)"
micromamba activate unidrivevla

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT/nuScenes"
export PYTHONPATH="$PWD:$PYTHONPATH"

OUT="data/infos_mini"
mkdir -p "$OUT"
echo "=== generating v1.0-mini infos into $OUT (trainval infos untouched) ==="

python tools/data_converter/nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes --canbus ./data/nuscenes \
    --out-dir "./$OUT/" --extra-tag nuscenes --version v1.0-mini
# the converter drops basic infos into <out>/mini/ for the mini version; lift them
[ -d "$OUT/mini" ] && mv -f "$OUT"/mini/*.pkl "$OUT"/ 2>/dev/null || true

python tools/data_converter/vad_nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes --canbus ./data/nuscenes \
    --out-dir "./$OUT/" --extra-tag vad_nuscenes --version v1.0-mini
[ -d "$OUT/mini" ] && mv -f "$OUT"/mini/*.pkl "$OUT"/ 2>/dev/null || true

echo "=== done; contents of $OUT ==="
ls -lh "$OUT"/*.pkl 2>/dev/null
echo "(trainval infos still intact:)"; ls -lh data/infos/nuscenes_infos_train.pkl
