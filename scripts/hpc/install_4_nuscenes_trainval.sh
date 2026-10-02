#!/bin/bash
#
# Blackwell bootstrap — TRAINVAL: assemble full nuScenes trainval on ubix for
# full-val (6019-sample) Stage-2 eval.
# ----------------------------------------------------------------------------
# Run AFTER install_3_nuscenes_mini.sh (which downloaded checkpoints, Occ3D gts,
# map expansion, CAN bus to the mini data root). FROM the repo root:
#     bash scripts/hpc/install_4_nuscenes_trainval.sh
#
# Strategy: the huge blobs (samples/sweeps/v1.0-trainval) go on a big writable
# disk (TRAINVAL_ROOT, auto-detected under /mnt/shared); gts/can_bus/maps are
# reused from the mini root (MINI_ROOT) via per-subdir symlinks.
#
#   A. resolve a writable TRAINVAL_ROOT with >=400 GB free (override: TRAINVAL_ROOT=)
#   B. auto-download the 11 trainval tarballs from the AWS Open Data mirror
#      (login-free), extract each, delete after (one at a time to cap disk)
#   C. re-point repo symlinks (samples/sweeps/v1.0-trainval -> TRAINVAL_ROOT;
#      maps/can_bus/gts/v1.0-mini -> MINI_ROOT)
#   D. generate trainval info pkls (~1.5 h CPU)
#   E. regenerate k-means anchors on trainval
#   F. print the full-val eval command
#
# Idempotent: per-tarball extract markers; skips pkl/kmeans if present.

set -eo pipefail
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

# MINI_ROOT: dir holding the mini-era assets (gts/can_bus/maps/v1.0-mini).
# Probe by EXISTENCE of v1.0-mini/ (not just writability) so we land where step
# 3 actually put the data, not just any new writable spot.
need_gb=400
avail_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
if [ -z "${MINI_ROOT:-}" ]; then
    for cand in "/mnt/shared/$USER/nuscenes" "/mnt/shared/nuscenes" "$HOME/nuscenes"; do
        if [ -d "$cand/v1.0-mini" ]; then MINI_ROOT="$cand"; break; fi
    done
fi
if [ -z "${MINI_ROOT:-}" ]; then
    echo "!! MINI_ROOT not found — could not locate v1.0-mini/ under"
    echo "   /mnt/shared/$USER/nuscenes, /mnt/shared/nuscenes, or \$HOME/nuscenes."
    echo "   Set MINI_ROOT=/path/to/mini/nuscenes (the dir that has v1.0-mini,"
    echo "   gts, can_bus, maps from step 3) and re-run."
    exit 1
fi

# TRAINVAL_ROOT: a writable big-disk path scoped to YOU. Prefer /mnt/shared/$USER,
# else /mnt/shared root if writable. NO glob over /mnt/shared/*/ — that risks
# picking another user's subdir (which is what just happened with /mnt/shared/sc).
if [ -z "${TRAINVAL_ROOT:-}" ]; then
    for d in "/mnt/shared/$USER" "/mnt/shared"; do
        if [ -d "$d" ] && [ -w "$d" ]; then
            a=$(avail_gb "$d")
            if [ "${a:-0}" -ge "$need_gb" ]; then TRAINVAL_ROOT="${d%/}/nuscenes_trainval"; break; fi
        fi
    done
fi
if [ -z "${TRAINVAL_ROOT:-}" ]; then
    cat <<EOF
!! Could not auto-pick TRAINVAL_ROOT. Need a writable dir YOU own/group-own with
   >=${need_gb} GB free. Two options:
     1) mkdir -p /mnt/shared/$USER  &&  re-run    (preferred — scopes data to you)
     2) TRAINVAL_ROOT=/path/with/space  bash scripts/hpc/install_4_nuscenes_trainval.sh
   Current mounts:
EOF
    df -h | grep -vE 'tmpfs|udev|loop'
    exit 1
fi
mkdir -p "$TRAINVAL_ROOT"
a=$(avail_gb "$TRAINVAL_ROOT")
echo "TRAINVAL_ROOT=$TRAINVAL_ROOT (${a} GB free)"
echo "MINI_ROOT=$MINI_ROOT     (v1.0-mini found here)"
[ "${a:-0}" -ge "$need_gb" ] || { echo "!! Only ${a} GB free at TRAINVAL_ROOT; need >=${need_gb}."; exit 1; }

if [ ! -d nuScenes ] || [ ! -d third_party/mmcv-1.7.2 ]; then
    echo "!! Run from the repo root."; exit 1
fi
REPO_ROOT="$(pwd)"
LOG="$REPO_ROOT/install_4_nuscenes_trainval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "logging to $LOG"

eval "$("$MM" shell hook --shell bash)"
micromamba activate unidrivevla

# --- B. trainval tarballs (auto-download from the AWS Open Data mirror) ------
echo "=== B. trainval tarballs -> download + extract into $TRAINVAL_ROOT ==="
TGZ_DIR="$TRAINVAL_ROOT/trainval_tgz"
mkdir -p "$TGZ_DIR"
# nuScenes is mirrored login-free on AWS Open Data (bucket s3://motional-nuscenes,
# region ap-northeast-1, CloudFront https://d36yt3mvayqw5m.cloudfront.net) — no
# account or JS-gated link needed. Override NUSC_CF for another mirror, or
# NUSCENES_TRAINVAL_BASE to point at a directory of pre-downloaded tarballs.
NUSC_CF="${NUSC_CF:-https://d36yt3mvayqw5m.cloudfront.net/public/v1.0}"
DL_BASE="${NUSCENES_TRAINVAL_BASE:-$NUSC_CF}"
TARBALLS=(v1.0-trainval_meta.tgz \
    v1.0-trainval01_blobs.tgz v1.0-trainval02_blobs.tgz v1.0-trainval03_blobs.tgz \
    v1.0-trainval04_blobs.tgz v1.0-trainval05_blobs.tgz v1.0-trainval06_blobs.tgz \
    v1.0-trainval07_blobs.tgz v1.0-trainval08_blobs.tgz v1.0-trainval09_blobs.tgz \
    v1.0-trainval10_blobs.tgz)
# One tarball at a time: download (resumable), extract, delete — caps peak disk at
# ~one extra tarball over the extracted set rather than all 11 at once. Re-running
# resumes: the .done markers skip finished pieces and wget -c continues partials.
for f in "${TARBALLS[@]}"; do
    [ -f "$TGZ_DIR/.$f.done" ] && { echo "  $f already extracted — skipping"; continue; }
    # wget -c is a no-op if the file is already complete and resumes it if partial,
    # so a re-run after an interrupted download just picks up where it stopped.
    echo "  downloading $f from $DL_BASE ..."
    wget -c --tries=20 --waitretry=15 --timeout=60 -O "$TGZ_DIR/$f" "$DL_BASE/$f"
    echo "  extracting $f ..."
    tar -xzf "$TGZ_DIR/$f" -C "$TRAINVAL_ROOT" && touch "$TGZ_DIR/.$f.done" && rm -f "$TGZ_DIR/$f"
    echo "  done $f (tarball removed)"
done
echo "all trainval tarballs downloaded + extracted"

# --- C. re-point repo symlinks ----------------------------------------------
echo "=== C. symlinks: big blobs -> TRAINVAL_ROOT; gts/can_bus/maps -> MINI_ROOT ==="
mkdir -p nuScenes/data/nuscenes
for d in samples sweeps v1.0-trainval; do
    [ -e "$TRAINVAL_ROOT/$d" ] && ln -sfn "$TRAINVAL_ROOT/$d" "nuScenes/data/nuscenes/$d" && echo "  $d -> $TRAINVAL_ROOT/$d"
done
for d in maps can_bus gts v1.0-mini; do
    [ -e "$MINI_ROOT/$d" ] && ln -sfn "$MINI_ROOT/$d" "nuScenes/data/nuscenes/$d" && echo "  $d -> $MINI_ROOT/$d"
done

# --- D. trainval info pkls --------------------------------------------------
echo "=== D. generate trainval info pkls (overwrites mini pkls; ~1.5 h) ==="
if [ -f nuScenes/data/infos/nuscenes_infos_train.pkl ] && \
   [ "$(stat -c%s nuScenes/data/infos/nuscenes_infos_train.pkl)" -gt 100000000 ]; then
    echo "trainval-sized pkls already present — skipping (delete to regenerate)"
else
    pushd nuScenes >/dev/null
    export PYTHONPATH="$PWD:$PYTHONPATH"
    # --version v1.0 builds trainval THEN tries v1.0-test (no test data here) —
    # the test step fails AFTER the trainval pkls are written, so tolerate it.
    python tools/data_converter/nuscenes_converter.py nuscenes \
        --root-path ./data/nuscenes --canbus ./data/nuscenes \
        --out-dir ./data/infos/ --extra-tag nuscenes --version v1.0 \
        || echo "(nuscenes_converter v1.0-test step failed as expected — verifying trainval pkls)"
    python tools/data_converter/vad_nuscenes_converter.py nuscenes \
        --root-path ./data/nuscenes --canbus ./data/nuscenes \
        --out-dir ./data/infos/ --extra-tag vad_nuscenes --version v1.0 \
        || echo "(vad_converter v1.0-test step failed as expected — verifying trainval pkls)"
    popd >/dev/null
    [ -f nuScenes/data/infos/nuscenes_infos_train.pkl ] && \
    [ -f nuScenes/data/infos/nuscenes_infos_val.pkl ] && \
    [ -f nuScenes/data/infos/vad_nuscenes_infos_temporal_val.pkl ] || {
        echo "!! trainval pkl generation failed (trainval pkls missing)."; exit 1; }
fi
ls -la nuScenes/data/infos/*.pkl 2>/dev/null

# --- E. trainval k-means anchors --------------------------------------------
echo "=== E. regenerate k-means anchors on trainval ==="
pushd nuScenes >/dev/null
export PYTHONPATH="$PWD:$PYTHONPATH"
python tools/kmeans/kmeans_det.py
python tools/kmeans/kmeans_map.py
python tools/kmeans/kmeans_motion.py
python tools/kmeans/kmeans_plan.py   # works on trainval (mini lacked plan trajs)
popd >/dev/null
ls -la nuScenes/data/kmeans/*.npy 2>/dev/null

# --- F. eval command --------------------------------------------------------
echo ""
echo "=== TRAINVAL SETUP DONE — log: $LOG ==="
cat <<EOF

Run the full-val Stage-2 eval (NUSC_VERSION=trainval -> v1.0-trainval val split,
6019 samples; full eval_mode incl. tracking/motion):

  NUSC_VERSION=trainval bash scripts/hpc/eval_stage2_2b_smoke.sh

(One GPU; flow-matching denoise makes this take a while over 6019 samples, but
no queue/timeout. Output tees to ./eval_stage2_2b_smoke_<ts>.log.)
EOF
