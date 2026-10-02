#!/bin/bash
#
# Blackwell bootstrap — STEP 3: data + checkpoints + mini pkls + anchors + cfg.
# ----------------------------------------------------------------------------
# Run AFTER step 2 (full stack imports) FROM the repo root, on ubix:
#     bash scripts/hpc/install_3_nuscenes_mini.sh
#
# Follows docs/data_preparation.md + docs/train_eval_nuscenes.md, cross-checked
# against what unidrivevla_stage1_2b_no_cotraining.py actually loads, adapted to
# v1.0-mini for a fast Stage-1 TRAINING smoke:
#   A. checkpoints: Stage-1 VLM weights + Stage-2 full model + OccVAE
#      (Stage-2 is what the multi-mode fine-tune loads via FT_LOAD_FROM)
#   B. nuScenes v1.0-mini (~4 GB, self-contained: samples/sweeps/maps/metadata)
#   C. Occ3D gts  -> data/nuscenes/gts   (train_pipeline LoadOccWorldLabels)
#   D. symlink data into the repo layout
#   E. mini info pkls (converters with --version v1.0-mini)
#   F. K-means anchors (kmeans.sh -> data/kmeans/*.npy; config hardcodes these)
#   G. derive a `_mini` Stage-1 config (version=mini, v1.0-mini)
#   H. print the exact smoke command
#
# Deliberately NOT downloaded (verified unused by the Stage-1 TRAIN config):
#   - eval pkls (vad_gt_seg.pkl, planing_gt_segmentation_val) — eval only. The
#     one file Stage-2 eval actually reads (planing_gt_segmentation_val) is staged
#     by eval_stage2_2b_smoke.sh, NOT here, so this train-setup step stays light.
#   - map expansion — handled in section B if the smoke's VectorizeMap needs it.
#
# Large data + checkpoints land on /mnt/shared by default (/home is ~92% full).
# Idempotent: re-running skips downloads/extractions/pkls that already exist.

set -eo pipefail
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

MM="$HOME/.local/bin/micromamba"
ENV_NAME=unidrivevla
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

# --- tunables ---------------------------------------------------------------
# Roots are probed by EXISTENCE of the assets, not writability. Writability lies
# after a group-perm change (e.g. /mnt/shared becomes writable but the ckpts/data
# were already downloaded to $HOME) — probing writability would relocate every
# path to an empty dir and re-download everything. So: reuse an existing install
# wherever it physically is; only fall back to a writable base for a FRESH setup.
# Override explicitly with DATA_ROOT=... / CKPT_ROOT=... / UNIDRIVEVLA_BASE=...
pick_writable_base() {
    for b in "/mnt/shared/$USER" "/mnt/shared"; do
        [ -d "$b" ] && [ -w "$b" ] && { echo "$b"; return; }
    done
    echo "$HOME"
}
BASE="${UNIDRIVEVLA_BASE:-$(pick_writable_base)}"
# CKPT_ROOT: existing Stage-1 VLM dir wins; else the writable base (fresh setup).
if [ -z "${CKPT_ROOT:-}" ]; then
    for cand in "/mnt/shared/$USER/UniDriveVLA/checkpoints" \
                "/mnt/shared/UniDriveVLA/checkpoints" \
                "$HOME/UniDriveVLA/checkpoints" \
                "$BASE/UniDriveVLA/checkpoints"; do
        [ -d "$cand/UniDriveVLA_Nusc_Base_Stage1" ] && { CKPT_ROOT="$cand"; break; }
    done
    CKPT_ROOT="${CKPT_ROOT:-$BASE/UniDriveVLA/checkpoints}"
fi
# DATA_ROOT: existing nuScenes data (mini or trainval) wins; else writable base.
if [ -z "${DATA_ROOT:-}" ]; then
    for cand in "/mnt/shared/$USER/nuscenes" \
                "/mnt/shared/nuscenes" \
                "$HOME/nuscenes" \
                "$BASE/nuscenes"; do
        { [ -d "$cand/v1.0-mini" ] || [ -d "$cand/v1.0-trainval" ]; } && { DATA_ROOT="$cand"; break; }
    done
    DATA_ROOT="${DATA_ROOT:-$BASE/nuscenes}"
fi
echo "resolved DATA_ROOT=$DATA_ROOT  CKPT_ROOT=$CKPT_ROOT"
# nuScenes raw data is on the AWS Open Data mirror (bucket s3://motional-nuscenes,
# region ap-northeast-1, CloudFront https://d36yt3mvayqw5m.cloudfront.net) — fully
# public, no login/terms gate, unlike the website's JS-gated download links. These
# defaults download everything anonymously; override NUSC_CF for a closer mirror.
NUSC_CF="${NUSC_CF:-https://d36yt3mvayqw5m.cloudfront.net/public/v1.0}"
NUSCENES_MINI_URL="${NUSCENES_MINI_URL:-$NUSC_CF/v1.0-mini.tgz}"
NUSCENES_MAP_URL="${NUSCENES_MAP_URL:-$NUSC_CF/nuScenes-map-expansion-v1.3.zip}"
NUSCENES_CANBUS_URL="${NUSCENES_CANBUS_URL:-$NUSC_CF/can_bus.zip}"
OCCVAE_GDRIVE_ID="${OCCVAE_GDRIVE_ID:-1gvzuz8ne6bJZZgCPxKdaiu-JkdAbfv3c}"
# Direct file id for gts.tar.gz inside the Occ3D folder 1Xarc91cNCNN3h8Vum-REbI-f0UlSf5Fc.
# Fetch the file directly (not --folder) so the quota'd annotations.json doesn't
# abort the whole download.
GTS_FILE_ID="${GTS_FILE_ID:-17HubGsfioQr1d_39VwVPXelobAFo4Xqh}"

LOG="./install_3_nuscenes_mini_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

if [ ! -d nuScenes ] || [ ! -d third_party/mmcv-1.7.2 ]; then
    echo "!! Run from the repo root. Aborting."; exit 1
fi
REPO_ROOT="$(pwd)"
echo "=== Blackwell bootstrap step 3 — $(date -Is) on $(hostname -s) ==="
echo "DATA_ROOT=$DATA_ROOT  CKPT_ROOT=$CKPT_ROOT  log=$LOG"

# step3 is mini-TRAIN setup. If a usable install already exists (checkpoints +
# nuScenes data + info pkls), do NOT re-run: section D re-points the repo symlinks
# and can relink a trainval setup back to mini, and B/C would prompt for uploads
# you already have ("waits for map expansion"). Bail with guidance instead.
# Force a full re-run with FORCE_STEP3=1.
# The pkl guard checks the LAST artifact section E produces (the vad pkl), not the
# first (nuscenes_infos_val.pkl): a run that made the basic pkl but crashed before
# the vad converter would otherwise look "done" and skip the rest.
if [ "${FORCE_STEP3:-0}" != "1" ] \
   && [ -d "$CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage1" ] \
   && { [ -d nuScenes/data/nuscenes/v1.0-trainval ] || [ -d nuScenes/data/nuscenes/v1.0-mini ]; } \
   && [ -f nuScenes/data/infos/nuscenes_infos_val.pkl ] \
   && [ -f nuScenes/data/infos/vad_nuscenes_infos_temporal_val.pkl ]; then
    echo "=== step3: install already present — nothing to do. ==="
    echo "  checkpoints:  $CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage1"
    echo "  nuScenes data + info pkls: present (symlinks under nuScenes/data/nuscenes)"
    echo "  step3 is mini-TRAIN setup; it is NOT needed for eval."
    echo "  -> to evaluate:            bash scripts/hpc/eval_stage2_2b_smoke.sh"
    echo "  -> to force a full re-run:  FORCE_STEP3=1 bash scripts/hpc/install_3_nuscenes_mini.sh"
    exit 0
fi

eval "$("$MM" shell hook --shell bash)"
micromamba activate "$ENV_NAME"
mkdir -p "$DATA_ROOT" "$CKPT_ROOT"

# --- download helpers for account-gated files -------------------------------
# _dl LINK DEST : download LINK to DEST (gdown for Google-Drive links, wget else).
_dl() {
    case "$1" in
        *drive.google.com*|*docs.google.com*) gdown --fuzzy "$1" -O "$2" || echo "  !! download failed — check the link and retry" ;;
        *) wget -O "$2" "$1" || echo "  !! download failed — check the link and retry" ;;
    esac
}
# fetch_gated NAME DEST IS_VALID URL HINT : ensure DEST exists and passes the
# IS_VALID test, fetching it FOR the user. Tries URL first (if non-empty, for
# non-interactive runs), then loops asking the user to PASTE a download link from
# their logged-in account — which we download. Empty input = "the file is already
# at DEST", so a manual upload still works as a fallback.
fetch_gated() {
    local name="$1" dest="$2" isvalid="$3" url="$4" hint="$5" link
    if eval "$isvalid"; then echo "  [$name] already present — skipping"; return 0; fi
    if [ -n "$url" ]; then echo "  [$name] trying preset URL..."; _dl "$url" "$dest"; fi
    while ! eval "$isvalid"; do
        cat <<EOF

  +-- $name — could not be fetched automatically ---------------------
  | Provide a working download LINK and THIS machine will fetch it (no scp).
  |   1. $hint
  |   2. Right-click its download button -> "Copy link address"
  |      (that signed link works with wget/gdown for a while).
  |   3. Paste it below and press ENTER.
  | (Already placed the file at $dest yourself? Just press ENTER.)
  +--------------------------------------------------------------------
EOF
        printf '  paste link (or ENTER if the file is already in place): '
        read -r link
        [ -n "$link" ] && { echo "  downloading..."; _dl "$link" "$dest"; }
    done
    echo "  [$name] OK -> $dest"
}

# --- A. checkpoints ---------------------------------------------------------
echo "=== A. checkpoints (Stage-1 VLM + Stage-2 model + OccVAE) ==="
VLM_DIR="$CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage1"
STAGE2_DIR="$CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage2"
OCCVAE="$CKPT_ROOT/occworld/occvae_latest.pth"
if [ ! -f "$VLM_DIR/config.json" ]; then
    hf download owl10/UniDriveVLA_Nusc_Base_Stage1 --local-dir "$VLM_DIR"
else
    echo "Stage-1 VLM already present — skipping"
fi
# Stage-2 is the released full model the multi-mode fine-tune loads (FT_LOAD_FROM
# in finetune_stage2_modes.sh). It is public on HF, so fetch it here rather than
# leaving a reproducer to discover the gap when training fails to find it.
if [ ! -f "$STAGE2_DIR/UniDriveVLA_Stage2_Nuscenes_2B.pt" ]; then
    hf download owl10/UniDriveVLA_Nusc_Base_Stage2 --local-dir "$STAGE2_DIR"
else
    echo "Stage-2 model already present — skipping"
fi
if [ ! -f "$OCCVAE" ]; then
    mkdir -p "$CKPT_ROOT/occworld"
    gdown "$OCCVAE_GDRIVE_ID" -O "$OCCVAE"
else
    echo "OccVAE already present — skipping"
fi

# --- B. nuScenes v1.0-mini --------------------------------------------------
echo "=== B. nuScenes v1.0-mini ==="
if [ ! -d "$DATA_ROOT/v1.0-mini" ]; then
    wget -c "$NUSCENES_MINI_URL" -O "$DATA_ROOT/v1.0-mini.tgz"
    tar -xzf "$DATA_ROOT/v1.0-mini.tgz" -C "$DATA_ROOT"
    echo "extracted v1.0-mini"
else
    echo "v1.0-mini already extracted — skipping"
fi
# Map expansion (vector maps): the converter reads maps/expansion/<loc>.json.
# Not listed in data_preparation.md but required by NuscMapExtractor. The public
# URL is gated (returns an HTML login page), so validate the zip and fall back
# to a manual upload, like gts.
if [ ! -d "$DATA_ROOT/maps/expansion" ]; then
    MAPZIP="$DATA_ROOT/map-expansion.zip"
    fetch_gated "nuScenes map expansion (nuScenes-map-expansion-v1.3.zip)" "$MAPZIP" \
        "python -c \"import zipfile,sys; sys.exit(0 if zipfile.is_zipfile('$MAPZIP') else 1)\" 2>/dev/null" \
        "$NUSCENES_MAP_URL" \
        "Open https://www.nuscenes.org/nuscenes#download and log in; find the 'Map expansion' pack (nuScenes-map-expansion-v1.3.zip)."
    python -m zipfile -e "$MAPZIP" "$DATA_ROOT/maps/"
    echo "extracted map expansion into $DATA_ROOT/maps/"
else
    echo "map expansion already present — skipping"
fi
# CAN bus expansion (ego pose/velocity): not in v1.0-mini.tgz, account-gated.
if [ ! -d "$DATA_ROOT/can_bus" ]; then
    CANZIP="$DATA_ROOT/can_bus.zip"
    fetch_gated "nuScenes CAN bus expansion (can_bus.zip)" "$CANZIP" \
        "python -c \"import zipfile,sys; sys.exit(0 if zipfile.is_zipfile('$CANZIP') else 1)\" 2>/dev/null" \
        "$NUSCENES_CANBUS_URL" \
        "Open https://www.nuscenes.org/nuscenes#download and log in; find the 'CAN bus expansion' (can_bus.zip)."
    python -m zipfile -e "$CANZIP" "$DATA_ROOT/"
    echo "extracted can_bus into $DATA_ROOT/"
else
    echo "can_bus already present — skipping"
fi

# --- C. Occ3D gts (occupancy GT for the train pipeline) ---------------------
echo "=== C. Occ3D gts -> $DATA_ROOT/gts ==="
GTS_TGZ="$DATA_ROOT/gts.tar.gz"
if [ -d "$DATA_ROOT/gts" ]; then
    echo "gts already present — skipping"
else
    # Occ3D gts is ONLY on Google Drive (no login-free mirror like nuScenes has),
    # and that file is frequently quota-blocked ("too many users"). Order of
    # attempts: an explicit mirror URL if you set GTS_URL (e.g. your own HF repo),
    # else gdown the canonical file id, else paste a link.
    fetch_gated "Occ3D gts.tar.gz" "$GTS_TGZ" \
        "python -c \"import tarfile,sys; sys.exit(0 if tarfile.is_tarfile('$GTS_TGZ') else 1)\" 2>/dev/null" \
        "${GTS_URL:-https://drive.google.com/uc?id=$GTS_FILE_ID}" \
        "Occ3D gts is Google-Drive only and often quota-blocked. Fixes: (a) wait a few hours and re-run (idempotent); (b) paste a fresh browser link from https://drive.google.com/file/d/$GTS_FILE_ID/view ; or (c) set GTS_URL to a direct-download mirror URL, e.g. an HF resolve link https://huggingface.co/datasets/<you>/<repo>/resolve/main/gts.tar.gz , and re-run."
    echo "found $(du -h "$GTS_TGZ" | cut -f1) at $GTS_TGZ — extracting..."
    tar -xzf "$GTS_TGZ" -C "$DATA_ROOT"   # gts.tar.gz has a gts/ root
    echo "extracted gts"
fi

# --- D. symlink data into the repo layout -----------------------------------
echo "=== D. symlink $DATA_ROOT -> nuScenes/data/nuscenes ==="
mkdir -p nuScenes/data/nuscenes
for d in can_bus maps samples sweeps v1.0-mini gts; do
    if [ -e "$DATA_ROOT/$d" ]; then
        ln -sfn "$DATA_ROOT/$d" "nuScenes/data/nuscenes/$d"; echo "  linked $d"
    else
        echo "  (missing $DATA_ROOT/$d — skipped)"
    fi
done

# --- E. mini info pkls ------------------------------------------------------
echo "=== E. generate v1.0-mini info pkls ==="
mkdir -p nuScenes/data/infos
# nuscenes_converter.py redirects basic infos to data/infos/mini/ for v1.0-mini
# (create_nuscenes_infos: out_path = out_path/'mini'). kmeans + the config want
# them in data/infos/, so lift any that a prior run left in mini/.
[ -d nuScenes/data/infos/mini ] && mv -f nuScenes/data/infos/mini/*.pkl nuScenes/data/infos/ 2>/dev/null || true
if [ -f nuScenes/data/infos/nuscenes_infos_train.pkl ] && \
   [ -f nuScenes/data/infos/vad_nuscenes_infos_temporal_train.pkl ]; then
    echo "info pkls already present — skipping (delete to regenerate)"
else
    pushd nuScenes >/dev/null
    export PYTHONPATH="$PWD:$PYTHONPATH"
    python tools/data_converter/nuscenes_converter.py nuscenes \
        --root-path ./data/nuscenes --canbus ./data/nuscenes \
        --out-dir ./data/infos/ --extra-tag nuscenes --version v1.0-mini
    python tools/data_converter/vad_nuscenes_converter.py nuscenes \
        --root-path ./data/nuscenes --canbus ./data/nuscenes \
        --out-dir ./data/infos/ --extra-tag vad_nuscenes --version v1.0-mini
    # lift the basic infos out of the mini/ subdir into data/infos/
    [ -d data/infos/mini ] && mv -f data/infos/mini/*.pkl data/infos/ 2>/dev/null || true
    popd >/dev/null
fi
ls -la nuScenes/data/infos/*.pkl 2>/dev/null

# --- F. K-means anchors -----------------------------------------------------
echo "=== F. K-means anchors (det/map/motion -> data/kmeans/*.npy) ==="
if [ -f nuScenes/data/kmeans/kmeans_det_900.npy ] && \
   [ -f nuScenes/data/kmeans/kmeans_map_100.npy ] && \
   [ -f nuScenes/data/kmeans/kmeans_motion_6.npy ]; then
    echo "kmeans anchors already present — skipping"
else
    pushd nuScenes >/dev/null
    export PYTHONPATH="$PWD:$PYTHONPATH"
    # Only the 3 anchors the Stage-1 config loads. kmeans_plan.py is skipped:
    # not referenced by the Stage-1 config, and it fails on v1.0-mini (no
    # planning trajectories to concatenate). Plan anchors are a Stage-2 concern.
    python tools/kmeans/kmeans_det.py
    python tools/kmeans/kmeans_map.py
    python tools/kmeans/kmeans_motion.py
    popd >/dev/null
fi
ls -la nuScenes/data/kmeans/*.npy 2>/dev/null

# --- G. print the smoke command ---------------------------------------------
# No separate mini config: the base config reads NUSC_VERSION, and the smoke
# script sets NUSC_VERSION=mini off-SLURM (ubix). Just run the smoke script.
echo ""
echo "=== STEP 3 DONE — log: $LOG ==="
cat <<EOF

Next — run the Stage-1 smoke (1 GPU, v1.0-mini). From the repo root on ubix:

  bash scripts/hpc/train_stage1_2b_smoke.sh

(The smoke script activates the env, picks GPU=1, sets NUSC_VERSION=mini off
SLURM, and points at the checkpoints downloaded above.)

Watch nuScenes/work_dirs/unidrivevla_stage1_2b_mini/logs/baseline/train-baseline-all-train.txt
With flash-attn + 96 GB this should sail past iter 0 -> iter 1 (the V100 wall).
EOF
