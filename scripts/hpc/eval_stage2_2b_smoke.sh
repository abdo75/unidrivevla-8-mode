#!/bin/bash -l
#
# Stage-2 evaluation (SMOKE) — host-agnostic
# ------------------------------------------
# What:  open-loop planning eval of the released Stage-2 checkpoint, 1 GPU.
# How:   1 GPU, interactive. Works on both clusters:
#   - iris (V100-32GB): inside an salloc allocation (do NOT sbatch):
#       salloc -p gpu --qos=normal --gres=gpu:1 -C volta32 -t 0-04:00:00 \
#              --mem=100G --cpus-per-task=7
#       bash scripts/hpc/eval_stage2_2b_smoke.sh
#   - ubix (Blackwell, 96 GB): GPU always available; just run it — off-SLURM the
#     script sets NUSC_VERSION=mini and evaluates the v1.0-mini val split:
#       bash scripts/hpc/eval_stage2_2b_smoke.sh
#
# Notes:
#   - Downloads the released Stage-2 checkpoint from HF if not already present.
#   - STAGE1_CHECKPOINT is NOT needed for eval (it only affects training's
#     load_from). VLM weights + OccVAE must exist (run install_3_nuscenes_mini.sh
#     on ubix, or have them from the iris setup).
#   - dist_eval.sh tees output to the screen and work_dirs/.../logs/eval.<ts>.
#   - Self-heals the planning seg GT: symlinks planing_gt_segmentation_val into
#     data/infos if staged on the shared disk, else prints the relay command and
#     runs without the vehicle-only (UniAD/STP-3) planning tables. Non-blocking.

set -eo pipefail
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

# Capture the whole run (banner + download + eval output) to a top-level log,
# like the bootstrap scripts. dist_eval.sh tees to stdout, so this catches it.
LOG="$(pwd)/eval_stage2_2b_smoke_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "logging to $LOG"

# micromamba: on PATH (iris) or rootless install (ubix).
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
eval "$("$MM" shell hook --shell bash)"
micromamba activate unidrivevla

# Checkpoint root: probe by EXISTENCE of the Stage-1 VLM dir, not writability
# (writability lies after a group-perm change). CKPT_ROOT / UNIDRIVEVLA_BASE are
# honored but STILL VALIDATED — a stale override must not silently win; if it
# doesn't actually contain the VLM dir we fall through to the probe. On failure we
# print what each candidate actually holds, so "not found" is diagnosable.
ckpt_cands=()
[ -n "${CKPT_ROOT:-}" ]        && ckpt_cands+=("$CKPT_ROOT")
[ -n "${UNIDRIVEVLA_BASE:-}" ] && ckpt_cands+=("$UNIDRIVEVLA_BASE/UniDriveVLA/checkpoints")
ckpt_cands+=("/mnt/shared/$USER/UniDriveVLA/checkpoints" \
             "/mnt/shared/UniDriveVLA/checkpoints" \
             "$HOME/UniDriveVLA/checkpoints")
CKPT_ROOT=""
for c in "${ckpt_cands[@]}"; do
    [ -d "$c/UniDriveVLA_Nusc_Base_Stage1" ] && { CKPT_ROOT="$c"; break; }
done
if [ -z "$CKPT_ROOT" ]; then
    echo "!! Stage-1 VLM dir (UniDriveVLA_Nusc_Base_Stage1) not found. Checked:"
    for c in "${ckpt_cands[@]}"; do
        if [ -d "$c/UniDriveVLA_Nusc_Base_Stage1" ]; then
            echo "   PRESENT  $c/UniDriveVLA_Nusc_Base_Stage1"
        elif [ -d "$c" ]; then
            echo "   missing  $c/UniDriveVLA_Nusc_Base_Stage1  (parent exists — contents:)"
            ls -1 "$c" 2>&1 | sed 's/^/              /' | head -20
        else
            echo "   missing  $c  (no such directory)"
        fi
    done
    echo "   -> set CKPT_ROOT=/dir-that-CONTAINS-UniDriveVLA_Nusc_Base_Stage1 and re-run."
    exit 1
fi
echo "CKPT_ROOT=$CKPT_ROOT"

PROJECT_ROOT=$HOME/UniDriveVLA
cd "$PROJECT_ROOT/nuScenes"

export VLM_PRETRAINED_PATH="$CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage1"
export OCCWORLD_VAE_PATH="$CKPT_ROOT/occworld/occvae_latest.pth"
export TORCH_DIST_TIMEOUT=14400

# Dataset split: artifact-driven default — pick trainval if the val pkl is at
# trainval size (~155 MB) rather than mini size (~3 MB). The pkl is what eval
# actually loads, so it can't disagree with reality even if you flip back and
# forth between mini/trainval bootstraps. Override with NUSC_VERSION=...
if [ -z "${NUSC_VERSION:-}" ]; then
    PKL=data/infos/nuscenes_infos_val.pkl
    if [ -f "$PKL" ] && [ "$(stat -c%s "$PKL")" -gt 50000000 ]; then
        NUSC_VERSION=trainval
    else
        NUSC_VERSION=mini
    fi
fi
export NUSC_VERSION

GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
export NUM_GPUS=$GPUS

# Stage-2 checkpoint — download from HF if missing (~6 GB, public repo).
STAGE2_DIR="$CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage2"
CKPT="$STAGE2_DIR/UniDriveVLA_Stage2_Nuscenes_2B.pt"
if [ ! -f "$CKPT" ]; then
    echo "Stage-2 checkpoint not found — downloading owl10/UniDriveVLA_Nusc_Base_Stage2 ..."
    hf download owl10/UniDriveVLA_Nusc_Base_Stage2 --local-dir "$STAGE2_DIR"
fi
[ -f "$CKPT" ] || { echo "!! Stage-2 checkpoint missing at $CKPT after download."; exit 1; }

# Planning seg GT (planing_gt_segmentation_val): the ONLY extra file eval needs
# beyond detection + STRICT planning. It unlocks the GPT-Driver vehicle-only
# (UniAD/STP-3) collision + L2 tables. NON-BLOCKING: symlink it in if staged on
# the shared disk, else print the relay command and run without those tables.
# Split-agnostic — the one val file serves both mini and trainval eval.
SEG_GT=planing_gt_segmentation_val
EVAL_GT_SRC="${EVAL_GT_SRC:-$(dirname "$CKPT_ROOT")/eval_gt}"
if [ ! -e "data/infos/$SEG_GT" ]; then
    seg_src=""
    for cand in "$EVAL_GT_SRC/$SEG_GT" \
                "/mnt/shared/$USER/UniDriveVLA/eval_gt/$SEG_GT" \
                "/mnt/shared/UniDriveVLA/eval_gt/$SEG_GT" \
                "$HOME/UniDriveVLA/eval_gt/$SEG_GT"; do
        [ -e "$cand" ] && { seg_src="$cand"; break; }
    done
    if [ -n "$seg_src" ]; then
        ln -sfn "$(readlink -f "$seg_src")" "data/infos/$SEG_GT"
        echo "planning seg GT: linked $SEG_GT <- $seg_src"
    else
        IRIS_SSH="${IRIS_SSH:-iris}"; UBIX_SSH="${UBIX_SSH:-ubix}"
        mkdir -p "$EVAL_GT_SRC"
        echo "planning seg GT: $SEG_GT not staged — vehicle-only (UniAD/STP-3) tables will be SKIPPED."
        echo "  (detection + STRICT planning are unaffected; this only adds those tables.)"
        echo "  iris<->ubix can't reach each other; relay through your computer, then re-run:"
        echo "    scp -3 $IRIS_SSH:/mnt/aiongpfs/users/$USER/UniDriveVLA/nuScenes/data/infos/$SEG_GT \\"
        echo "           $UBIX_SSH:$EVAL_GT_SRC/"
    fi
fi

CFG=projects/configs/UniDriveVLA/unidrivevla_stage2_2b.py

echo "============================================"
echo "  Host:        $(hostname)"
echo "  GPUs:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) (x$GPUS)"
echo "  Config:      $CFG"
echo "  NUSC_VERSION: $NUSC_VERSION"
echo "  Checkpoint:  $CKPT"
echo "  VLM weights: $VLM_PRETRAINED_PATH"
echo "  OccVAE:      $OCCWORLD_VAE_PATH"
echo "============================================"

bash tools/dist_eval.sh "$CFG" "$CKPT" "$GPUS"
