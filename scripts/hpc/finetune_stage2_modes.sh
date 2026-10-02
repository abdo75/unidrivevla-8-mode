#!/bin/bash -l
#
# Stage-2 REAL multi-mode fine-tune (DrivoR thesis, Increment 2 — deployable model).
# --------------------------------------------------------------------------------
# What: freeze the backbone + perception, train ONLY the action-side planning head
#       + the 6 mode-query embeddings (mode-conditioned flow + winner-take-all) on
#       FULL nuScenes trainval, for a partial epoch (3000 iters ~= 2.5 days). This
#       produces the deployable multi-mode model that feeds Stage B -> NavSim oracle
#       ceiling on navhard. NOT a smoke / overfit (those are *_smoke / *_overfit).
# How:  1 GPU on ubix (Blackwell 96 GB). Mirrors finetune_stage2_smoke.sh, but:
#         - config = unidrivevla_stage2_2b_modes_ft.py (inherits BASE, real schedule)
#         - NUSC_VERSION=trainval (full data)
#         - RESUMABLE: does NOT wipe work_dir. Re-running after a kill auto-resumes
#           from the latest checkpoint (mmdet_train.py auto-fills resume_from). The
#           torch-2.7 weights_only resume bug is fixed (mmdet_train.py
#           _resume_with_trusted_load), so a preempted run can be picked back up.
#
# !! RUN UNDER tmux !! A dropped SSH has killed a multi-day run before:
#     tmux new -s modeft       # then run this script inside; detach with Ctrl-b d
#     tmux attach -t modeft    # to reattach later
#
set -eo pipefail
MM="$(command -v micromamba || echo "$HOME/.local/bin/micromamba")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
eval "$("$MM" shell hook --shell bash)"
micromamba activate unidrivevla

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT/nuScenes"

# checkpoints: existence probe (same convention as eval_stage2_2b_smoke.sh)
CKPT_ROOT="${CKPT_ROOT:-}"
if [ -z "$CKPT_ROOT" ]; then
    for c in "/mnt/shared/$USER/UniDriveVLA/checkpoints" "/mnt/shared/UniDriveVLA/checkpoints" "$HOME/UniDriveVLA/checkpoints" "$PROJECT_ROOT/checkpoints"; do
        [ -d "$c/UniDriveVLA_Nusc_Base_Stage1" ] && { CKPT_ROOT="$c"; break; }
    done
fi
[ -n "$CKPT_ROOT" ] || { echo "!! CKPT_ROOT not found (UniDriveVLA_Nusc_Base_Stage1)"; exit 1; }

export VLM_PRETRAINED_PATH="$CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage1"
export OCCWORLD_VAE_PATH="$CKPT_ROOT/occworld/occvae_latest.pth"
export DEEPSPEED_CONFIG="$PROJECT_ROOT/nuScenes/zero_configs/adam_zero1_bf16.json"
export FT_LOAD_FROM="$CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage2/UniDriveVLA_Stage2_Nuscenes_2B.pt"
export NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
# --- NCCL on GCP A3-Ultra disk images ---------------------------------------
# The GCP DLVM disk ships NCCL "gIB" net + "tuner_tcpx" tuner plugins (in
# /usr/local/gib, with an A3-Ultra fabric config forced via NCCL_NET=gIB /
# NCCL_TUNER_CONFIG_PATH). On non-A3 hardware they abort: 8-GPU NCCL init fails
# ("network gIB not found" / "No NCCL_TUNER_CONFIG_PATH"), and even a 1-GPU
# broadcast segfaulted (that was the real cause, not "NCCL 2.26" as first thought).
# We run single-node, so force vanilla NCCL — built-in Socket net, no external
# net/tuner plugins. Harmless off GCP: single-node comm is NVLink/P2P regardless;
# the net transport only matters cross-node.
export NCCL_NET=Socket
export NCCL_NET_PLUGIN=none
export NCCL_TUNER_PLUGIN=none
unset NCCL_TUNER_CONFIG_PATH NCCL_NET_GDR_LEVEL

# Backend: nccl for real multi-GPU (now that the gIB plugins above are neutralised);
# gloo for single-GPU (no real cross-rank comm, and it sidesteps NCCL init as an
# extra safety net). Override by exporting DIST_BACKEND explicitly.
export DIST_BACKEND="${DIST_BACKEND:-$([ "$NUM_GPUS" -eq 1 ] && echo gloo || echo nccl)}"
export TORCH_DIST_TIMEOUT=14400
export MLP_WORKER_0_PORT="${MLP_WORKER_0_PORT:-28600}"
# reduce fragmentation on a shared GPU (does NOT create free VRAM — you still need
# ~30-40 GB free; this run OOMs if other users are occupying the card)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Full nuScenes trainval (NOT mini): the real fine-tune needs diverse real scenes.
# Annotation load is ~20-40 GB system RAM — make sure the node has it free.
export NUSC_VERSION="${NUSC_VERSION:-trainval}"

# Default = the v1 6-mode FT; override CFG for the 8-mode >=5-epoch run, e.g.
#   CFG=projects/configs/UniDriveVLA/unidrivevla_stage2_2b_modes8_ft.py EXP_NAME=modes8 ...
CFG="${CFG:-projects/configs/UniDriveVLA/unidrivevla_stage2_2b_modes_ft.py}"
EXP_NAME="${EXP_NAME:-modes_ft_stage2_${NUSC_VERSION}}"
WORK_DIR="work_dirs/${EXP_NAME}"
# dist_train.sh redirects ALL training output here (not to this script's stdout):
TRAIN_LOG="${WORK_DIR}/logs/baseline/train-baseline-all-train.txt"

echo "=== Stage-2 multi-mode fine-tune (REAL) — $(date -Is) on $(hostname -s) ==="
echo "  CKPT_ROOT=$CKPT_ROOT  NUSC_VERSION=$NUSC_VERSION  GPUS=$NUM_GPUS"
echo "  config=$CFG  (num_modes=6, mode_init_std=0.08, lr_mult=5x, max_iters=3000)"
echo "  load_from=$FT_LOAD_FROM"
echo "  work_dir=$PROJECT_ROOT/nuScenes/$WORK_DIR"
echo "  TRAINING LOG (watch this): $PROJECT_ROOT/nuScenes/$TRAIN_LOG"
echo "  tail it from another shell:  tail -f $PROJECT_ROOT/nuScenes/$TRAIN_LOG"
echo "  watch for:  planning.mode_endpoint_std climbing (modes diverge),"
echo "              and (mode_mean_all - mode_winner_min) growing (WTA specializing)."

# RESUMABLE (unlike the smoke, which wipes): if a checkpoint already exists in the
# work_dir, mmdet_train.py auto-resume picks up the latest and CONTINUES training
# (it does NOT restart from FT_LOAD_FROM). This is what we want after a preemption.
# To force a fresh start instead (e.g. after changing levers), delete the work_dir
# manually first:  rm -rf "$WORK_DIR"
if [ -d "$WORK_DIR" ] && ls "$WORK_DIR"/iter_* >/dev/null 2>&1; then
    echo "  [resume] existing checkpoints found in $WORK_DIR -> will AUTO-RESUME and continue."
    echo "           (delete the work_dir first if you intended a fresh restart.)"
else
    echo "  [fresh]  no checkpoints in $WORK_DIR -> fresh start from FT_LOAD_FROM."
fi

# --no-validate: the metric we care about is NavSim PDMS (offline, Stage B/C), not
# nuScenes val L2 here; skipping eval saves RAM + time on the long run.
bash tools/dist_train.sh "$CFG" "$NUM_GPUS" "$EXP_NAME" --no-validate
echo "=== launcher returned; training log: $PROJECT_ROOT/nuScenes/$TRAIN_LOG ==="