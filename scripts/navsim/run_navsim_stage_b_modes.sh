#!/bin/bash
#
# Stage B MODES (Increment 2) — FULL run. N candidates/token from the LEARNED modes
# (return_all_modes=True) on the multi-mode overfit config + checkpoint. Zero flags.
# Smoke (5 tokens) = run_navsim_stage_b_modes_smoke.sh. Output feeds the oracle ceiling.
#
# Pre: Stage A produced .npz under exp/uni_inputs/<split>. Runs in the unidrivevla env.
#
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$(dirname "${BASH_SOURCE[0]}")/navsim_paths.sh" 2>/dev/null || true
NAVSIM_WS="${NAVSIM_WS:-$HOME/navsim_ws}"
SPLIT="${TRAIN_TEST_SPLIT:-warmup_two_stage}"

export MODE_CANDIDATES=1
# Multi-mode fine-tune used in the thesis: N=8 config + the iter-7000 checkpoint.
# CKPT is either a flat .pt (e.g. the released delta merged by
# scripts/hpc/apply_trainable_delta.py) or a DeepSpeed iter dir from your own fine-tune
# (Stage B walks it for mp_rank_00_model_states.pt).
export CONFIG="${CONFIG:-$PROJECT_ROOT/nuScenes/projects/configs/UniDriveVLA/unidrivevla_stage2_2b_modes8_ft.py}"
export CKPT="${CKPT:-$PROJECT_ROOT/checkpoints/UniDriveVLA_NAVSIM_modes8_iter7000.pt}"
# The modes8_ft config references FT_LOAD_FROM at build time; point it at the released .pt.
export FT_LOAD_FROM="${FT_LOAD_FROM:-$PROJECT_ROOT/checkpoints/UniDriveVLA_Nusc_Base_Stage2/UniDriveVLA_Stage2_Nuscenes_2B.pt}"
# Dedicated modes output dir (the oracle ceiling reads this by default).
export STAGE_B_OUT="${STAGE_B_OUT:-$NAVSIM_WS/exp/uni_trajectories/${SPLIT}_modes}"

exec bash "$HERE/run_navsim_stage_b.sh"