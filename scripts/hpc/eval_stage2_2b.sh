#!/bin/bash -l
#
# Stage-2 evaluation (PROD)
# -------------------------
# What:  end-to-end nuScenes val eval (L2, collision, perception mAP) on the
#        published Stage-2 2B checkpoint, distributed across 4 GPUs.
# Where: 4 x V100-32GB (-C volta32), 2-day wall, normal QoS.
# Note:  there is only ONE eval pipeline in this repo. Stage-1 has no
#        end-to-end metrics; Stage-2 eval covers everything.
# Resume: writes per-rank partials to projects/work_dirs/.../resume_state/.
#         Re-running this script picks up where the last run left off.
#
#SBATCH -J unidrivevla_eval
#SBATCH --mail-type=start,end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH -p gpu
#SBATCH -C volta32
#SBATCH --qos=normal
#SBATCH -o slurm-eval-%j.out

set -eo pipefail

eval "$(micromamba shell hook --shell bash)"
micromamba activate unidrivevla

PROJECT_ROOT=$HOME/UniDriveVLA
cd "$PROJECT_ROOT/nuScenes"

export VLM_PRETRAINED_PATH="$PROJECT_ROOT/checkpoints/UniDriveVLA_Nusc_Base_Stage1"
export OCCWORLD_VAE_PATH="$PROJECT_ROOT/checkpoints/occworld/occvae_latest.pth"
export TORCH_DIST_TIMEOUT=14400

CFG=projects/configs/UniDriveVLA/unidrivevla_stage2_2b.py
CKPT="$PROJECT_ROOT/checkpoints/UniDriveVLA_Nusc_Base_Stage2/UniDriveVLA_Stage2_Nuscenes_2B.pt"

# Stable per-rank partial-results dir: survives SLURM timeouts so the next
# launch of this script picks up where the previous one left off. Lives under
# work_dirs/<cfg>/resume_state/ (the dir dist_eval.sh already derives for logs)
# and is removed by rank 0 on clean completion.
WORK_DIR="projects/work_dirs/$(basename "${CFG%.*}")"
RESUME_DIR="$WORK_DIR/resume_state"
mkdir -p "$RESUME_DIR"

echo "============================================"
echo "  Host:        $(hostname)"
echo "  GPUs:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  Config:      $CFG"
echo "  Checkpoint:  $CKPT"
echo "  VLM weights: $VLM_PRETRAINED_PATH"
echo "  OccVAE:      $OCCWORLD_VAE_PATH"
echo "  Resume dir:  $RESUME_DIR"
echo "============================================"

bash tools/dist_eval.sh "$CFG" "$CKPT" "${GPUS:-4}" --resume-dir "$RESUME_DIR"
