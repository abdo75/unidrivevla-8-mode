import os

# 8-GPU variant of unidrivevla_stage2_2b_modes8_ft.py (full-trainval convergence run).
# -----------------------------------------------------------------------------------
# Inherits EVERYTHING from the 8-mode fine-tune (N=8 modes, gloo/nccl dist_params,
# load_from, the milder mode levers) and only changes the DATA BUDGET for 8-way
# data-parallel.
#
# EPOCH MATH (correcting an earlier wrong comment that said "~2.3 epochs"):
# each runner *iter* consumes num_gpus * batch_size = 8 * 1 = 8 samples (grad-accum
# updates the optimizer every 4 iters but does NOT change samples/iter). So one epoch
# over the ~28k-scene trainval train split is  28130 / 8 = ~3516 iters. The first
# 8-GPU run (max_iters=2000) was therefore only ~0.57 epochs -- under one pass, which
# proves nothing. This run caps at ~5 epochs and we STOP AT THE PLATEAU of the
# held-out minADE_k curve (see scripts/hpc/eval_minade.sh), not a fixed length.
#
# Only the ~5M-param action head + 8 mode embeddings train; the backbone stays frozen.
#
# Launch:  CFG=projects/configs/UniDriveVLA/unidrivevla_stage2_2b_modes8_ft_8gpu.py \
#          EXP_NAME=modes8_ft_8gpu bash scripts/hpc/finetune_stage2_modes.sh
_base_ = ["./unidrivevla_stage2_2b_modes8_ft.py"]

load_from = os.environ.get("FT_LOAD_FROM", None)

# Full trainval instead of the 5000-scene 1-GPU subset (a cap above the split size
# means no truncation). Diversity over repetition, as the base config argues.
data = dict(train=dict(max_samples=50000))

# ~5-epoch CAP (5 * 3516). We monitor minADE_k on nuScenes val per checkpoint and stop
# once it flattens, rather than running the whole budget blindly.
runner = dict(type="IterBasedRunner", max_iters=17580)
lr_config = dict(warmup_iters=500)

# Checkpoint every 1000 iters and keep ALL of them -- each is a data-point on the
# minADE_k convergence curve, and they let us resume/pick the best. ~17 checkpoints x
# ~12 GB = ~210 GB; fits the 1 TB disk with the ~500 GB of data. If disk gets tight,
# lower max_keep_ckpts or offload older checkpoints to a bucket.
checkpoint_config = dict(interval=1000, max_keep_ckpts=-1)
