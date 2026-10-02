import os

# Eight-mode fine-tune on one GPU: N=8 modes, several epochs on a subset of trainval.
# (The thesis run uses the 8-GPU variant, unidrivevla_stage2_2b_modes8_ft_8gpu.py.)
# -----------------------------------------------------------------------------------
# THE REAL CONSTRAINT IS GPU-DAYS, NOT EPOCHS. A full nuScenes-trainval epoch is ~28k
# iters; at ~50 s/iter (N=8 -> 1+N=9 transformer forwards/iter) that is ~23 days PER
# EPOCH on one GPU. So the budget is a fixed number of gradient STEPS (max_iters), and
# epochs = max_iters / max_samples.
#
# FAVOUR DATA DIVERSITY OVER EPOCHS. For a fixed step budget, seeing more UNIQUE scenes
# generalises better than revisiting fewer -- especially for the mode embeddings, which
# need diverse scenes to learn broadly-useful behaviours; and WTA specialisation depends
# on total steps (~hundreds-thousands), not on epoch count. So set max_iters from the
# GPU-time you have, then make max_samples as LARGE as that allows.
#
# Defaults below: 5000 scenes x 2 epochs = 10000 iters (~6 days on one GPU). Lower
# max_samples for more epochs at the same cost. Only the ~5M-param action head +
# 8 mode embeddings train; the released backbone stays frozen.
_base_ = ["./unidrivevla_stage2_2b_modes_ft.py"]

load_from = os.environ.get("FT_LOAD_FROM", None)

# Distributed backend: nccl by default, but a single-GPU run can set DIST_BACKEND=gloo
# because NCCL 2.26 (torch 2.7/cu128) segfaults at DeepSpeed's init broadcast on some
# virtualized cloud A100s (bare dist.broadcast crashes; gloo works). The launcher
# auto-selects gloo when NUM_GPUS=1. Overrides the base config's dist_params.
dist_params = dict(backend=os.environ.get("DIST_BACKEND", "nccl"))

# N=8 modes (was 6). 1+N=9 forwards/iter. Keep the milder levers from the v1 FT.
model = dict(planning_head=dict(num_modes=8, mode_eps_share=0.05, mode_init_std=0.08))

# max_samples truncates trainval to the first S scenes (token-keyed, consistent);
# IterBasedRunner loops them. epochs = max_iters / max_samples. Default 5000 x 2 epochs.
data = dict(workers_per_gpu=0, train=dict(max_samples=5000))
runner = dict(type="IterBasedRunner", max_iters=10000)

# Cosine over the 10000-iter budget; short warmup so most steps run at useful lr.
lr_config = dict(warmup_iters=300)

# Checkpoint every 1000, keep only 2. Each DeepSpeed checkpoint is ~12 GB (it saves the
# whole model+optimizer even though only ~5M params train), so 5 of them = 60 GB, which
# filled the shared /home disk mid-write and killed the first run at iter 1500. keep=2
# caps it at ~24 GB; write the work_dir to /mnt/shared (space) rather than /home.
checkpoint_config = dict(interval=1000, max_keep_ckpts=2)
