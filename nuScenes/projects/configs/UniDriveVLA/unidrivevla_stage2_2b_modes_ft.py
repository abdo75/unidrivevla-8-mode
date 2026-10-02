import os

# Increment 2 — REAL multi-mode fine-tune (mode-conditioned flow + winner-take-all).
# -----------------------------------------------------------------------------------
# This is the DEPLOYABLE multi-mode model, NOT a smoke / overfit throwaway.
# It inherits the BASE Stage-2 config (full nuScenes trainval data + the real
# optimizer/schedule), unlike *_ftsmoke.py (mini, 20 iters) and *_overfit_modes.py
# (4 fixed samples, aggressive levers). The mechanism is already PROVEN on the
# overfit (mode_endpoint_std 0.05 -> 0.35 m); the goal here is to train the
# action-side head + N mode embeddings on diverse real scenes so the modes are
# behaviourally useful, then feed Stage B -> NavSim oracle ceiling on navhard.
#
# Levers are MILDER than the overfit (which used std 0.2, lr_mult 50x, eps 0 to
# force divergence fast on 4 samples). On diverse data we want stable, useful
# specialization, not maximal spread:
#   mode_init_std  0.2  -> 0.08   (modes start mildly differentiated)
#   mode_query lr  50x  -> 5x     (the 6k-param lever moves, but doesn't dominate)
#   mode_eps_share 0.0  -> 0.05   (small share to all modes -> no dead modes)
#
# Backbone + perception stay FROZEN (freeze_except below): we train only the
# ~5M-param action-side head plus the 6 mode embeddings.
#
# COST: the mode-loop is 1+N=7 transformer forwards/iter (~72 s/iter measured on
# the overfit). max_iters=3000 ~= 2.5 days on one Blackwell GPU. This is a
# PARTIAL epoch (a full trainval epoch on 1 GPU is ~28k iters ~= 23 days, which is
# infeasible) — a thin frozen-backbone head does not need a full epoch. Checkpoints
# every 500 iters let us run the NavSim oracle on intermediate ckpts, pick the
# best, and resume-extend toward ~5000 if headroom is still climbing.
_base_ = ["./unidrivevla_stage2_2b.py"]

# Fine-tune the released Stage-2 checkpoint (override the base's Stage-1 load_from).
# The run script exports FT_LOAD_FROM = path to the Stage-2 .pt. Loaded strict=False,
# so the new mode_query parameter is tolerated and keeps its fresh init.
load_from = os.environ.get("FT_LOAD_FROM", None)

# N learned mode-query embeddings -> N candidate trajectories, milder levers.
model = dict(planning_head=dict(
    num_modes=6,
    mode_eps_share=0.05,
    mode_init_std=0.08,
))

# Train ONLY these (substring match in parameter names); freeze everything else.
# tools/train.py applies this via cfg.freeze_except, before the optimizer is built.
freeze_except = [
    "planning_head.action_in_proj",
    "planning_head.action_out_proj",
    "planning_head.action_time_mlp",
    "planning_head.status_mlp",
    "planning_head.hist_traj_encoder",
    "planning_head.mode_query",  # the per-mode query embeddings
]

# Give the per-mode embedding a MILD 5x learning-rate multiplier (vs the overfit's
# 50x). mmcv merges this into the base paramwise_cfg.custom_keys recursively, so
# the base's vlm lr_mult key is preserved (the vlm is frozen anyway, but the merge
# must not drop the key). Base optimizer type/lr (AdamW, 1e-4)/wd are untouched.
optimizer = dict(paramwise_cfg=dict(custom_keys={
    'planning_head.mode_query': dict(lr_mult=5.0, decay_mult=1.0),
}))

# Keep the base CosineAnnealing policy (no _delete_ needed — same policy, so
# min_lr_ratio etc. merge cleanly). Only shorten the warmup so a 3000-iter run
# spends most of its budget at useful lr rather than crawling through the base's
# 500-iter warmup. Cosine anneals over runner.max_iters automatically.
lr_config = dict(warmup_iters=300)

# workers_per_gpu=0 (main-process loading, NO prefetch fork). The first attempt
# (2026-06-17, workers=4) ran fine on the GPU — 39 s/iter, 31 GB VRAM — but was
# SIGKILL'd (-9, system-RAM OOM-killer, no py traceback) at ~iter 37. Cause: the
# full-trainval annotation pickle expands to ~20-40 GB of Python objects in the
# main process; each forked dataloader worker's refcounting defeats copy-on-write,
# multiplying system RAM ~4-5x on the shared node. With workers=0 there is no
# fork-multiply. Throughput cost is ~nil: data_time was 0.052 s vs 39 s of compute,
# so synchronous loading hides completely. (Same fix the overfit run needed.)
data = dict(workers_per_gpu=0)

# Partial-epoch fine-tune: 3000 iters (~2.5 days). Resumable (the run script does
# NOT wipe the work_dir, unlike the smoke), and the torch-2.7 weights_only resume
# bug is fixed in mmdet_train.py so a preempted run can be picked back up.
runner = dict(type="IterBasedRunner", max_iters=3000)

# Checkpoint every 250 iters, keep the last 5 (merges with base -> deepspeed=True
# preserved). Tightened from 500 after the 2026-06-17 run was OOM-killed (shared-
# node system-RAM contention) ~iter 837, losing the 337 iters since the iter_500
# save. At ~40 s/iter, 250 iters ~= 2.8 h of rework worst-case. The run is resumable
# (mmdet_train.py auto-resume + the weights_only resume fix), so on a contended node
# it can finish across several resume cycles, each losing at most one interval.
checkpoint_config = dict(interval=250, max_keep_ckpts=5)

# Log every 25 iters: ~120 log lines over the run, frequent enough to watch
# planning.mode_endpoint_std / (mode_mean_all - mode_winner_min) climb, sparse
# enough not to flood the log over 2.5 days.
log_config = dict(interval=25, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")])

# Skip in-training eval: the metric we care about is NavSim PDMS (offline, via
# Stage B/C), not nuScenes val L2 here, and building the val set adds RAM/time.
# The run script also passes --no-validate; this is a belt-and-braces guard.
evaluation = dict(interval=100000)