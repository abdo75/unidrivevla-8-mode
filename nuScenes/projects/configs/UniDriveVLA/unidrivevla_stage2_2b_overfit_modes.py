import os

# Increment 2 — OVERFIT divergence test (NOT the smoke, NOT for results).
# ----------------------------------------------------------------------
# Goal: prove (or disprove) that the mode-conditioned flow + winner-take-all
# mechanism makes the N modes SPECIALIZE. The 20-iter smoke validated the
# plumbing but ran 20 *distinct* mini scenes, far too few steps for WTA
# specialization to emerge, with a tiny slow-moving mode lever -> modes stayed
# collapsed (mode_endpoint_std ~6 mm, winner_min == mean_all).
#
# This config gives the mode lever real authority and overfits a FIXED handful
# of samples for many steps, so divergence (if the mechanism works) becomes
# visible:
#   * mode_init_std 0.02 -> 0.2   (modes start differentiated, not identical)
#   * mode_query lr_mult ~50x     (the 6k-param lever moves fast vs the 5.28M shared params)
#   * mode_eps_share 0.05 -> 0.0  (pure winner-take-all; no anti-diversity pull)
#   * data.train.max_samples = 4  (fixed few samples, looped by IterBasedRunner)
#   * 150 iters, short warmup, fixed lr
#
# WATCH in the train log:
#   planning.mode_endpoint_std       -> should climb well above ~0.1 m
#   planning.mode_mean_all - mode_winner_min -> the per-sample loss SPREAD the
#                                               modes create; should grow (was ~0).
# If both grow -> Increment 2 mechanism PROVEN. If they stay flat even here ->
# the mechanism is too weak as designed and needs a redesign (stronger
# conditioning / explicit diversity term), not just more training.
_base_ = ["./unidrivevla_stage2_2b_ftsmoke.py"]

load_from = os.environ.get("FT_LOAD_FROM", None)

# Strong mode levers (override the smoke's num_modes=6, mode_eps_share=0.05).
model = dict(planning_head=dict(
    num_modes=6,
    mode_eps_share=0.0,
    mode_init_std=0.2,
))

# Fixed 4-sample overfit set (looped by the IterBasedRunner's infinite sampler).
# workers_per_gpu=0: load in the main process. With only 4 samples cycled forever
# by the infinite group sampler, prefetch workers add RAM + a known stall risk
# (the 1st overfit attempt ran 10 clean iters then hung 28 min inside a forward
# and was SIGKILLed -9 — a host-side kill, not a model error). Synchronous loading
# of 4 cached samples is cheap and removes both the workers and their RAM.
data = dict(workers_per_gpu=0, train=dict(max_samples=4))

# Give the per-mode embedding a 50x learning-rate multiplier so it can actually
# move relative to the dominant shared action-head params. Merges into the base
# custom_keys (the vlm key is preserved by mmcv's recursive dict merge).
optimizer = dict(paramwise_cfg=dict(custom_keys={
    'planning_head.mode_query': dict(lr_mult=50.0, decay_mult=1.0),
}))

# Short warmup + fixed lr: 150 iters is well inside the base's 500-iter warmup,
# which would otherwise keep lr crawling. Hold lr steady so the overfit can move.
# _delete_=True REPLACES the base lr_config rather than merging it — otherwise the
# base's min_lr_ratio leaks in and FixedLrUpdaterHook rejects that kwarg.
lr_config = dict(_delete_=True, policy="fixed", warmup="linear", warmup_iters=10, warmup_ratio=1.0 / 3)

runner = dict(type="IterBasedRunner", max_iters=150)
checkpoint_config = dict(interval=150, max_keep_ckpts=1)
log_config = dict(interval=1, hooks=[dict(type="TextLoggerHook")])
evaluation = dict(interval=100000)