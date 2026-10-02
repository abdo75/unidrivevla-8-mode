import os

# Increment 1 — frozen-backbone fine-tune SMOKE.
# Validates that we can fine-tune ONLY the action-side planning head (backbone +
# perception frozen) end-to-end, for a handful of iterations. NOT for results;
# this is the "does the fine-tune harness run / overfit / not crash" gate before
# the N-mode + winner-take-all change (Increment 2).
_base_ = ["./unidrivevla_stage2_2b.py"]

# Fine-tune the released Stage-2 checkpoint (override the base's Stage-1 load_from).
# The run script exports FT_LOAD_FROM = path to the Stage-2 .pt.
load_from = os.environ.get("FT_LOAD_FROM", None)

# Increment 2 — mode-conditioned flow + winner-take-all. N learned mode-query
# embeddings -> N candidate trajectories. num_modes=1 (base default) is a strict
# no-op that preserves the released-checkpoint behavior; here we turn on N modes.
model = dict(planning_head=dict(num_modes=6, mode_eps_share=0.05))

# Train ONLY these (substring match in parameter names); freeze everything else.
# tools/train.py applies this via cfg.freeze_except, before the optimizer is built.
freeze_except = [
    "planning_head.action_in_proj",
    "planning_head.action_out_proj",
    "planning_head.action_time_mlp",
    "planning_head.status_mlp",
    "planning_head.hist_traj_encoder",
    "planning_head.mode_query",  # Increment 2: train the per-mode query embeddings
]

# Run on v1.0-mini (low system RAM) using infos in a SEPARATE dir so trainval
# infos are untouched. Generate them once with scripts/hpc/gen_mini_infos.sh,
# and run the smoke with NUSC_VERSION=mini (the run script defaults to that).
mini_infos = "data/infos_mini/"
data = dict(
    workers_per_gpu=2,  # fewer workers -> lower RAM than the base's 8
    train=dict(
        ann_file=mini_infos + "nuscenes_infos_train.pkl",
        vad_ann_file=mini_infos + "vad_nuscenes_infos_temporal_train.pkl",
    ),
    val=dict(
        ann_file=mini_infos + "nuscenes_infos_val.pkl",
        vad_ann_file=mini_infos + "vad_nuscenes_infos_temporal_val.pkl",
    ),
)

# Tiny run: 20 iterations, frequent logging, one checkpoint, no eval.
runner = dict(type="IterBasedRunner", max_iters=20)
checkpoint_config = dict(interval=20, max_keep_ckpts=1)
log_config = dict(interval=1, hooks=[dict(type="TextLoggerHook")])
evaluation = dict(interval=100000)  # skip eval during the smoke
