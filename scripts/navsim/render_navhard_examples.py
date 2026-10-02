#!/usr/bin/env python
"""
Render BEV examples of navhard scenes with UniDriveVLA's N candidates overlaid on the
REAL scene (map + agents), for the good/bad reels the thesis needs.

For each token it draws the devkit BEV (plot_bev_frame: drivable area, lanes, agents,
ego) and overlays, ALWAYS the same four things so panels read identically:
  - the 8 candidate trajectories       faint grey
  - the ground-truth human future      black          (the reference)
  - the first trajectory (mode 0)      blue           (deploys with no scorer)
  - the scorer's choice                green, dashed  (what the learned scorer selects)
No oracle line (that's a table number, not a per-scene visual) and no score text.
One PNG per token; per-category PNGs are stitched into an mp4.

Env (navsim env; source navsim_env.sh first):
  STAGE_B_TRAJ   .../navhard_two_stage_modes/trajectories.pkl   (the candidates to draw)
  EXAMPLES_JSON  find_examples.py output {category: [tokens]}
  SELECTION      scorer selection.json (optional; for the green scorer's-choice line)
  OUT_DIR        where PNGs/mp4 land (default ./navhard_examples)
  CATEGORIES     comma list (default: first_traj_offroad,scorer_better,scorer_worse,first_traj_good)
  PER_CAT        tokens per category (default 12)   FRAME_IDX (default 3)   FPS (default 2)
"""
import glob
import json
import os
import os.path as osp
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.visualization.plots import plot_bev_frame
from navsim.visualization.bev import add_trajectory_to_bev_ax

DEVKIT = os.environ["NAVSIM_DEVKIT_ROOT"]
DATA = os.environ["OPENSCENE_DATA_ROOT"]
SPLIT = os.environ.get("TRAIN_TEST_SPLIT", "navhard_two_stage")
TS = f"{DATA}/{SPLIT}"
CONFIG_DIR = f"{DEVKIT}/navsim/planning/script/config/pdm_scoring"
LATERAL_SIGN = float(os.environ.get("LATERAL_SIGN", "-1"))
N_OUT = 8
FRAME_IDX = int(os.environ.get("FRAME_IDX", "3"))   # current observation frame (4 history)


def to_traj(uni):
    """Stage C's (6,2) (lateral,forward) -> NavSim Trajectory poses (8,3)[x,y,heading]."""
    uni = np.asarray(uni, dtype=np.float64)
    xy = np.stack([uni[:, 1], LATERAL_SIGN * uni[:, 0]], axis=1)
    v = xy[-1] - xy[-2]
    while xy.shape[0] < N_OUT:
        xy = np.concatenate([xy, (xy[-1] + v)[None]], axis=0)
    prev = np.concatenate([np.zeros((1, 2)), xy[:-1]], axis=0)
    d = xy - prev
    heading = np.arctan2(d[:, 1], d[:, 0])
    return Trajectory(np.concatenate([xy, heading[:, None]], axis=1).astype(np.float32))


def line_cfg(color, lw=3.0, alpha=0.95, ls="-", z=5):
    return dict(line_color=color, line_color_alpha=alpha, line_width=lw, line_style=ls,
                marker="o", marker_size=0, marker_edge_color=color, zorder=z)


def build_loader():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default_run_pdm_score", overrides=[
            f"train_test_split={SPLIT}",
            f"synthetic_sensor_path={TS}/sensor_blobs",
            f"synthetic_scenes_path={TS}/synthetic_scene_pickles",
        ])
    sf = instantiate(cfg.train_test_split.scene_filter)
    return SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=sf,
        sensor_config=SensorConfig.build_no_sensors(),
    )


def _legend(ax, has_scorer, has_gt):
    from matplotlib.lines import Line2D
    # ALWAYS the same set, in this order, so every figure reads identically.
    handles = [Line2D([0], [0], color="0.6", lw=1.0, label="candidates (8 modes)"),
               Line2D([0], [0], color="tab:blue", lw=2, label="first trajectory")]
    if has_scorer:
        handles.append(Line2D([0], [0], color="tab:green", lw=2, ls="--", label="scorer's choice"))
    if has_gt:
        handles.append(Line2D([0], [0], color="black", lw=2, label="ground truth"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)


def gt_frames(scene):
    """How many GT future poses are available (0 = synthetic stage-2 scene, no human future),
    capped to the candidate horizon so GT is the same duration as the drawn candidates."""
    try:
        return max(0, min(N_OUT, len(scene.frames) - scene.scene_metadata.num_history_frames))
    except Exception:  # noqa: BLE001
        return 0


def render_token(scene, cand_trajs, pick, out_png, n_gt):
    """Four things, always drawn the same way: the 8 candidates (grey), the ground-truth
    human path (black), the FIRST trajectory (mode 0, blue) = what deploys with no scorer,
    and the SCORER'S CHOICE (green dashed) = what the learned scorer selects. Dashed so it
    stays visible even when the scorer agrees with the first trajectory. No oracle line
    (that's a table number, not a per-scene visual) and no score text."""
    fig, ax = plot_bev_frame(scene, FRAME_IDX)
    N = cand_trajs.shape[0]
    for k in range(N):                                   # all candidates, faint grey
        add_trajectory_to_bev_ax(ax, to_traj(cand_trajs[k]), line_cfg("0.6", lw=1.0, alpha=0.5, z=3))
    if n_gt > 0:                                         # ground-truth human future (black)
        add_trajectory_to_bev_ax(ax, scene.get_future_trajectory(n_gt), line_cfg("black", lw=2.0, z=4))
    add_trajectory_to_bev_ax(ax, to_traj(cand_trajs[0]), line_cfg("tab:blue", lw=2.0, z=5))
    if pick is not None:                                 # scorer's choice (green dashed)
        add_trajectory_to_bev_ax(ax, to_traj(cand_trajs[pick]), line_cfg("tab:green", lw=2.0, ls="--", z=6))
    _legend(ax, has_scorer=(pick is not None), has_gt=(n_gt > 0))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def stitch(pngs, mp4, fps):
    if not pngs:
        return
    try:
        import cv2
    except ImportError:
        print(f"  (cv2 not in this env — PNGs written, skipping {osp.basename(mp4)}; "
              f"stitch later with: ffmpeg -framerate {fps} -pattern_type glob -i '*.png' {mp4})")
        return
    h, w = cv2.imread(pngs[0]).shape[:2]
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in pngs:
        im = cv2.imread(p)
        vw.write(cv2.resize(im, (w, h)) if im.shape[:2] != (h, w) else im)
    vw.release()
    print(f"  video -> {mp4}")


def main():
    traj_pkl = os.environ["STAGE_B_TRAJ"]
    out_dir = os.environ.get("OUT_DIR", "./navhard_examples")
    per_cat = int(os.environ.get("PER_CAT", "12"))
    gt_only = os.environ.get("GT_ONLY", "1") == "1"   # keep only scenes that HAVE a GT future
    fps = int(os.environ.get("FPS", "2"))
    # Only the visually-unambiguous reel by default (first trajectory drives off-road while
    # GT/other candidates stay on). The scorer_better/scorer_worse reels are EPDMS-labelled and
    # the metric != how a path looks, so they confuse as figures — the results TABLE carries the
    # scorer story instead. (CATEGORIES=scorer_better,... to render them anyway.)
    cats = os.environ.get("CATEGORIES", "first_traj_offroad").split(",")

    # pickle is safe here: STAGE_B_TRAJ is our own Stage-B output (trusted, local file),
    # same trust basis as stage_c_build_submission.py — never point it at an untrusted pkl.
    with open(traj_pkl, "rb") as f:
        trajs = pickle.load(f)                            # {token: (N,6,2)}
    examples = json.load(open(os.environ["EXAMPLES_JSON"]))
    sel = json.load(open(os.environ["SELECTION"])) if os.environ.get("SELECTION") else {}

    loader = build_loader()
    os.makedirs(out_dir, exist_ok=True)
    for cat in cats:
        pool = examples.get(cat, [])                      # a LARGE pool (find_examples TOPK)
        if not pool:
            print(f"[{cat}] no tokens — skipping")
            continue
        cdir = osp.join(out_dir, cat)
        os.makedirs(cdir, exist_ok=True)
        pngs, skipped_nogt = [], 0
        for tok in pool:
            if len(pngs) >= per_cat:
                break
            if tok not in trajs:
                continue
            try:
                scene = loader.get_scene_from_token(tok)
            except Exception as e:  # noqa: BLE001
                print(f"[{cat}] {tok}: scene load FAILED {type(e).__name__}: {e}", flush=True)
                continue
            n_gt = gt_frames(scene)
            if gt_only and n_gt == 0:                     # synthetic stage-2 scene, no GT — skip
                skipped_nogt += 1
                continue
            cand = np.asarray(trajs[tok])[..., :2]
            png = osp.join(cdir, f"{len(pngs):03d}_{tok}.png")
            try:
                render_token(scene, cand, int(sel.get(tok, 0)) if sel else None, png, n_gt)
                pngs.append(png)
                print(f"[{cat}] {len(pngs)} {tok} -> {png}  (gt={n_gt})", flush=True)
            except Exception as e:  # noqa: BLE001 — one bad scene shouldn't stop the reel
                print(f"[{cat}] {tok}: render FAILED {type(e).__name__}: {e}", flush=True)
        note = f" (short: only {len(pngs)} GT-scenes in pool of {len(pool)})" if len(pngs) < per_cat else ""
        print(f"[{cat}] {len(pngs)} panels, skipped {skipped_nogt} without GT{note}", flush=True)
        stitch(pngs, osp.join(out_dir, f"{cat}.mp4"), fps)
    print(f"=== done -> {out_dir} ===")


if __name__ == "__main__":
    main()
