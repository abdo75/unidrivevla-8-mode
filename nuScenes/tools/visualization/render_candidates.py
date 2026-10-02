#!/usr/bin/env python
"""
BEV visualisation of the N candidate ego trajectories (multi-mode planner),
drawn ON TOP OF the real scene context (HD-map lanes + agent boxes).
-----------------------------------------------------------------------------
For each val sample it runs forward_test(return_all_modes=True) to get the N
candidate ego trajectories, then reuses the repo's own BEVRender to draw the
scene backdrop (map vectors + detected/GT agent boxes + the ego-car sprite +
the driving command), and overlays:
  - the N candidate trajectories        (faint grey)
  - the DEPLOYED trajectory (mode 0)    (blue, bold)
  - the ORACLE-BEST candidate           (orange, bold)  = closest to GT
  - the ground-truth ego future         (green, dashed)
One PNG per sample; consecutive samples can be stitched into an mp4 (--video).

Two views of each sample, aligned by index (one shuffle=False dataset):
  - model input : the collated/pipelined batch from the dataloader (images...)
  - scene GT    : dataset.get_data_info(i) -> map_infos, gt_bboxes_3d, gt futures
Loader batch i and get_data_info(i) are the SAME scene because test_mode=True
disables empty-GT filtering, so the two stay in lock-step.

COORDINATE FRAME: candidate trajectories are plotted with the SAME mapping the
repo's BEVRender uses for the GT ego path (component 0 -> plot x, component 1 ->
plot y, in the box/ego frame). So a well-trained mode 0 should overlay the green
GT line -- which both fixes the axis convention by construction and gives an
instant visual sanity check on the model.

Usage:
  python tools/visualization/render_candidates.py <config> <checkpoint> \
      --out-dir viz_out --num-samples 12 [--start 0] [--video] [--agent-motion]
Model-loading mirrors tools/dump_modes.py.
"""
import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# make nuScenes/ (parent of tools/) importable, like dump_modes.py / visualize.py
_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from tools.visualization.bev_render import BEVRender


def _find_model_states(path):
    if osp.isfile(path):
        return path
    for root, _dirs, files in os.walk(path):
        if "mp_rank_00_model_states.pt" in files:
            return osp.join(root, "mp_rank_00_model_states.pt")
    raise FileNotFoundError(f"no mp_rank_00_model_states.pt under {path}")


def _extract_state_dict(ckpt):
    sd = ckpt
    for key in ("state_dict", "module", "model"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def gt_ego_positions(info):
    """Cumulative GT ego future positions (T, 2) in the box/ego frame, matching
    BEVRender.draw_planning_gt, or None if this sample has no valid GT ego path.
    get_data_info stores per-step displacements + a validity mask."""
    trajs = info.get("gt_ego_fut_trajs", None)
    masks = info.get("gt_ego_fut_masks", None)
    if trajs is None or masks is None:
        return None
    masks = np.asarray(masks).astype(bool)
    if masks.size == 0 or not bool(masks[0]):
        return None
    trajs = np.asarray(info["gt_ego_fut_trajs"])[masks].astype(np.float64)
    trajs[np.abs(trajs) < 0.01] = 0.0            # same denoise as draw_planning_gt
    return np.cumsum(trajs, axis=0)              # displacements -> positions


class CandidateBEVRender(BEVRender):
    """BEVRender that skips the gt/pred output dirs and adds a candidate overlay."""

    def __init__(self, plot_choices, xlim=40, ylim=40):
        # deliberately do NOT call super().__init__ (it makes bev_gt/bev_pred dirs
        # and needs an out_dir); we save our own single PNG per sample.
        self.plot_choices = plot_choices
        self.xlim = xlim
        self.ylim = ylim

    def _render_sdc_car(self):
        # resources/sdc_car.png is loaded relative to cwd (nuScenes/). Fall back to a
        # plain marker if it's missing so a bad cwd doesn't kill the whole render.
        try:
            super()._render_sdc_car()
        except Exception:
            self.axes.plot(0, 0, marker="^", color="black", markersize=12, zorder=3)

    def draw_candidates(self, traj_modes, gt_ego=None):
        """traj_modes: (N, T, 2) cumulative ego-frame positions (same frame as the
        GT ego path). Returns (best_idx, per_mode_ADE) or (None, None) if no GT."""
        N = traj_modes.shape[0]
        ego0 = np.zeros((N, 1, 2), dtype=traj_modes.dtype)
        tm = np.concatenate([ego0, traj_modes], axis=1)          # (N, T+1, 2) start at ego

        best, ade = None, None
        if gt_ego is not None and len(gt_ego) > 0:
            T = min(traj_modes.shape[1], gt_ego.shape[0])
            ade = np.sqrt(((traj_modes[:, :T] - gt_ego[None, :T]) ** 2).sum(-1)).mean(1)
            best = int(ade.argmin())
            gt_line = np.concatenate([np.zeros((1, 2)), gt_ego], axis=0)
            self.axes.plot(gt_line[:, 0], gt_line[:, 1], color="tab:green", ls="--",
                           lw=3.0, zorder=4, label="ground truth")

        for n in range(N):                                        # candidates -- faint grey
            self.axes.plot(tm[n, :, 0], tm[n, :, 1], color="0.55", lw=2.0, alpha=0.6, zorder=4)
        self.axes.plot(tm[0, :, 0], tm[0, :, 1], color="tab:blue", lw=3.6, zorder=6,
                       label="deployed (mode 0)")
        if best is not None:
            self.axes.plot(tm[best, :, 0], tm[best, :, 1], color="tab:orange", lw=3.6,
                           zorder=6, label=f"oracle-best (mode {best})")
        # legend in the lower-right corner (the ego drives "up", so the map detail is
        # ahead/above -- keep the legend out of it).
        self.axes.legend(loc="lower right", fontsize=22, framealpha=0.9)
        return best, ade


def render_sample(render, info, traj_modes, idx, out_png, show_motion):
    """Draw one BEV: map + boxes (+ optional agent motion) + command, then candidates."""
    render.reset_canvas()
    render.draw_map_gt(info)
    render.draw_detection_gt(info)
    if show_motion:
        render.draw_motion_gt(info)
    render._render_sdc_car()
    try:
        render._render_command(info)
    except Exception:
        pass
    gt_ego = gt_ego_positions(info)
    best, ade = render.draw_candidates(traj_modes, gt_ego)
    if ade is not None:
        render.axes.text(0, render.xlim - 2,
                         f"oracle minADE={ade[best]:.2f}m   mode0={ade[0]:.2f}m",
                         fontsize=26, color="black", ha="center", va="top")
    render.save_fig(out_png)
    return best, ade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--out-dir", default="viz_candidates")
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--start", type=int, default=0, help="first sample index to render")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--agent-motion", action="store_true",
                    help="also draw other agents' GT future paths (more context, more clutter)")
    ap.add_argument("--video", action="store_true", help="stitch the PNGs into candidates.mp4")
    ap.add_argument("--fps", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config.fromfile(args.config)
    import importlib
    importlib.import_module("projects.mmdet3d_plugin")   # register custom models/datasets

    ds_cfg = cfg.data[args.split]
    ds_cfg.test_mode = True   # deterministic order: loader idx i == dataset.get_data_info(i)
    dataset = build_dataset(ds_cfg)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False)

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    # weights_only=False: DeepSpeed model_states hold non-tensor objects, so the safe
    # loader can't read them. These are our own trusted training checkpoints (same load
    # as tools/dump_modes.py) -- never point this at an untrusted file.
    ckpt = torch.load(_find_model_states(args.checkpoint), map_location="cpu", weights_only=False)
    sd = _extract_state_dict(ckpt)
    sd = {k: v for k, v in sd.items() if not k.startswith("ema_")}   # drop EMA copies
    model.load_state_dict(sd, strict=False)
    model = MMDataParallel(model.cuda().eval(), device_ids=[0])

    # det + map on; planning off (we draw our own GT + candidates); pred panel off
    # (map/motion/tracking heads are disabled in the eval config).
    plot_choices = dict(draw_pred=False, det=True, track=False,
                        motion=args.agent_motion, map=True, planning=False)
    render = CandidateBEVRender(plot_choices)

    pngs = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i < args.start:
                continue
            if i >= args.start + args.num_samples:
                break
            out = model(return_loss=False, return_all_modes=True, rescale=True, **data)
            tm = out[0]["img_bbox"].get("traj_modes")
            if tm is None:
                print(f"{i:4d} | no traj_modes (num_modes<=1?) -- skipping", flush=True)
                continue
            tm = (tm.cpu().numpy() if torch.is_tensor(tm) else np.asarray(tm))[..., :2]
            info = dataset.get_data_info(i)   # raw scene GT for the backdrop
            png = osp.join(args.out_dir, f"cand_{i:05d}.png")
            best, ade = render_sample(render, info, tm, i, png, args.agent_motion)
            pngs.append(png)
            tag = "" if ade is None else f"  minADE={ade[best]:.2f}m (mode0={ade[0]:.2f}m)"
            print(f"{i:4d} | rendered {png}{tag}", flush=True)

    if args.video and pngs:
        # cv2 is already a dependency (bev_render imports it); use it instead of imageio,
        # which may not be installed. All frames are the same size (BEVRender's fixed
        # 20x20 canvas), so no resize is needed for the writer.
        import cv2
        h, w = cv2.imread(pngs[0]).shape[:2]
        mp4 = osp.join(args.out_dir, "candidates.mp4")
        writer = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        for p in pngs:
            writer.write(cv2.imread(p))
        writer.release()
        print(f"video -> {mp4}  ({len(pngs)} frames @ {args.fps} fps)", flush=True)


if __name__ == "__main__":
    main()
