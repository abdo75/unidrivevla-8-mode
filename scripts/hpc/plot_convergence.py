#!/usr/bin/env python
"""
Plot the multi-mode fine-tune convergence curve from the eval_minade logs.
-----------------------------------------------------------------------------
Parses every eval_<iter>.log (or eval_minade_iter_<iter>_*.log) in a directory,
reads the "[Multi-mode] minADE_k=.. mode0_ADE=.. diversity_gap=.." line each one
prints, and draws the held-out convergence figure for the thesis:
  - mode-0 ADE (what actually gets deployed)
  - minADE_k  (oracle over the N modes)  -> the gap between them is the in-domain
    diversity/oracle headroom the learned scorer is meant to recover.

Reproducible: re-run it after adding more checkpoints' logs and the figure updates.

Usage (in the unidrivevla env, where matplotlib lives):
  python scripts/hpc/plot_convergence.py [--log-dir scripts/hpc] [--out convergence_minade.png]
"""
import argparse
import glob
import os
import os.path as osp
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r"minADE_k=([\d.]+).*?mode0_ADE=([\d.]+).*?diversity_gap\(mode0-min\)=([\d.]+)")
ITER_RE = re.compile(r"(?:iter[_-])?(\d+)")


def iter_of(path):
    """Pull the iteration number out of eval_5000.log / eval_minade_iter_5000_*.log."""
    base = osp.basename(path)
    m = re.search(r"iter[_-]?(\d+)", base) or re.search(r"eval_(\d+)", base)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="scripts/hpc")
    ap.add_argument("--out", default="convergence_minade.png")
    args = ap.parse_args()

    rows = []
    for path in glob.glob(osp.join(args.log_dir, "eval_*.log")):
        it = iter_of(path)
        if it is None:
            continue
        with open(path) as f:
            txt = f.read()
        hits = LINE_RE.findall(txt)
        if not hits:
            continue
        mind, mode0, gap = (float(x) for x in hits[-1])   # last [Multi-mode] line in the log
        rows.append((it, mind, mode0, gap))

    if not rows:
        raise SystemExit(f"no [Multi-mode] lines found under {args.log_dir}/eval_*.log")
    rows.sort()
    it, mind, mode0, gap = zip(*rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(it, mode0, "o-", color="0.5", lw=2,
            label="first predicted trajectory (the one deployed)")
    ax.plot(it, mind, "o-", color="tab:blue", lw=2.4,
            label="best of the 8 candidates (closest to ground truth)")
    ax.fill_between(it, mind, mode0, color="tab:orange", alpha=0.18,
                    label="headroom (gain from picking the best candidate)")
    ax.set_xlabel("training iteration  (8 scenes per iteration: 8 GPUs × 1)")
    ax.set_ylabel("average distance to the human trajectory (m)\nlower is better")
    ax.set_title("Fine-tuning the mode embeddings on nuScenes trainval\n"
                 "trajectory error on held-out validation (frozen backbone, 8 candidates)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="center right")
    # one-line explanation so the figure stands on its own
    fig.text(0.5, 0.005,
             "Distance = mean gap between a predicted trajectory and the human-driven "
             "ground truth over the 3 s horizon (ADE).",
             ha="center", fontsize=7.5, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.out, dpi=160)
    print(f"parsed {len(rows)} checkpoints from {args.log_dir}; saved {args.out}")
    for r in rows:
        print("  iter=%-6d minADE_k=%.4f  mode0=%.4f  gap=%.4f" % r)


if __name__ == "__main__":
    main()
