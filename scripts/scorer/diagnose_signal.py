"""Diagnose whether there is any learnable selection signal in the scorer dataset.

For each scene we have N candidates, each with true sub-scores -> a true EPDMS. If the
candidates' EPDMS are nearly identical within a scene, no scorer can beat random and the
bottleneck is candidate diversity, not the model. If there is real within-scene spread
but the CV scorer can't capture it, the bottleneck is features (e.g. missing ego state).

Reports, over scenes: distribution of within-scene EPDMS spread (max-min, std), the
fraction of scenes where the best candidate beats candidate 0 by >0.01 / >0.05, and the
mean oracle-vs-random gap (the learnable ceiling).

Run:  micromamba run -n unidrivevla python diagnose_signal.py <dataset.pt>
"""
import sys
from collections import defaultdict

import numpy as np
import torch

from epdms import SUBSCORE_KEYS, reconstruct_epdms


def main(ds_path):
    ds = torch.load(ds_path, weights_only=True)
    # group records by scene (phi tensor identity isn't stable; use phi shape+sum hash is
    # brittle -- instead rely on insertion order: build_dataset emits all candidates of a
    # token consecutively, so we regroup by equal phi via a content key).
    by_scene = defaultdict(list)
    for r in ds:
        key = (r["phi"].shape, float(r["phi"].sum()))
        by_scene[key].append(r["y"].numpy())

    spreads, stds, best_minus_c0, oracle, randm = [], [], [], [], []
    win01 = win05 = 0
    for ys in by_scene.values():
        e = np.array([reconstruct_epdms(dict(zip(SUBSCORE_KEYS, y))) for y in ys])
        spreads.append(e.max() - e.min())
        stds.append(e.std())
        best_minus_c0.append(e.max() - e[0])
        oracle.append(e.max())
        randm.append(e.mean())
        win01 += (e.max() - e[0]) > 0.01
        win05 += (e.max() - e[0]) > 0.05

    n = len(by_scene)
    sp = np.array(spreads)
    print(f"scenes={n}  candidates/scene~{len(ds)/max(n,1):.1f}")
    print(f"within-scene EPDMS spread (max-min): mean={sp.mean():.4f}  median={np.median(sp):.4f}  "
          f"p90={np.percentile(sp,90):.4f}  frac==0: {(sp==0).mean():.2%}")
    print(f"within-scene EPDMS std:              mean={np.mean(stds):.4f}")
    print(f"oracle_mean={np.mean(oracle):.4f}  random_mean={np.mean(randm):.4f}  "
          f"learnable_ceiling={np.mean(oracle)-np.mean(randm):+.4f}")
    print(f"scenes where best beats cand0 by >0.01: {win01/n:.1%}   by >0.05: {win05/n:.1%}")
    print("--> if spread is ~0 on most scenes, candidates are too similar (diversity is the "
          "bottleneck, not the scorer). If spread is real, features are the bottleneck.")


if __name__ == "__main__":
    main(sys.argv[1])
