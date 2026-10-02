"""Self-contained test for build_dataset: fabricate a tiny Stage-B dump + sub-score
CSVs in a temp dir, then check the join produces well-formed records. Runs without
the real ubix dump (which validates it again later on the 4-scene smoke output).

Run:  micromamba run -n unidrivevla python test_build_dataset.py
"""
import os
import csv
import pickle
import tempfile

import numpy as np
import torch

from build_dataset import build_dataset
from epdms import SUBSCORE_KEYS


def _make_fixture(root, tokens=("tokA", "tokB"), n_cand=3, t_s=5, c=8, t=6):
    os.makedirs(os.path.join(root, "phi_s"), exist_ok=True)
    trajs = {}
    for tok in tokens:
        np.save(os.path.join(root, "phi_s", f"{tok}.npy"),
                np.random.randn(t_s, c).astype(np.float16))
        trajs[tok] = np.random.randn(n_cand, t, 2).astype(np.float32)  # bare (N,T,2)
    with open(os.path.join(root, "trajectories.pkl"), "wb") as fh:
        pickle.dump(trajs, fh)

    csv_dir = os.path.join(root, "scores")
    for n in range(n_cand):
        d = os.path.join(csv_dir, f"uni_oracle_cand{n}")
        os.makedirs(d, exist_ok=True)
        # A stale earlier run (wrong values) that must be ignored in favour of the newest.
        stale = os.path.join(d, "old.csv")
        with open(stale, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["token"] + SUBSCORE_KEYS)
            for tok in tokens:
                w.writerow([tok] + [9.0 for _ in SUBSCORE_KEYS])  # out-of-range sentinel
        os.utime(stale, (1, 1))  # force an older mtime
        # Newest run: valid values + a trailing "average" aggregate row to be dropped.
        with open(os.path.join(d, "new.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["token"] + SUBSCORE_KEYS)
            for tok in tokens:
                w.writerow([tok] + [round(np.random.rand(), 4) for _ in SUBSCORE_KEYS])
            w.writerow(["average"] + [0.5 for _ in SUBSCORE_KEYS])
    return csv_dir


def main():
    with tempfile.TemporaryDirectory() as root:
        csv_dir = _make_fixture(root)
        out = os.path.join(root, "d.pt")
        n = build_dataset(root, csv_dir, out)
        ds = torch.load(out, weights_only=True)
        assert n == len(ds), (n, len(ds))
        assert n == 2 * 3, n  # 2 tokens x 3 candidates, all parts present
        r = ds[0]
        assert r["phi"].ndim == 2 and r["phi"].dtype == torch.float32
        assert r["traj"].shape[1] == 2
        assert r["y"].shape == (len(SUBSCORE_KEYS),)
        assert (r["y"] >= 0).all() and (r["y"] <= 1).all()
    print(f"OK: build_dataset -> {n} records, phi{tuple(r['phi'].shape)} "
          f"traj{tuple(r['traj'].shape)} y{tuple(r['y'].shape)}")


if __name__ == "__main__":
    main()