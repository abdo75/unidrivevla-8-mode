"""Self-contained test for cv_select: fabricate a Stage-B dump + sub-score CSVs, build
the dataset, run 5-fold CV selection, and check every scene gets exactly one out-of-fold
selection with a valid candidate index. Runs without ubix data.

Run:  micromamba run -n unidrivevla python test_cv_select.py
"""
import os
import json
import tempfile

from test_build_dataset import _make_fixture
from build_dataset import build_dataset
import cv_select


def main():
    with tempfile.TemporaryDirectory() as root:
        toks = tuple(f"t{i}" for i in range(40))
        csv_dir = _make_fixture(root, tokens=toks, n_cand=6)
        ds = os.path.join(root, "d.pt")
        sel = os.path.join(root, "sel.json")
        assert build_dataset(root, csv_dir, ds) == 40 * 6
        n = cv_select.cv_select(ds, sel, k=5, epochs=10)
        s = json.load(open(sel))
        assert n == len(toks) == len(s), (n, len(s))
        assert set(s) == set(toks)                       # every scene selected, out-of-fold
        assert all(0 <= v < 6 for v in s.values())
    print(f"OK: cv_select -> {n} out-of-fold selections, all indices valid")


if __name__ == "__main__":
    main()