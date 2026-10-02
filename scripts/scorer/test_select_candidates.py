"""Self-contained test for select_candidates: fabricate a Stage-B dir + a tiny trained
scorer, run selection, check it emits a valid {token: idx} JSON. Runs without ubix data.

Run:  micromamba run -n unidrivevla python test_select_candidates.py
"""
import os
import json
import pickle
import tempfile

import numpy as np
import torch

from scorer_model import Scorer
from epdms import SUBSCORE_KEYS
from select_candidates import select


def main():
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "phi_s"), exist_ok=True)
        toks = [f"t{i}" for i in range(5)]
        n_cand, t_s, c, t = 6, 7, 16, 6
        trajs = {}
        for tok in toks:
            np.save(os.path.join(root, "phi_s", f"{tok}.npy"),
                    np.random.randn(t_s, c).astype(np.float16))
            trajs[tok] = np.random.randn(n_cand, t, 2).astype(np.float32)
        # one token deliberately has no phi_s -> must be skipped, not crash
        trajs["no_phi"] = np.random.randn(n_cand, t, 2).astype(np.float32)
        with open(os.path.join(root, "trajectories.pkl"), "wb") as fh:
            pickle.dump(trajs, fh)

        scorer = os.path.join(root, "scorer.pt")
        torch.save({"state": Scorer(c, len(SUBSCORE_KEYS)).state_dict(),
                    "c_in": c, "k": len(SUBSCORE_KEYS)}, scorer)

        out = os.path.join(root, "sel.json")
        n = select(scorer, root, out)
        sel = json.load(open(out))
        assert n == len(toks), (n, len(toks))           # 5 with phi_s, no_phi skipped
        assert set(sel) == set(toks)
        assert all(0 <= v < n_cand for v in sel.values())
    print(f"OK: select_candidates -> {n} tokens, indices in [0,{n_cand}), skipped no-phi token")


if __name__ == "__main__":
    main()