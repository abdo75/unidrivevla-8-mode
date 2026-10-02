"""Use a trained scorer to pick one candidate trajectory per token.

For each token: load its scene tokens phi_s and its N candidate trajectories, predict
the PDMS sub-scores for each candidate, reconstruct the EPDMS from those predictions,
and select the argmax candidate. Writes {token: best_candidate_idx} as JSON for
stage_c_build_submission.py (SELECTION=...) to turn into a NAVSIM submission.

Runs in the unidrivevla env (torch). Inputs come from a DUMP_SCENE_TOKENS Stage B run.

Run:  python select_candidates.py <scorer.pt> <stage_b_dir> <selection.json>
"""
import os
import sys
import json
import pickle

import numpy as np
import torch

from scorer_model import Scorer
from epdms import SUBSCORE_KEYS, reconstruct_epdms


def select(scorer_path, stage_b_dir, out_json):
    ckpt = torch.load(scorer_path, map_location="cpu", weights_only=True)
    model = Scorer(ckpt["c_in"], ckpt["k"])
    model.load_state_dict(ckpt["state"])
    model.eval()

    with open(os.path.join(stage_b_dir, "trajectories.pkl"), "rb") as fh:
        trajs = pickle.load(fh)  # our own Stage-B output (trusted): {token: (N,T,2)}
    phi_dir = os.path.join(stage_b_dir, "phi_s")

    selection, skipped = {}, 0
    with torch.no_grad():
        for tok, entry in trajs.items():
            phi_path = os.path.join(phi_dir, f"{tok}.npy")
            if not os.path.exists(phi_path):
                skipped += 1
                continue
            modes = np.asarray(entry["traj_modes"] if isinstance(entry, dict) else entry)  # (N,T,2)
            n = modes.shape[0]
            phi = torch.from_numpy(np.load(phi_path)).float().unsqueeze(0).expand(n, -1, -1)
            traj = torch.from_numpy(modes).float()                       # (N,T,2)
            pred = model(phi, traj).cpu().numpy()                        # (N,K)
            epdms = [reconstruct_epdms(dict(zip(SUBSCORE_KEYS, p))) for p in pred]
            selection[tok] = int(np.argmax(epdms))

    with open(out_json, "w") as fh:
        json.dump(selection, fh)
    # distribution of picked candidate indices -> sanity (not all collapsing to one mode)
    picks = np.bincount(list(selection.values()), minlength=1) if selection else np.array([])
    print(f"selected {len(selection)} tokens ({skipped} skipped, no phi_s) -> {out_json}")
    print(f"  candidate-index histogram: {picks.tolist()}")
    return len(selection)


if __name__ == "__main__":
    select(sys.argv[1], sys.argv[2], sys.argv[3])