"""Join the Stage B dumps (phi_s scene tokens + candidate trajectories) with the
per-candidate NAVSIM sub-score CSVs into one training set for the learned scorer.

One record per (token, candidate) for which all three parts exist:
    {"token": str, "cand": int,
     "phi": (T_s, C) float32, "traj": (T, 2) float32, "y": (K,) float32}
where K = len(SUBSCORE_KEYS) and y holds that candidate's sub-scores in [0, 1].
``token`` lets the trainer group candidates by scene for the ranking metric.

The candidate index is parsed from the CSV directory name ("...cand<N>...") rather
than from sort order, so cand10 does not collate before cand2. Sub-score columns are
read either plainly (``ego_progress``) or stage-suffixed (``ego_progress_stage_one`` /
``_stage_two``), coalescing the non-null stage for each token.
"""
import os
import re
import glob
import pickle

import numpy as np
import torch
import pandas as pd

from epdms import SUBSCORE_KEYS


def _subscore_series(df, key):
    """Return df[key], falling back to coalescing the *_stage_one/_stage_two columns."""
    if key in df.columns:
        return df[key]
    one, two = key + "_stage_one", key + "_stage_two"
    if one in df.columns or two in df.columns:
        a = df[one] if one in df.columns else None
        b = df[two] if two in df.columns else None
        if a is None:
            return b
        if b is None:
            return a
        return a.combine_first(b)
    raise KeyError(f"sub-score column {key!r} not found (have {list(df.columns)})")


def _candidate_dfs(cand_glob):
    """Map candidate index -> DataFrame indexed by token, one per *cand<N>* dir.

    cand_glob expands to the per-candidate score dirs for ONE split, e.g.
    "$NAVSIM_EXP_ROOT/uni_oracle_navtrain_*_cand*". Using a split-specific glob (not a
    bare "*cand*" under a shared exp root) avoids cand0 from another split overwriting
    this split's cand0.
    """
    by_cand = {}
    for d in sorted(glob.glob(cand_glob)):
        if not os.path.isdir(d):
            continue
        m = re.search(r"cand(\d+)", os.path.basename(d))
        if not m:
            continue
        csvs = glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
        if not csvs:
            continue
        # A cand dir accumulates one timestamped CSV per scoring run; take only the
        # newest (what the oracle aggregator uses) so re-runs don't duplicate tokens.
        newest = max(csvs, key=os.path.getmtime)
        df = pd.read_csv(newest)
        # run_pdm_score appends a final aggregate row (token == "average"); drop it and
        # any accidental dup tokens, keeping the last occurrence.
        df = df[df["token"] != "average"].drop_duplicates(subset="token", keep="last")
        by_cand[int(m.group(1))] = df.set_index("token")
    return by_cand


def build_dataset(stage_b_dir, cand_glob, out):
    """Build and save the scorer dataset; returns the number of records written.

    cand_glob: glob expanding to this split's per-candidate score dirs (see
    _candidate_dfs). A plain directory path is accepted too (its *cand* children
    are used), for convenience on a single-split layout.
    """
    if os.path.isdir(cand_glob):
        cand_glob = os.path.join(cand_glob, "*cand*")
    with open(os.path.join(stage_b_dir, "trajectories.pkl"), "rb") as fh:
        trajs = pickle.load(fh)
    by_cand = _candidate_dfs(cand_glob)
    recs = []
    for tok, d in trajs.items():
        phi_path = os.path.join(stage_b_dir, "phi_s", f"{tok}.npy")
        if not os.path.exists(phi_path):
            continue
        phi = torch.from_numpy(np.load(phi_path)).float()             # (T_s, C)
        modes = np.asarray(d["traj_modes"] if isinstance(d, dict) else d)  # (N, T, 2)
        for n, df in by_cand.items():
            if n >= modes.shape[0] or tok not in df.index:
                continue
            row = df.loc[tok]
            y = torch.tensor([float(_subscore_series(df, k).loc[tok]) for k in SUBSCORE_KEYS],
                             dtype=torch.float32)
            recs.append({"token": tok,
                         "cand": n,
                         "phi": phi,
                         "traj": torch.from_numpy(modes[n]).float(),
                         "y": y})
    torch.save(recs, out)
    return len(recs)


if __name__ == "__main__":
    import sys
    n = build_dataset(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"wrote {n} records -> {sys.argv[3]}")