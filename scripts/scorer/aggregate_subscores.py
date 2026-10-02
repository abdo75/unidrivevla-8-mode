"""Break the EPDMS down into its nine sub-scores for each selection policy.

For every scene the dataset holds the sub-scores of all candidates. We report the mean of
each sub-score (and the reconstructed EPDMS) under four policies:
  single     - always candidate 0 (the model's default / deployed mode)
  best-fixed - the one candidate index with the highest mean EPDMS over all scenes
  oracle     - per scene, the candidate with the highest true EPDMS (upper bound)
  scorer     - per scene, the candidate chosen by a selection JSON (optional arg)

This shows WHERE the oracle headroom comes from (which sub-scores improve) and what the
learned scorer actually changes, with no model retraining.

Run:  python aggregate_subscores.py <dataset.pt> [selection.json]
"""
import sys
import json
from collections import defaultdict

import numpy as np
import torch

from epdms import SUBSCORE_KEYS, reconstruct_epdms


def _epdms(y):
    return reconstruct_epdms(dict(zip(SUBSCORE_KEYS, [float(x) for x in y])))


def main(ds_path, sel_path=None):
    ds = torch.load(ds_path, weights_only=True)
    by_tok = defaultdict(dict)            # token -> {cand_idx: y(9,)}
    for r in ds:
        by_tok[r["token"]][int(r["cand"])] = np.asarray(r["y"], dtype=np.float64)
    tokens = list(by_tok)

    # best-fixed: candidate index with the highest mean EPDMS across scenes
    cand_ids = sorted({c for d in by_tok.values() for c in d})
    mean_epdms = {c: np.mean([_epdms(by_tok[t][c]) for t in tokens if c in by_tok[t]])
                  for c in cand_ids}
    best_fixed = max(mean_epdms, key=mean_epdms.get)

    sel = json.load(open(sel_path)) if sel_path else None

    policies = ["single", f"best-fixed(c{best_fixed})", "oracle"]
    if sel is not None:
        policies.append("scorer")

    acc = {p: [] for p in policies}       # p -> list of chosen y vectors
    for t in tokens:
        d = by_tok[t]
        es = {c: _epdms(y) for c, y in d.items()}
        acc["single"].append(d.get(0, d[min(d)]))
        acc[f"best-fixed(c{best_fixed})"].append(d.get(best_fixed, d[min(d)]))
        acc["oracle"].append(d[max(es, key=es.get)])
        if sel is not None:
            c = int(sel[t]) if t in sel and int(sel[t]) in d else 0
            acc["scorer"].append(d.get(c, d[min(d)]))

    hdr = "policy".ljust(20) + "".join(k[:10].rjust(11) for k in SUBSCORE_KEYS) + "   EPDMS"
    print(f"scenes={len(tokens)}  candidates/scene~{len(ds)/max(len(tokens),1):.1f}")
    print(hdr)
    print("-" * len(hdr))
    for p in policies:
        M = np.stack(acc[p])              # (scenes, 9)
        means = M.mean(axis=0)
        epd = np.mean([_epdms(y) for y in M])
        print(p.ljust(20) + "".join(f"{m:11.3f}" for m in means) + f"{epd:9.4f}")
    print("\nsub-score order:", ", ".join(SUBSCORE_KEYS))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
