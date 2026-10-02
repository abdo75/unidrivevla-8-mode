#!/usr/bin/env python
"""
Build the ORACLE selection {token: argmax candidate by REAL EPDMS} from the per-candidate
score CSVs, so the oracle can be scored the SAME way as the learned scorer (re-score the
per-scene best pick as one submission -> official two-stage EPDMS Final). This makes the
table's oracle cell apples-to-apples with the scorer cell, not the per-token-mean proxy.

Run:  python build_oracle_selection.py "<cand_glob>" <out_selection.json>
  e.g. cand_glob = "$NAVSIM_EXP_ROOT/uni_oracle_navhard_two_stage*_cand*"
"""
import glob
import json
import os.path as osp
import sys

import numpy as np
import pandas as pd


def _scores(csv):
    df = pd.read_csv(csv)
    df = df[df["token"] != "average"]
    s = df["score"] if "score" in df.columns else \
        df["score_stage_one"].combine_first(df["score_stage_two"])
    return dict(zip(df["token"], s.astype(float)))


def main(cand_glob, out):
    dirs = sorted(d for d in glob.glob(cand_glob) if osp.isdir(d))
    if not dirs:
        raise SystemExit(f"no candidate dirs match {cand_glob!r}")
    per = []
    for d in dirs:
        csvs = glob.glob(osp.join(d, "**", "*.csv"), recursive=True)
        per.append(_scores(max(csvs, key=osp.getmtime)))
    toks = set(per[0]).intersection(*[set(p) for p in per[1:]])
    sel = {t: int(np.argmax([per[k][t] for k in range(len(per))])) for t in toks}
    json.dump(sel, open(out, "w"))
    print(f"oracle selection: {len(sel)} tokens over {len(dirs)} candidates -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
