#!/usr/bin/env python
"""
Pick concrete BAD and GOOD example scenes from the navhard per-candidate EPDMS CSVs,
to (a) understand the single-quality distribution and (b) feed the BEV videos.

For every token we assemble the 8 candidates' official per-token EPDMS (the `score`
column each run_pdm_score CSV writes), plus the deployed default (cand0), the oracle
(max over candidates), and the learned scorer's pick (from selection.json). Then it
categorises scenes and writes token lists the renderer can turn into good/bad reels.

Categories:
  bad_single      : cand0 is poor (< SINGLE_LOW) BUT a good candidate exists (oracle > OK)
                    -> the deployed trajectory fails though the model *could* have driven it
  scorer_win      : scorer pick clearly beats cand0 (>= WIN margin) -> the method working
  scorer_hurt     : scorer pick is worse than cand0 (<= -WIN) -> honest failure cases
  good_single     : cand0 already near the oracle -> nothing to gain (context)

Run (navhard):
  micromamba run -n unidrivevla python find_examples.py \
      <oracle_csv_glob> <selection.json> [out.json]
  e.g. oracle_csv_glob = "$NAVSIM_EXP_ROOT/uni_oracle_navhard_two_stage*_cand*"
"""
import glob
import json
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd

SINGLE_LOW = float(os.environ.get("SINGLE_LOW", "0.05"))   # cand0 "poor" threshold
OK = float(os.environ.get("OK", "0.30"))                   # oracle "a good candidate exists"
WIN = float(os.environ.get("WIN", "0.05"))                 # scorer vs cand0 margin
# Emit a big pool per category: the renderer keeps only scenes that have a GROUND-TRUTH
# future (the real stage-1 scenes; synthetic stage-2 scenes have none), so the pool must be
# large enough that ~PER_CAT stage-1 tokens survive that filter.
TOPK = int(os.environ.get("TOPK", "250"))                  # tokens per list (pool, pre GT-filter)


def _latest_csv(cand_dir):
    """run_pdm_score writes to <cand_dir>/<date>/<date>.csv; take the newest."""
    csvs = glob.glob(osp.join(cand_dir, "**", "*.csv"), recursive=True)
    return max(csvs, key=osp.getmtime) if csvs else None


def _column(csv, base):
    """{token: value} for a CSV column `base` (handles the _stage_one/_stage_two split);
    None if the column is absent."""
    df = pd.read_csv(csv)
    df = df[df["token"] != "average"]
    if base in df.columns:
        s = df[base]
    elif f"{base}_stage_one" in df.columns:
        s = df[f"{base}_stage_one"].combine_first(df[f"{base}_stage_two"])
    else:
        return None
    return dict(zip(df["token"], s.astype(float)))


def main(cand_glob, sel_path, out_path="examples_navhard.json"):
    # one directory per candidate (uni_oracle_..._cand0 ... _cand7)
    cand_dirs = sorted(d for d in glob.glob(cand_glob) if osp.isdir(d))
    if not cand_dirs:
        raise SystemExit(f"no candidate dirs match {cand_glob!r}")
    per_cand, per_dac = [], []
    for d in cand_dirs:
        csv = _latest_csv(d)
        if csv is None:
            raise SystemExit(f"no CSV under {d}")
        per_cand.append(_column(csv, "score"))
        per_dac.append(_column(csv, "drivable_area_compliance"))
    N = len(per_cand)
    have_dac = all(x is not None for x in per_dac)
    print(f"loaded {N} candidate score CSVs" + ("  (+ DAC sub-score)" if have_dac else "  (no DAC column)"))

    sel = json.load(open(sel_path)) if osp.isfile(sel_path) else {}
    tokens = set(per_cand[0])
    for c in per_cand[1:]:
        tokens &= set(c)

    rows = []
    for tok in tokens:
        e = np.array([per_cand[k][tok] for k in range(N)])
        pick = int(sel.get(tok, 0))
        row = dict(token=tok, cand0=e[0], oracle=e.max(), best_idx=int(e.argmax()),
                   scorer_pick=int(pick), scorer=e[pick], spread=e.max() - e.min())
        if have_dac:
            dac = np.array([per_dac[k][tok] for k in range(N)])
            row.update(cand0_dac=dac[0], best_sibling_dac=dac.max())
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"tokens: {len(df)}")
    print(f"cand0 (single) : mean={df.cand0.mean():.3f}  frac<{SINGLE_LOW}={ (df.cand0<SINGLE_LOW).mean():.2%}  frac==0={(df.cand0==0).mean():.2%}")
    print(f"oracle         : mean={df.oracle.mean():.3f}")
    print(f"scorer pick    : mean={df.scorer.mean():.3f}  beats cand0 in {(df.scorer>df.cand0).mean():.2%}, ties {(df.scorer==df.cand0).mean():.2%}, worse {(df.scorer<df.cand0).mean():.2%}")
    print(f"within-scene spread==0 (no diversity) in {(df.spread==0).mean():.2%} of scenes")

    # Plain-English category names (these become the reel/folder names). NB: better/worse are
    # by EPDMS (the deployable metric), which is NOT how a path looks — a GT-hugging path can
    # still score 0 by clipping the drivable area (DAC) or nearing an agent (TTC).
    #   first_traj_fails    - first trajectory scores near 0, but a better candidate exists
    #   scorer_better       - the scorer's choice beats the first trajectory (by EPDMS)
    #   scorer_worse        - the scorer's choice is worse than the first trajectory (by EPDMS)
    #   first_traj_offroad  - the first trajectory drives OFF the drivable area (DAC=0) while
    #                         another candidate stays ON it (the clearest failure figure)
    lists = {
        "first_traj_fails":  df[(df.cand0 < SINGLE_LOW) & (df.oracle > OK)].sort_values("cand0").head(TOPK),
        "scorer_better":     df[(df.scorer - df.cand0) >= WIN].sort_values("scorer", ascending=False).head(TOPK),
        "scorer_worse":      df[(df.scorer - df.cand0) <= -WIN].sort_values("scorer").head(TOPK),
    }
    if have_dac:
        lists["first_traj_offroad"] = df[(df.cand0_dac == 0) & (df.best_sibling_dac > 0.9) & (df.oracle > OK)] \
            .sort_values("oracle", ascending=False).head(TOPK)
    out = {k: v["token"].tolist() for k, v in lists.items()}
    for k, v in lists.items():
        print(f"  {k:12s}: {len(v)} tokens (showing scores of first 3)")
        for _, r in v.head(3).iterrows():
            print(f"      {r.token}  cand0={r.cand0:.2f} oracle={r.oracle:.2f} (best mode {r.best_idx}) scorer[{r.scorer_pick}]={r.scorer:.2f}")
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"wrote example token lists -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:4])
