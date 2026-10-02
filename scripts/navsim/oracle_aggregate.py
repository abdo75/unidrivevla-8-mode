"""
Oracle-ceiling aggregation (DrivoR thesis). Given one NavSim PDMS score CSV per
candidate (built by scoring CAND_IDX=0..N-1 submissions of the SAME tokens),
compute the best-of-N upper bound:

  single   = mean over tokens of the CAND_IDX=0 score   (one-trajectory baseline)
  oracle   = mean over tokens of max_k score_k          (best-of-N, peeks at PDMS)
  headroom = oracle - single                            (what a perfect scorer could win)

The oracle is NOT deployable (it uses the real PDMS, i.e. the GT future); it is the
ceiling Increment 3's *learned* scorer aims to recover without the future.

CSV format (NavSim run_pdm_score_from_submission): columns include `token`, `valid`,
the PDMS sub-scores, and final `score`. A trailing aggregate row (token in
{average, final_score, ...}) is dropped.

Usage: python oracle_aggregate.py score_cand0.csv score_cand1.csv ...
       (candidate index = argument order; arg 0 is the single-trajectory baseline)
"""
import sys

import numpy as np
import pandas as pd

_DROP = {"average", "final_score", "mean", "aggregate", "nan", ""}


def load_scores(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    tok_c = cols.get("token")
    score_c = cols.get("score")
    if tok_c is None or score_c is None:
        raise ValueError(f"{path}: need 'token' and 'score' columns; got {list(df.columns)}")
    if "valid" in cols:
        df = df[df[cols["valid"]].astype(str).str.lower() == "true"]
    df = df[~df[tok_c].astype(str).str.lower().isin(_DROP)]
    df = df[pd.to_numeric(df[score_c], errors="coerce").notna()]
    return {str(t): float(s) for t, s in zip(df[tok_c], df[score_c])}


def main():
    paths = sys.argv[1:]
    if len(paths) < 2:
        print("usage: oracle_aggregate.py score_cand0.csv score_cand1.csv ...")
        sys.exit(1)

    per_cand = [load_scores(p) for p in paths]
    n = len(per_cand)
    # tokens scored by every candidate (should be identical sets)
    common = set(per_cand[0])
    for d in per_cand[1:]:
        common &= set(d)
    common = sorted(common)
    if not common:
        print("!! no tokens common to all candidate score files"); sys.exit(1)
    dropped = max(len(d) for d in per_cand) - len(common)
    if dropped:
        print(f"WARNING: {dropped} tokens not scored by every candidate were dropped "
              f"(kept {len(common)} common).")

    M = np.array([[d[t] for d in per_cand] for t in common])  # (T, N) score per token x candidate
    cand_means = M.mean(axis=0)                                # per-candidate mean PDMS
    single = cand_means[0]                                     # CAND_IDX=0 baseline
    best_single = cand_means.max()                             # best fixed mode (still deployable)
    oracle = M.max(axis=1).mean()                              # best-of-N per token (NOT deployable)

    print(f"\n=== Oracle ceiling over {len(common)} tokens, N={n} candidates ===")
    for k, m in enumerate(cand_means):
        tag = "  (single baseline)" if k == 0 else ("  (best fixed mode)" if m == best_single else "")
        print(f"  candidate {k}: mean PDMS = {m:.4f}{tag}")
    print(f"\n  single  (cand 0)      : {single:.4f}")
    print(f"  best fixed mode       : {best_single:.4f}")
    print(f"  ORACLE best-of-{n}      : {oracle:.4f}")
    print(f"  headroom (oracle-single): {oracle - single:+.4f}   "
          f"({100 * (oracle - single) / max(single, 1e-9):+.1f}% of single)")
    print(f"  headroom over best mode : {oracle - best_single:+.4f}")
    print("\n  Interpretation: headroom > 0 => candidate diversity buys real PDMS that a")
    print("  learned scorer (Increment 3) could recover. ~0 => generator not diverse enough")
    print("  on this split, or all modes already near-equal under PDMS.")


if __name__ == "__main__":
    main()