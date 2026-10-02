"""K-fold cross-validated candidate selection on a single split (the navhard headline).

Folds are isolated BY SCENE (token): for each fold, a fresh scorer trains on the other
folds and selects candidates for the held-out tokens, so every token's pick is
out-of-fold (no leakage). The pooled {token: best_idx} JSON covers all tokens and is
scored once with the real EPDMS by stage_c + run_pdm_score (via run_scorer_navhard.sh).

Also prints the pooled out-of-fold ranking summary against the oracle ceiling.

Run:  python cv_select.py <dataset.pt> <selection.json> [k] [epochs]
"""
import sys
import json

import numpy as np
import torch
import torch.nn as nn

from scorer_model import Scorer
from epdms import SUBSCORE_KEYS, MULTIPLICATIVE, WEIGHTED, reconstruct_epdms

# Differentiable EPDMS over a (M, K) sub-score tensor, same formula as reconstruct_epdms:
# product of the multiplicative terms times the weighted average of the rest. Lets the
# selection objective backprop through the reconstructed score.
_NMULT = len(MULTIPLICATIVE)
_W = torch.tensor(list(WEIGHTED.values()), dtype=torch.float32)
_DENOM = float(sum(WEIGHTED.values()))


def epdms_torch(pred):
    mult = pred[:, :_NMULT].prod(dim=1)
    w = _W.to(pred.device)
    wsum = (pred[:, _NMULT:] * w).sum(dim=1) / _DENOM
    return mult * wsum


def _pack_scenes(records, device):
    """Group candidate records by scene -> list of (phi (Ts,C), traj (N,T,2), y (N,K),
    true_e (N,), cand_ids (N,)). All candidates of a scene share one phi."""
    by = {}
    for r in records:
        by.setdefault(r["token"], []).append(r)
    scenes = []
    for tok, recs in by.items():
        y = torch.stack([r["y"] for r in recs])
        true_e = epdms_torch(y).numpy()
        scenes.append({
            "phi": recs[0]["phi"], "traj": torch.stack([r["traj"] for r in recs]),
            "y": y, "true_e": true_e, "cand": [int(r["cand"]) for r in recs],
            "disc": float(true_e.max() - true_e.min()),
        })
    return scenes


def _train(records, c_in, device, epochs, lr=1e-3, lam=1.0, margin=0.02, chunk=64,
           pool="attn", d=128):
    """BCE on all candidates (calibrates sub-scores) + within-scene pairwise hinge that
    pushes the oracle candidate's reconstructed EPDMS above its siblings (only on scenes
    that actually have a winner). Selection is ranking, so a ranking term is essential.
    pool/d/lam are ablation knobs (lam=0 -> BCE only)."""
    model = Scorer(c_in, len(SUBSCORE_KEYS), d=d, pool=pool).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    scenes = _pack_scenes(records, device)
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(scenes))
        for c0 in range(0, len(order), chunk):
            batch = [scenes[i] for i in order[c0:c0 + chunk]]
            max_ts = max(s["phi"].shape[0] for s in batch)
            phis, trajs, ys, slices, p = [], [], [], [], 0
            for s in batch:
                n, ts = s["traj"].shape[0], s["phi"].shape[0]
                pad = torch.zeros(n, max_ts, s["phi"].shape[1])
                pad[:, :ts] = s["phi"].unsqueeze(0).expand(n, -1, -1)
                phis.append(pad); trajs.append(s["traj"]); ys.append(s["y"])
                slices.append((p, p + n, ts)); p += n
            phi = torch.cat(phis).to(device)
            traj = torch.cat(trajs).to(device)
            y = torch.cat(ys).to(device)
            mask = torch.ones(phi.shape[0], max_ts, dtype=torch.bool, device=device)
            for (a, b, ts) in slices:
                mask[a:b, :ts] = False
            opt.zero_grad()
            pred = model(phi, traj, key_padding_mask=mask)
            loss = bce(pred, y)
            pe = epdms_torch(pred)
            rank, ndisc = pred.new_zeros(()), 0
            for s, (a, b, _ts) in zip(batch, slices):
                if s["disc"] <= 1e-6:
                    continue
                o = a + int(np.argmax(s["true_e"]))
                others = [j for j in range(a, b) if j != o]
                rank = rank + torch.relu(margin - (pe[o] - pe[others])).mean()
                ndisc += 1
            if ndisc:
                loss = loss + lam * rank / ndisc
            loss.backward()
            opt.step()
    return model


def _epdms(vec):
    return reconstruct_epdms(dict(zip(SUBSCORE_KEYS, [float(x) for x in vec])))


def cv_select(ds_path, out_json, k=5, epochs=200, seed=0, tau=0.0,
              n_cand=None, lam=1.0, pool="attn", d=128):
    """tau: selective-override margin. Keep the default mode (cand 0) unless the scorer's
    best candidate beats cand0's *predicted* EPDMS by more than tau. tau=0 is pure argmax.
    tau must be tuned on the dev split (warmup), never on navhard.
    Ablation knobs: n_cand (use only candidates 0..n_cand-1), lam (ranking weight; 0=BCE
    only), pool ('attn'/'mean'/'trajonly'), d (scorer width)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = torch.load(ds_path, weights_only=True)
    if n_cand is not None:
        ds = [r for r in ds if int(r["cand"]) < n_cand]
    c_in = ds[0]["phi"].shape[1]

    by_scene = {}
    for r in ds:
        by_scene.setdefault(r["token"], []).append(r)
    tokens = sorted(by_scene)
    rng = np.random.default_rng(seed)
    rng.shuffle(tokens)
    folds = [tokens[i::k] for i in range(k)]                  # round-robin assignment

    selection = {}
    # per scene: true EPDMS of (selected-by-argmax, cand0, random/mean, oracle) + the
    # predicted argmax-over-cand0 margin, so we can evaluate selective override at any tau.
    argmax_e, c0_e, rand_e, oracle_e, margins, c0_true = [], [], [], [], [], []
    for fi, test_tokens in enumerate(folds):
        test_set = set(test_tokens)
        train_recs = [r for r in ds if r["token"] not in test_set]
        model = _train(train_recs, c_in, device, epochs, lam=lam, pool=pool, d=d)
        model.eval()
        with torch.no_grad():
            for tok in test_tokens:
                recs = by_scene[tok]
                phi = torch.nn.utils.rnn.pad_sequence([r["phi"] for r in recs],
                                                      batch_first=True).to(device)
                mask = torch.ones(phi.shape[0], phi.shape[1], dtype=torch.bool, device=device)
                for i, r in enumerate(recs):
                    mask[i, :r["phi"].shape[0]] = False
                traj = torch.stack([r["traj"] for r in recs]).to(device)
                pred = model(phi, traj, key_padding_mask=mask).cpu().numpy()
                true_e = np.array([_epdms(r["y"]) for r in recs])
                pred_e = np.array([_epdms(p) for p in pred])
                cands = [int(r["cand"]) for r in recs]
                c0 = cands.index(0) if 0 in cands else int(true_e.argmax())  # default-mode row
                am = int(pred_e.argmax())
                # selective override: switch away from cand0 only if the predicted gain clears tau
                pick = am if (pred_e[am] - pred_e[c0]) > tau else c0
                selection[tok] = int(recs[pick]["cand"])
                argmax_e.append(true_e[am]); c0_e.append(true_e[c0])
                rand_e.append(true_e.mean()); oracle_e.append(true_e.max())
                margins.append(pred_e[am] - pred_e[c0]); c0_true.append(true_e[c0])
        print(f"  fold {fi}: {len(test_tokens)} held-out tokens done", flush=True)

    with open(out_json, "w") as fh:
        json.dump(selection, fh)
    argmax_e = np.array(argmax_e); c0_e = np.array(c0_e); rand_e = np.array(rand_e)
    oracle_e = np.array(oracle_e); margins = np.array(margins)
    rand_m, c0_m, orc_m = rand_e.mean(), c0_e.mean(), oracle_e.mean()
    # selective-override mean at the chosen tau (and a small sweep for diagnosis only)
    def override_mean(t):
        return np.where(margins > t, argmax_e, c0_e).mean()
    sel_m = override_mean(tau)
    print(f"=== {k}-fold CV on {len(tokens)} scenes ===", flush=True)
    print(f"  baselines: random={rand_m:.4f}  cand0(default)={c0_m:.4f}  oracle={orc_m:.4f}", flush=True)
    print(f"  pure-argmax select={argmax_e.mean():.4f}  "
          f"recovered vs random={(argmax_e.mean()-rand_m)/(orc_m-rand_m+1e-9)*100:.1f}%  "
          f"vs cand0={(argmax_e.mean()-c0_m)/(orc_m-c0_m+1e-9)*100:.1f}%", flush=True)
    print(f"  selective-override (tau={tau:.3f}) select={sel_m:.4f}  "
          f"vs cand0={(sel_m-c0_m)/(orc_m-c0_m+1e-9)*100:+.1f}% of deployable headroom", flush=True)
    print("  tau sweep (DIAGNOSTIC ONLY -- tune on warmup, not here): "
          + "  ".join(f"{t:.2f}:{override_mean(t):.4f}" for t in (0.0, 0.01, 0.02, 0.05, 0.1)), flush=True)
    print(f"  wrote {len(selection)} selections -> {out_json}", flush=True)
    return len(selection)


if __name__ == "__main__":
    import os
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    tau = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    # ablation knobs via env (keeps the positional CLI stable)
    nc = os.environ.get("N_CAND"); lam = os.environ.get("LAM")
    cv_select(sys.argv[1], sys.argv[2], k, epochs, tau=tau,
              n_cand=int(nc) if nc else None,
              lam=float(lam) if lam is not None else 1.0,
              pool=os.environ.get("POOL", "attn"),
              d=int(os.environ.get("SCORER_D", "128")))