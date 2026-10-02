"""Train the learned PDMS scorer with per-sub-score BCE, then run the ranking
go/no-go: does picking the candidate whose *predicted* sub-scores reconstruct the
highest EPDMS beat picking at random, on the data we trained on?

If BCE stays flat OR selection does not beat random, phi_s is not informative —
stop and revisit the Task 2 dump (e.g. add ego state) before generating 5k scenes.

Run:  micromamba run -n unidrivevla python train_scorer.py <dataset.pt> <scorer.pt>
"""
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scorer_model import Scorer
from epdms import SUBSCORE_KEYS, reconstruct_epdms


def collate(batch):
    """Right-pad phi to the batch's max T_s; return a key-padding mask (True=pad)."""
    t_s = max(r["phi"].shape[0] for r in batch)
    c = batch[0]["phi"].shape[1]
    phi = torch.zeros(len(batch), t_s, c)
    mask = torch.ones(len(batch), t_s, dtype=torch.bool)
    for i, r in enumerate(batch):
        n = r["phi"].shape[0]
        phi[i, :n] = r["phi"]
        mask[i, :n] = False
    traj = torch.stack([r["traj"] for r in batch])
    y = torch.stack([r["y"] for r in batch])
    return phi, traj, y, mask


def _ranking_check(model, ds, device):
    """For each scene, pick argmax predicted-EPDMS candidate; compare its TRUE EPDMS
    to the scene's mean candidate EPDMS (= expected EPDMS of a random pick)."""
    by_scene = {}
    for r in ds:
        by_scene.setdefault(r["token"], []).append(r)
    sel, rand = [], []
    model.eval()
    with torch.no_grad():
        for recs in by_scene.values():
            if len(recs) < 2:
                continue
            true_e = np.array([reconstruct_epdms(dict(zip(SUBSCORE_KEYS, r["y"].tolist())))
                               for r in recs])
            phi = torch.nn.utils.rnn.pad_sequence([r["phi"] for r in recs],
                                                  batch_first=True).to(device)
            mask = torch.ones(phi.shape[0], phi.shape[1], dtype=torch.bool, device=device)
            for i, r in enumerate(recs):
                mask[i, :r["phi"].shape[0]] = False
            traj = torch.stack([r["traj"] for r in recs]).to(device)
            pred = model(phi, traj, key_padding_mask=mask).cpu()
            pred_e = np.array([reconstruct_epdms(dict(zip(SUBSCORE_KEYS, p.tolist())))
                               for p in pred])
            sel.append(true_e[int(pred_e.argmax())])
            rand.append(true_e.mean())
    return float(np.mean(sel)), float(np.mean(rand)), len(sel)


def main(ds_path, out, epochs=200, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = torch.load(ds_path, weights_only=True)
    c_in = ds[0]["phi"].shape[1]
    k = len(SUBSCORE_KEYS)
    dl = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=collate)
    model = Scorer(c_in, k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    for e in range(epochs):
        model.train()
        tot = 0.0
        for phi, traj, y, mask in dl:
            phi, traj, y, mask = phi.to(device), traj.to(device), y.to(device), mask.to(device)
            opt.zero_grad()
            loss = bce(model(phi, traj, key_padding_mask=mask), y)
            loss.backward()
            opt.step()
            tot += loss.item()
        if e % 20 == 0 or e == epochs - 1:
            print(f"epoch {e:4d}  bce {tot / len(dl):.4f}", flush=True)

    sel, rand, n = _ranking_check(model, ds, device)
    print(f"ranking go/no-go on {n} scenes:  select_mean={sel:.4f}  random_mean={rand:.4f}  "
          f"lift={sel - rand:+.4f}", flush=True)
    if sel <= rand:
        print("WARNING: selection does not beat random -- phi_s may be uninformative "
              "(revisit Task 2 before scaling).", flush=True)
    torch.save({"state": model.state_dict(), "c_in": c_in, "k": k}, out)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])