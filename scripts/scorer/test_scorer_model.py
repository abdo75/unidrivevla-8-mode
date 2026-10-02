"""Shape/range check for Scorer.  Run:  python test_scorer_model.py"""
import torch
from scorer_model import Scorer
from epdms import SUBSCORE_KEYS


def main():
    k = len(SUBSCORE_KEYS)
    m = Scorer(c_in=256, k=k)
    phi = torch.randn(3, 400, 256)
    traj = torch.randn(3, 6, 2)
    out = m(phi, traj)
    assert out.shape == (3, k), out.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0, (out.min(), out.max())
    # padding mask path
    mask = torch.zeros(3, 400, dtype=torch.bool); mask[:, 300:] = True
    out2 = m(phi, traj, key_padding_mask=mask)
    assert out2.shape == (3, k)
    print("OK: Scorer forward -> (3, %d) in [0,1], padding-mask path works" % k)


if __name__ == "__main__":
    main()