#!/usr/bin/env python
"""
Reconstruct the converged N=8 multi-mode checkpoint from the released Stage-2 base +
the tiny trained delta (the inverse of extract_trainable_delta.py). Needed because the
delta was extracted on GCP and only it + figures were worth carrying back (12GB base
checkpoint is already present on every machine as the released download).

Run (unidrivevla env):
  python apply_trainable_delta.py <released_stage2.pt> <delta.pt> <out.pt>

The output keeps the released checkpoint's full DeepSpeed dict structure (module/
optimizer/param_shapes/...) with the delta's ~21 tensors overlaid into `module`, so it
loads via the same `load_from=<out.pt>` path every other fine-tune config already uses.
"""
import sys
import torch


def main(base_path, delta_path, out_path):
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    delta = torch.load(delta_path, map_location="cpu", weights_only=False)
    module = base["module"] if isinstance(base, dict) and "module" in base else base
    overwritten = sum(1 for k in delta if k in module)
    added = sum(1 for k in delta if k not in module)
    module.update(delta)
    torch.save(base, out_path)
    print(f"merged {len(delta)} delta tensors ({overwritten} overwritten, {added} new) "
          f"into a {len(module)}-key module -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
