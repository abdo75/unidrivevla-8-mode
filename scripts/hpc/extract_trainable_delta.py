#!/usr/bin/env python
"""
Extract ONLY the fine-tuned params (the freeze_except modules: mode embeddings + action
head) from a 12 GB DeepSpeed model_states.pt into a tiny .pt (~a few MB). The rest of the
checkpoint is the frozen 2.7B backbone = the released Stage-2 checkpoint, so it doesn't need
moving. On the target machine, reconstruct the fine-tuned model by loading the released
checkpoint then `model.load_state_dict(this_delta, strict=False)`.

Run (unidrivevla env):  python extract_trainable_delta.py <model_states.pt> <out_delta.pt>
"""
import sys
import torch

# the freeze_except substrings from unidrivevla_stage2_2b_modes_ft.py — the only trained parts
KEEP = ("planning_head.action_in_proj", "planning_head.action_out_proj",
        "planning_head.action_time_mlp", "planning_head.status_mlp",
        "planning_head.hist_traj_encoder", "planning_head.mode_query")


def main(src, out):
    # weights_only=False: our own trusted DeepSpeed checkpoint (holds non-tensor objects).
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck
    for k in ("state_dict", "module", "model"):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    delta = {k: v for k, v in sd.items()
             if any(s in k for s in KEEP) and not k.startswith("ema_")}
    if not delta:
        raise SystemExit("!! no trainable-delta keys matched — is this the right checkpoint?")
    torch.save(delta, out)
    n = sum(v.numel() for v in delta.values())
    print(f"kept {len(delta)} tensors ({n/1e6:.2f}M params) -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
