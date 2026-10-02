"""
Increment 2 diagnostic: dump the N candidate trajectories per scene and report
whether the modes are *behaviorally* diverse (useful) or just numerical jitter /
degenerate. Runs forward_test(return_all_modes=True) on a handful of samples.

Usage (on a GPU node, env vars VLM_PRETRAINED_PATH / OCCWORLD_VAE_PATH set):
  python tools/dump_modes.py CONFIG CHECKPOINT [--num-samples K] [--out FILE.pkl]

CHECKPOINT may be:
  - a DeepSpeed checkpoint dir   (work_dirs/overfit_modes_mini/iter_150)
  - a DeepSpeed model_states.pt  (.../global_step5/mp_rank_00_model_states.pt)
  - a flat state_dict .pt        (released checkpoint)
"""
import argparse
import os
import os.path as osp
import pickle
import sys

# Make the project root (nuScenes/, the parent of tools/) importable so
# `projects.mmdet3d_plugin...` resolves even when run as `python tools/dump_modes.py`
# (which otherwise only puts tools/ on sys.path). Mirrors dist_*.sh PYTHONPATH=...
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def _find_model_states(path):
    """Resolve a checkpoint path to the file torch.load should read."""
    if osp.isfile(path):
        return path
    if osp.isdir(path):
        # DeepSpeed layout: <dir>/[iter_N/]global_step*/mp_rank_00_model_states.pt
        for root, _dirs, files in os.walk(path):
            for f in files:
                if f.endswith("model_states.pt"):
                    return osp.join(root, f)
    raise FileNotFoundError(f"no model_states.pt under {path}")


def _extract_state_dict(ckpt):
    # DeepSpeed model_states.pt stores the model under 'module'; mmcv ckpts under
    # 'state_dict'; a flat dump is the state dict itself.
    if isinstance(ckpt, dict) and "module" in ckpt and isinstance(ckpt["module"], dict):
        sd = ckpt["module"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def _summarize(traj_modes):
    """traj_modes: (N, T, 2) numpy. Return a one-line spread summary."""
    endpoints = traj_modes[:, -1, :]                       # (N, 2) final 3s point
    n = endpoints.shape[0]
    # pairwise endpoint distances
    d = np.sqrt(((endpoints[:, None, :] - endpoints[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(n, k=1)
    pair = d[iu]
    std_xy = endpoints.std(axis=0)                         # per-coord std
    return dict(
        endpoint_std=float(np.linalg.norm(std_xy)),
        pair_mean=float(pair.mean()),
        pair_max=float(pair.max()),
        endpoints=endpoints,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=None, help="denoise ODE steps (default: model's)")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--out", default=None, help="pickle to save {idx: traj_modes (N,T,2)}")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    # Register the custom plugin modules (detector, dataset, head, ...) so the
    # registries know 'UniDriveVLA' / 'NuScenes3DDataset' etc. (mirrors tools/test.py).
    if getattr(cfg, "plugin", False):
        import importlib
        plugin_dir = getattr(cfg, "plugin_dir", osp.dirname(args.config))
        _module_path = ".".join(osp.dirname(plugin_dir).split("/"))
        importlib.import_module(_module_path)

    ds_cfg = cfg.data[args.split]
    if isinstance(ds_cfg, dict):
        ds_cfg = dict(ds_cfg)
        ds_cfg["test_mode"] = True
    dataset = build_dataset(ds_cfg)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    ms = _find_model_states(args.checkpoint)
    print(f"[dump_modes] loading weights from {ms}", flush=True)
    # Prefer the safe weights_only=True (DeepSpeed model_states.pt is pure tensors and
    # loads fine this way). Fall back to weights_only=False ONLY for our own trusted
    # local checkpoints if a non-tensor global is present (arbitrary-code-exec risk
    # otherwise — never do this for an untrusted file).
    try:
        ckpt = torch.load(ms, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"[dump_modes] weights_only=True failed ({type(e).__name__}); retrying "
              f"weights_only=False on this trusted local checkpoint.", flush=True)
        ckpt = torch.load(ms, map_location="cpu", weights_only=False)
    sd = _extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    has_mode = any("mode_query" in k for k in sd)
    print(f"[dump_modes] loaded: {len(sd)} keys; missing={len(missing)} unexpected={len(unexpected)}; "
          f"mode_query in ckpt={has_mode}", flush=True)
    num_modes = getattr(model.planning_head, "num_modes", 1)
    print(f"[dump_modes] model num_modes={num_modes}", flush=True)
    if num_modes <= 1:
        print("[dump_modes] WARNING: num_modes<=1 -> no candidates to dump (check the config).", flush=True)

    model = MMDataParallel(model.cuda().eval(), device_ids=[0])

    saved = {}
    print("\nidx | endpoint_std(m) | pair_mean(m) | pair_max(m) | per-mode endpoints (x,y)", flush=True)
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= args.num_samples:
                break
            kw = dict(return_loss=False, return_all_modes=True)
            if args.num_steps is not None:
                kw["num_steps"] = args.num_steps
            result = model(**kw, **data)
            tm = result[0]["img_bbox"].get("traj_modes")
            if tm is None:
                print(f"{i:3d} | <no traj_modes returned> (num_modes={num_modes}, return_all_modes path?)", flush=True)
                continue
            tm = tm.numpy() if torch.is_tensor(tm) else np.asarray(tm)  # (N, T, 2)
            saved[i] = tm
            s = _summarize(tm)
            eps = "  ".join(f"({x:+.2f},{y:+.2f})" for x, y in s["endpoints"])
            print(f"{i:3d} | {s['endpoint_std']:14.3f} | {s['pair_mean']:11.3f} | {s['pair_max']:10.3f} | {eps}", flush=True)

    if args.out and saved:
        with open(args.out, "wb") as f:
            pickle.dump(saved, f)
        print(f"\n[dump_modes] saved {len(saved)} samples -> {args.out}", flush=True)

    if saved:
        allstd = np.mean([_summarize(tm)["endpoint_std"] for tm in saved.values()])
        allmax = np.mean([_summarize(tm)["pair_max"] for tm in saved.values()])
        print(f"\n[dump_modes] across {len(saved)} samples: mean endpoint_std={allstd:.3f} m, "
              f"mean pair_max={allmax:.3f} m", flush=True)
        print("[dump_modes] interpretation: endpoint_std/pair_max ~cm => degenerate (modes ~identical); "
              "~meters => behaviorally diverse candidates.", flush=True)


if __name__ == "__main__":
    main()