"""
Stage B (DrivoR thesis) — run UniDriveVLA on the Stage-A dumps.
Runs in the `unidrivevla` env (torch-2.7 / Blackwell). Reads one .npz per token
(from Stage A), runs a single forward_test, and saves {token: traj(6,2)} plus
prints each trajectory so we can eyeball the frame (straight scene => x grows
forward, y small). Multi-candidate (noise=) comes later; this is 1 traj/token.

Env (set by the wrapper): CONFIG, CKPT, STAGE_A_IN, STAGE_B_OUT, optional LIMIT.
Must run with cwd on the nuScenes dir so `import projects.mmdet3d_plugin` works.
"""
import os
import glob
import pickle
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

import sys
sys.path.insert(0, os.getcwd())  # nuScenes dir (wrapper cd's here) so `projects` resolves
import projects.mmdet3d_plugin  # noqa: F401,E402 — registers custom detector/heads

CONFIG = os.environ["CONFIG"]
CKPT = os.environ["CKPT"]
STAGE_A_IN = Path(os.environ["STAGE_A_IN"])
STAGE_B_OUT = Path(os.environ["STAGE_B_OUT"])
LIMIT = int(os.environ.get("LIMIT", "0"))
N_CANDIDATES = int(os.environ.get("N_CANDIDATES", "1"))  # >1 = multi-candidate (re-seed noise)
CMD_SWEEP = os.environ.get("CMD_SWEEP", "0") == "1"      # run right/left/straight (noise fixed)
# Increment 2: candidates from the LEARNED modes (return_all_modes=True) instead of
# re-seeding noise. Requires a num_modes>1 config + a multi-mode checkpoint. Preferred
# candidate source (proven behaviorally diverse ~0.7 m; noise gave only ~0.1 m).
MODE_CANDIDATES = os.environ.get("MODE_CANDIDATES", "0") == "1"
# Increment 3: also dump the per-scene perception tokens (phi_s) the learned scorer
# attends over. One float16 .npy per token under STAGE_B_OUT/phi_s/, plus a meta json
# with the channel dim. Only meaningful alongside MODE_CANDIDATES (need the candidates).
DUMP_SCENE_TOKENS = os.environ.get("DUMP_SCENE_TOKENS", "0") == "1"
# NavSim loads RGB; the model swaps channels assuming BGR. Swap once here.
NAVSIM_IMG_IS_RGB = os.environ.get("NAVSIM_IMG_IS_RGB", "1") == "1"


def _load_weights(model, ckpt_path):
    """Load a flat .pt (mmcv) OR a DeepSpeed checkpoint dir / model_states.pt."""
    p = Path(ckpt_path)
    ms = None
    if p.is_dir():
        for root, _d, files in os.walk(p):
            for f in files:
                if f.endswith("model_states.pt"):
                    ms = os.path.join(root, f)
                    break
            if ms:
                break
    elif p.name.endswith("model_states.pt"):
        ms = str(p)
    if ms is None:
        load_checkpoint(model, str(ckpt_path), map_location="cpu")  # flat .pt
        return
    print(f"[stage_b] DeepSpeed checkpoint -> {ms}")
    try:
        ckpt = torch.load(ms, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001 — trusted local checkpoint fallback
        print(f"[stage_b] weights_only=True failed ({type(e).__name__}); retrying False")
        ckpt = torch.load(ms, map_location="cpu", weights_only=False)
    sd = ckpt.get("module", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[stage_b] loaded: missing={len(missing)} unexpected={len(unexpected)} "
          f"mode_query_in_ckpt={any('mode_query' in k for k in sd)}")


def build():
    cfg = Config.fromfile(CONFIG)
    cfg.model.train_cfg = None
    if "pretrained" in cfg.model:
        cfg.model.pretrained = None  # Stage-2 ckpt below provides weights
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    _load_weights(model, CKPT)
    nm = getattr(model.planning_head, "num_modes", 1)
    print(f"[stage_b] num_modes={nm}  MODE_CANDIDATES={MODE_CANDIDATES}")
    if MODE_CANDIDATES and nm <= 1:
        print("[stage_b] WARNING: MODE_CANDIDATES=1 but num_modes<=1 -> only 1 candidate.")
    return model.cuda().eval()


def make_data(npz):
    d = np.load(npz)
    img = d["img"]  # (6,544,960,3) uint8
    if NAVSIM_IMG_IS_RGB:
        img = img[..., ::-1]  # -> BGR (model swaps back to RGB internally)
    img = np.ascontiguousarray(img)
    img_t = torch.from_numpy(img).permute(0, 3, 1, 2).unsqueeze(0).float()  # (1,6,3,H,W)
    g = lambda k: torch.from_numpy(d[k]).unsqueeze(0).float().cuda()
    return dict(
        img=img_t.cuda(),
        projection_mat=g("projection_mat"),          # (1,6,4,4)
        image_wh=g("image_wh"),                       # (1,6,2)
        hist_traj=g("hist_traj"),                     # (1,4,2)
        # model uses gt_ego_fut_cmd as a (B,3) one-hot (status_in_features = 3 + ego_dim)
        gt_ego_fut_cmd=torch.eye(3, dtype=torch.float32)[int(d["command"])].unsqueeze(0).cuda(),
        ego_status=None,                              # model predicts its own at inference
        timestamp=torch.tensor([0], dtype=torch.long).cuda(),
        img_metas=[{"T_global": np.eye(4, dtype=np.float32),
                    "T_global_inv": np.eye(4, dtype=np.float32), "timestamp": 0}],
    )


def main():
    files = sorted(glob.glob(str(STAGE_A_IN / "*.npz")))
    if LIMIT:
        files = files[:LIMIT]
    print(f"Stage B: {len(files)} tokens  in={STAGE_A_IN}  out={STAGE_B_OUT}")
    model = build()
    STAGE_B_OUT.mkdir(parents=True, exist_ok=True)
    phi_s_dir = STAGE_B_OUT / "phi_s"
    phi_s_meta = {"c_in": None, "n_tokens": 0}
    if DUMP_SCENE_TOKENS:
        phi_s_dir.mkdir(parents=True, exist_ok=True)
        print(f"[stage_b] DUMP_SCENE_TOKENS=1 -> {phi_s_dir}")
    trajs = {}
    ok = fail = 0
    for i, f in enumerate(files):
        token = Path(f).stem
        try:
            data = make_data(f)
            if MODE_CANDIDATES:
                # Increment 2: N candidates from the learned modes in ONE forward.
                with torch.no_grad():
                    out = model(return_loss=False, rescale=True, return_all_modes=True,
                                return_scene_tokens=DUMP_SCENE_TOKENS, **data)
                tm = out[0]["img_bbox"].get("traj_modes")
                if tm is None:
                    raise RuntimeError("no traj_modes returned (num_modes<=1 or detector not updated)")
                arr = np.asarray(tm if not torch.is_tensor(tm) else tm.cpu())  # (N,6,2)
                trajs[token] = arr
                if DUMP_SCENE_TOKENS:
                    st = out[0]["img_bbox"].get("scene_tokens")
                    if st is None:
                        raise RuntimeError("no scene_tokens returned (head/detector not updated)")
                    st = np.asarray(st if not torch.is_tensor(st) else st.cpu(), dtype=np.float16)
                    np.save(phi_s_dir / f"{token}.npy", st)  # (T_s, C)
                    phi_s_meta["c_in"] = int(st.shape[-1])
                    phi_s_meta["n_tokens"] += 1
                if i < 5:
                    ends = arr[:, -1, :]
                    sx, sy = ends.std(axis=0)
                    print(f"  [{i}] {token}: modes={arr.shape[0]} "
                          + " ".join(f"({x:+.1f},{y:+.1f})" for x, y in ends)
                          + f"  endpoint_std=({sx:.2f},{sy:.2f})")
            elif CMD_SWEEP:
                # vary command (right/left/straight) with noise FIXED -> isolate command effect
                gen = torch.Generator(device="cuda").manual_seed(1000)
                noise = torch.randn(1, 6, 2, generator=gen, device="cuda", dtype=torch.float32)
                res = {}
                for cmd, name in ((0, "right"), (1, "left"), (2, "straight")):
                    data["gt_ego_fut_cmd"] = torch.eye(3, dtype=torch.float32)[cmd].unsqueeze(0).cuda()
                    data["noise"] = noise
                    with torch.no_grad():
                        out = model(return_loss=False, rescale=True, **data)
                    res[name] = np.asarray(out[0]["img_bbox"]["final_planning"])
                trajs[token] = res
                if i < 5:  # endpoint (x=sideways, y=forward) per command
                    msg = "  ".join(f"{n}:({t[-1,0]:+.1f},{t[-1,1]:+.1f})" for n, t in res.items())
                    print(f"  [{i}] {token}: {msg}")
            elif N_CANDIDATES <= 1:
                # single trajectory: let the model sample its own noise (the 0.289 path)
                with torch.no_grad():
                    out = model(return_loss=False, rescale=True, **data)
                trajs[token] = np.asarray(out[0]["img_bbox"]["final_planning"])  # (6,2)
                if i < 3:
                    pts = ", ".join(f"({x:+.1f},{y:+.1f})" for x, y in trajs[token])
                    print(f"  [{i}] {token}: {pts}")
            else:
                # N candidates: re-seed the flow-matching noise (seeds fixed for reproducibility)
                cand = []
                for k in range(N_CANDIDATES):
                    gen = torch.Generator(device="cuda").manual_seed(1000 + k)
                    data["noise"] = torch.randn(1, 6, 2, generator=gen, device="cuda", dtype=torch.float32)
                    with torch.no_grad():
                        out = model(return_loss=False, rescale=True, **data)
                    cand.append(np.asarray(out[0]["img_bbox"]["final_planning"]))
                arr = np.stack(cand)  # (N,6,2)
                trajs[token] = arr
                if i < 5:  # diversity readout: spread of the 6th (final) point across candidates
                    ends = arr[:, -1, :]
                    sx, sy = ends.std(axis=0)
                    print(f"  [{i}] {token}: N={N_CANDIDATES} endpoints "
                          + " ".join(f"({x:+.1f},{y:+.1f})" for x, y in ends)
                          + f"  endpoint_std=({sx:.2f},{sy:.2f})")
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  [{i}] {token}: FAILED {type(e).__name__}: {e}")
            if fail == 1:
                import traceback
                traceback.print_exc()
    with open(STAGE_B_OUT / "trajectories.pkl", "wb") as fh:
        pickle.dump(trajs, fh)
    if DUMP_SCENE_TOKENS:
        import json
        with open(STAGE_B_OUT / "phi_s_meta.json", "w") as fh:
            json.dump(phi_s_meta, fh)
        print(f"[stage_b] phi_s: {phi_s_meta['n_tokens']} tokens, c_in={phi_s_meta['c_in']} -> {phi_s_dir}")
    print(f"=== Stage B done: {ok} ok, {fail} failed -> {STAGE_B_OUT}/trajectories.pkl ===")


if __name__ == "__main__":
    main()
