#!/usr/bin/env python
"""
Validate stage_a_train.gt_ego_future's sign/frame conversion WITHOUT navtrain — reuse the
navhard loader (works wherever navhard data is, e.g. GCP) on scenes that have a real future
(the stage-1 scenes) and check the converted target against the raw NavSim future.

Expected (inverse of stage_c to_traj, LATERAL_SIGN=-1):
  uni_forward (deltas, comp 1) cumulative  ==  NavSim future x (forward), and POSITIVE for
      a normally-moving car;  uni_lateral (comp 0) cumulative  ==  -NavSim future y (left).

Run (navsim env, navhard data present):
  TRAIN_TEST_SPLIT=navhard_two_stage python verify_gt_conversion.py
"""
import numpy as np

from render_navhard_examples import build_loader   # navhard two-stage loader
from stage_a_train import gt_ego_future, N_FUT


def main():
    loader = build_loader()
    checked = 0
    print(f"{'token':>18}  {'navsim_fwd':>10} {'navsim_left':>11} | {'uni_fwd':>8} {'uni_lat':>8}  ok?")
    for tok in loader.tokens:
        if checked >= 12:
            break
        scene = loader.get_scene_from_token(tok)
        n_hist = scene.scene_metadata.num_history_frames
        n = min(N_FUT, len(scene.frames) - n_hist)
        if n <= 0:
            continue                                   # synthetic/short scene, no GT future
        poses = np.asarray(scene.get_future_trajectory(n).poses, dtype=np.float64)  # (n,3) x=fwd,y=left
        trajs, masks, _ = gt_ego_future(scene, 2)
        uni_fwd = float(trajs[:, 1].sum())             # cumulative forward (deltas summed)
        uni_lat = float(trajs[:, 0].sum())             # cumulative lateral
        ns_fwd, ns_left = float(poses[-1, 0]), float(poses[-1, 1])
        # forward must match NavSim x; lateral must be -NavSim y (LATERAL_SIGN=-1)
        ok = abs(uni_fwd - ns_fwd) < 0.5 and abs(uni_lat + ns_left) < 0.5
        print(f"{tok[:18]:>18}  {ns_fwd:>10.2f} {ns_left:>11.2f} | {uni_fwd:>8.2f} {uni_lat:>8.2f}  {'PASS' if ok else 'FAIL'}")
        checked += 1
    print(f"\nchecked {checked} scenes. All PASS => the sign/frame conversion is correct; "
          f"forward should be positive for moving cars.")


if __name__ == "__main__":
    main()
