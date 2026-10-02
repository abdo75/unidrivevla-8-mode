"""
Stage C (DrivoR thesis) — convert UniDriveVLA trajectories into a NavSim
two-stage submission pickle. Runs in the `navsim` env.

Reads Stage B's {token: (6,2)} (UniDriveVLA ego-frame, idx0=LATERAL, idx1=FORWARD
— established empirically) and writes a submission:
  {"first_stage_predictions":  [ {token: Trajectory} ],   # loader.tokens_stage_one
   "second_stage_predictions": [ {token: Trajectory} ],   # loader.reactive_tokens_stage_two
   + team metadata }
Trajectory.poses = (8,3) [x,y,heading], x=forward y=left, 4s @ 0.5s.

Conversions: axis swap (NavSim x=uni[:,1], y=LATERAL_SIGN*uni[:,0]); extend
3s/6pts -> 4s/8pts by last-velocity; heading = atan2 of successive deltas.
LATERAL_SIGN defaults -1 (verified on navhard, see the audit note below); +1 mirrors
trajectories left<->right. The straight-heavy warmup split is insensitive to it.
"""
import os
import pickle
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.common.dataloader import SceneLoader

DEVKIT = os.environ["NAVSIM_DEVKIT_ROOT"]
DATA = os.environ["OPENSCENE_DATA_ROOT"]
SPLIT = os.environ.get("TRAIN_TEST_SPLIT", "warmup_two_stage")
TS = f"{DATA}/{SPLIT}"   # v2.2: split is top-level ($DATA/<split>/{sensor_blobs,synthetic_scene_pickles})
CONFIG_DIR = f"{DEVKIT}/navsim/planning/script/config/pdm_scoring"
STAGE_B_TRAJ = Path(os.environ["STAGE_B_TRAJ"])
SUBMISSION_OUT = Path(os.environ["SUBMISSION_OUT"])
# Lateral-sign convention for the (lateral, forward) -> NAVSIM (x=fwd, y=left) map.
# AUDIT 2026-06-29: on navhard, sign=-1 scores 0.177 vs sign=+1 0.116 official EPDMS
# (+53%) -- +1 mirrored trajectories left<->right and drove off the drivable area on
# turns (the straight-heavy warmup split was insensitive). -1 is the correct convention.
LATERAL_SIGN = float(os.environ.get("LATERAL_SIGN", "-1"))
# Which candidate to put in this submission when Stage B saved multiple per token
# (mode candidates -> (N,6,2)). The oracle-ceiling loop runs this once per CAND_IDX
# and takes the best PDMS per token. Ignored for single-trajectory (6,2) Stage B.
CAND_IDX = int(os.environ.get("CAND_IDX", "0"))
# Increment 3: a learned-scorer selection {token: best_candidate_idx} (pickle). When set,
# each token uses its own selected candidate instead of the fixed CAND_IDX -- this is how
# the learned scorer's chosen trajectory gets scored by the real metric.
SELECTION_PATH = os.environ.get("SELECTION", "")
N_OUT = 8


def _select(entry, cand_idx=None):
    """Pick one (6,2) trajectory from a Stage-B entry: plain (6,2), multi-candidate
    (N,6,2) -> candidate cand_idx (default CAND_IDX), or a cmd-sweep dict -> 'straight'."""
    if cand_idx is None:
        cand_idx = CAND_IDX
    arr = entry
    if isinstance(arr, dict):  # cmd-sweep {name: (6,2)}
        arr = arr.get("straight", next(iter(arr.values())))
    arr = np.asarray(arr)
    if arr.ndim == 3:          # (N,6,2) multi-candidate
        arr = arr[min(cand_idx, arr.shape[0] - 1)]
    return arr


def to_traj(uni):
    uni = np.asarray(uni, dtype=np.float64)               # (6,2) (lateral, forward)
    xy = np.stack([uni[:, 1], LATERAL_SIGN * uni[:, 0]], axis=1)  # NavSim (x=fwd, y=left)
    v = xy[-1] - xy[-2]                                    # last-step velocity
    while xy.shape[0] < N_OUT:                             # extend 6 -> 8 (3s -> 4s)
        xy = np.concatenate([xy, (xy[-1] + v)[None]], axis=0)
    prev = np.concatenate([np.zeros((1, 2)), xy[:-1]], axis=0)
    d = xy - prev
    heading = np.arctan2(d[:, 1], d[:, 0])                 # tangent of path
    poses = np.concatenate([xy, heading[:, None]], axis=1).astype(np.float32)  # (8,3)
    return Trajectory(poses)                               # default sampling = 4s @ 0.5s = 8


def main():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default_run_pdm_score", overrides=[
            f"train_test_split={SPLIT}",
            f"synthetic_sensor_path={TS}/sensor_blobs",
            f"synthetic_scenes_path={TS}/synthetic_scene_pickles",
        ])
    sf = instantiate(cfg.train_test_split.scene_filter)
    loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=sf,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    s1 = list(loader.tokens_stage_one)
    s2 = list(loader.reactive_tokens_stage_two or [])
    # pickle is safe here: STAGE_B_TRAJ is our own Stage-B output (trusted, local).
    # The submission is also pickle by NavSim's API (run_pdm_score_from_submission).
    with open(STAGE_B_TRAJ, "rb") as f:
        trajs = pickle.load(f)
    selection = {}
    if SELECTION_PATH:
        import json
        with open(SELECTION_PATH) as f:
            selection = json.load(f)  # {token: best_candidate_idx}, plain JSON (safe)
        print(f"using learned-scorer selection for {len(selection)} tokens ({SELECTION_PATH})")
    print(f"uni trajectories={len(trajs)}  stage_one={len(s1)}  stage_two={len(s2)}  LATERAL_SIGN={LATERAL_SIGN}")

    def stage(tokens):
        out, miss = {}, 0
        for t in tokens:
            if t in trajs:
                idx = int(selection[t]) if t in selection else None
                out[t] = to_traj(_select(trajs[t], idx))
            else:
                miss += 1
        if miss:
            print(f"  WARNING: {miss}/{len(tokens)} tokens missing a trajectory — run Stage B FULL (not smoke)")
        return out

    sub = {
        "team_name": "unidrivevla", "authors": ["agader"], "email": "",
        "institution": "", "country / region": "",
        "first_stage_predictions": [stage(s1)],
        "second_stage_predictions": [stage(s2)],
    }
    SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SUBMISSION_OUT, "wb") as f:
        pickle.dump(sub, f)
    print(f"=== wrote {SUBMISSION_OUT}  (s1={len(sub['first_stage_predictions'][0])}, "
          f"s2={len(sub['second_stage_predictions'][0])}) ===")


if __name__ == "__main__":
    main()
