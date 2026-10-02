"""
Stage A (DrivoR thesis) — dump UniDriveVLA-ready inputs per NavSim token.
Runs in the `navsim` env (imports navsim, NOT UniDriveVLA). For each token it
extracts the 6 nuScenes-layout cameras, resized to 544x960 with intrinsics
rescaled in lockstep, the lidar->img projection, ego history, and the command,
and writes one .npz per token. Stage B (unidrivevla env) reads these.

Decoupling contract (the .npz format):
  img             (6,544,960,3) uint8   NavSim-native channel order (see CHANNELS note)
  projection_mat  (6,4,4) float32       lidar(ego)->image for the RESIZED image
  image_wh        (6,2) float32         [960,544] per cam
  hist_traj       (4,2) float32         past ego xy in current-ego frame (frame[-1]=origin)
  command         int64                 UniDriveVLA nav cmd: 0=right,1=left,2=straight
  ego_velocity    (2,) float32          current frame (extra; for ego_status if needed)
  ego_acceleration(2,) float32          current frame (extra)
Camera order = UniDriveVLA NUSCENES_VIEW_TOKENS:
  [FRONT, FRONT_LEFT, FRONT_RIGHT, BACK_LEFT, BACK_RIGHT, BACK]

CHANNELS: NavSim returns uint8 (H,W,3) from .jpg. We dump as-loaded; Stage B
converts to UniDriveVLA's BGR+mean. If a channel swap is needed it's a 1-liner there.
"""
import os
from pathlib import Path

import numpy as np
import cv2
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader

DEVKIT = os.environ["NAVSIM_DEVKIT_ROOT"]
DATA = os.environ["OPENSCENE_DATA_ROOT"]
SPLIT = os.environ.get("TRAIN_TEST_SPLIT", "warmup_two_stage")
OUT = Path(os.environ.get("STAGE_A_OUT", f"{os.environ['NAVSIM_EXP_ROOT']}/uni_inputs/{SPLIT}"))
LIMIT = int(os.environ.get("LIMIT", "0"))  # 0 = all tokens
TS = f"{DATA}/{SPLIT}"   # v2.2: split is top-level ($DATA/<split>/{sensor_blobs,synthetic_scene_pickles})
CONFIG_DIR = f"{DEVKIT}/navsim/planning/script/config/pdm_scoring"

TARGET_W, TARGET_H = 960, 544
# NavSim 8-cam -> UniDriveVLA 6-cam (drops side cams l1,r1)
NAV2UNI_CAMS = ["cam_f0", "cam_l0", "cam_r0", "cam_l2", "cam_r2", "cam_b0"]
# NavSim driving_command one-hot [left,straight,right,unknown] -> UniDriveVLA {0:right,1:left,2:straight}
CMD_MAP = {0: 1, 1: 2, 2: 0, 3: 2}


def lidar2cam(R, t):
    """sensor2lidar (point_lidar = R@point_cam + t) -> lidar2cam 4x4."""
    Rt = R.T
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = Rt
    M[:3, 3] = -Rt @ t
    return M


def adapt(ai):
    cams = ai.cameras[-1]
    imgs, projs, whs = [], [], []
    for name in NAV2UNI_CAMS:
        cam = getattr(cams, name)
        img = np.asarray(cam.image)
        h0, w0 = img.shape[:2]
        imgs.append(cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR))
        K = np.asarray(cam.intrinsics, dtype=np.float64).copy()
        K[0, :] *= TARGET_W / w0
        K[1, :] *= TARGET_H / h0
        K4 = np.eye(4)
        K4[:3, :3] = K
        l2c = lidar2cam(np.asarray(cam.sensor2lidar_rotation, dtype=np.float64),
                        np.asarray(cam.sensor2lidar_translation, dtype=np.float64))
        projs.append((K4 @ l2c).astype(np.float32))
        whs.append([TARGET_W, TARGET_H])
    cur = ai.ego_statuses[-1]
    hist = np.array([np.asarray(es.ego_pose)[:2] for es in ai.ego_statuses], dtype=np.float32)
    cmd = CMD_MAP.get(int(np.asarray(cur.driving_command).argmax()), 2)
    return dict(
        img=np.stack(imgs).astype(np.uint8),
        projection_mat=np.stack(projs),
        image_wh=np.array(whs, dtype=np.float32),
        hist_traj=hist,
        command=np.int64(cmd),
        ego_velocity=np.asarray(cur.ego_velocity, dtype=np.float32),
        ego_acceleration=np.asarray(cur.ego_acceleration, dtype=np.float32),
    )


def main():
    overrides = [
        f"train_test_split={SPLIT}",
        # stage-1 original scenes come from the standard test split: navsim_log_path=
        # navsim_logs/test and original_sensor_path=sensor_blobs/test are the hydra defaults
        # for data_split=test, so we don't override them (fetch_navsim_data.sh downloads the
        # navhard logs' test cameras into sensor_blobs/test). stage-2 synthetic scenes use
        # navhard's own bundle:
        f"synthetic_sensor_path={TS}/sensor_blobs",
        f"synthetic_scenes_path={TS}/synthetic_scene_pickles",
    ]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default_run_pdm_score", overrides=overrides)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig(
            cam_f0=True, cam_l0=True, cam_l1=False, cam_l2=True,
            cam_r0=True, cam_r1=False, cam_r2=True, cam_b0=True, lidar_pc=False,
        ),
    )
    tokens = loader.tokens
    if LIMIT:
        tokens = tokens[:LIMIT]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"split={SPLIT}  tokens={len(tokens)}  out={OUT}")
    ok = fail = 0
    for i, token in enumerate(tokens):
        try:
            data = adapt(loader.get_agent_input_from_token(token))
            np.savez_compressed(OUT / f"{token}.npz", **data)
            ok += 1
            if i < 2 or i % 50 == 0:
                print(f"  [{i}] {token}: img{data['img'].shape} proj{data['projection_mat'].shape} "
                      f"hist{data['hist_traj'].shape} cmd={int(data['command'])}")
        except Exception as e:  # noqa: BLE001 — keep going, report at end
            fail += 1
            print(f"  [{i}] {token}: FAILED {type(e).__name__}: {e}")
    print(f"=== Stage A done: {ok} ok, {fail} failed -> {OUT} ===")


if __name__ == "__main__":
    main()
