"""
Inspect ONE NavSim AgentInput (warmup_two_stage) to learn the exact tensor
formats the UniDriveVLA adapter must consume. READ-ONLY — loads one token and
prints shapes/dtypes/ranges. Run in the `navsim` env via run_navsim_inspect.sh.

Builds the SceneLoader exactly like run_pdm_score.py (commit 0a380a9, lines 63-76)
via hydra compose, so paths/scene_filter match the real eval.
"""
import os
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader

DEVKIT = os.environ["NAVSIM_DEVKIT_ROOT"]
DATA = os.environ["OPENSCENE_DATA_ROOT"]
SPLIT = os.environ.get("TRAIN_TEST_SPLIT", "warmup_two_stage")
TS = f"{DATA}/{SPLIT}"  # v2.2: split is top-level ($DATA/<split>/{sensor_blobs,synthetic_scene_pickles})
CONFIG_DIR = f"{DEVKIT}/navsim/planning/script/config/pdm_scoring"


def slen(x):
    return len(x) if x is not None else "None"


def fmt(a):
    if a is None:
        return "None"
    a = np.asarray(a)
    extra = f" range=[{a.min():.4g},{a.max():.4g}]" if a.size and a.dtype.kind in "fiu" else ""
    return f"shape={a.shape} dtype={a.dtype}{extra}"


def main():
    overrides = [
        f"train_test_split={SPLIT}",
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
            cam_f0=True, cam_l0=True, cam_l1=True, cam_l2=True,
            cam_r0=True, cam_r1=True, cam_r2=True, cam_b0=True, lidar_pc=False,
        ),
    )

    tokens = loader.tokens
    print(f"num tokens in {SPLIT}: {len(tokens)}")
    print(f"  stage_one: {slen(getattr(loader, 'tokens_stage_one', None))}  "
          f"reactive_2: {slen(getattr(loader, 'reactive_tokens_stage_two', None))}  "
          f"nonreactive_2: {slen(getattr(loader, 'non_reactive_tokens_stage_two', None))}")
    token = tokens[0]
    print(f"\ninspecting token[0] = {token}")

    ai = loader.get_agent_input_from_token(token)
    print(f"len(ego_statuses)={len(ai.ego_statuses)}  len(cameras)={len(ai.cameras)}  len(lidars)={len(ai.lidars)}")

    print("\n--- ego_statuses (history; last = current) ---")
    for i, es in enumerate(ai.ego_statuses):
        print(f" [{i}] pose={np.asarray(es.ego_pose)} vel={np.asarray(es.ego_velocity)} "
              f"acc={np.asarray(es.ego_acceleration)} cmd={np.asarray(es.driving_command)} "
              f"(cmd {fmt(es.driving_command)}) in_global={es.in_global_frame}")

    print("\n--- cameras[-1] (current frame), all 8 ---")
    cams = ai.cameras[-1]
    for name in ["cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"]:
        cam = getattr(cams, name)
        print(f" {name}: image {fmt(cam.image)}")
        print(f"        intrinsics {fmt(cam.intrinsics)} ->\n{np.asarray(cam.intrinsics)}")
        print(f"        sensor2lidar_rotation {fmt(cam.sensor2lidar_rotation)}  "
              f"translation={np.asarray(cam.sensor2lidar_translation)}")
        print(f"        distortion {fmt(cam.distortion)}  camera_path={cam.camera_path}")


if __name__ == "__main__":
    main()
