# VLA for E2E Autonomous Driving — multi-candidate UniDriveVLA on NAVSIM

Code for the master's thesis *VLA for E2E Autonomous Driving* (Abdollah Gader, University of
Luxembourg, Master in Information and Computer Sciences, 2026). The thesis PDF is in
[`thesis/thesis.pdf`](thesis/thesis.pdf).

This repository is a fork of [UniDriveVLA](https://github.com/xiaomi-research/unidrivevla)
(Li et al., 2026). It adds:

1. **Eight candidate trajectories per scene.** Eight learned mode embeddings in UniDriveVLA's
   action expert, trained with a relaxed winner-takes-all flow-matching objective. The rest of
   the model stays frozen.
2. **A NAVSIM v2 evaluation pipeline.** UniDriveVLA and the NAVSIM devkit need incompatible Python
   environments, so the pipeline is split into stages that pass files to each other.
3. **A learned scorer.** It predicts the EPDMS terms of each candidate and selects the candidate
   with the highest predicted score.

## Results on navhard (NAVSIM v2, 5912 scenes)

| Candidate selection | EPDMS ↑ |
|---|---|
| Mode 0 (default) | 10.0 |
| Best fixed mode (mode 6) | 12.0 |
| Learned scorer (5-fold CV on navhard) | **12.7** |

The scorer is trained and evaluated on navhard with five-fold cross-validation. Synthetic
variants of one real scene can fall into different folds, so this number may be optimistic
(thesis, Section "Learned scorer" of the Experiments chapter).

## Contents

- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Step 1 — Model environment (`unidrivevla`)](#step-1--model-environment-unidrivevla)
- [Step 2 — Checkpoints and nuScenes data](#step-2--checkpoints-and-nuscenes-data)
- [Step 3 — NAVSIM data and environment (`navsim`)](#step-3--navsim-data-and-environment-navsim)
- [Step 4 — Get the eight-mode model](#step-4--get-the-eight-mode-model)
- [Step 5 — Score the candidates on navhard](#step-5--score-the-candidates-on-navhard)
- [Step 6 — Learned scorer](#step-6--learned-scorer)
- [Tests](#tests)
- [Troubleshooting: what to change where](#troubleshooting-what-to-change-where)

## Repository layout

| Path | Content |
|---|---|
| `nuScenes/` | UniDriveVLA model code (upstream) and the multi-mode changes. Configs in `nuScenes/projects/configs/UniDriveVLA/`. |
| `nuScenes/projects/configs/UniDriveVLA/unidrivevla_stage2_2b_modes8_ft*.py` | Eight-mode fine-tune configs (`_8gpu` is the thesis run). |
| `nuScenes/projects/mmdet3d_plugin/unidrivevla/dense_heads/qwenvl3_vla_planning_head.py` | Planning head: mode embeddings, relaxed winner-takes-all, flow matching. |
| `scripts/hpc/` | Environment installation, data download, fine-tuning, checkpoint delta tools. |
| `scripts/navsim/` | NAVSIM pipeline: Stage A (inputs), Stage B (model), Stage C (submission), oracle. |
| `scripts/navsim/navsim_paths.sh` | **The one file to edit** for NAVSIM data and workspace paths. |
| `scripts/scorer/` | Learned scorer: dataset, model, cross-validation, EPDMS reconstruction, tests. |
| `thesis/` | LaTeX source and PDF of the thesis. |
| `third_party/` | Vendored mmcv 1.7.2, mmdetection3d 1.0.0rc6, transformers 4.57.1 (upstream). |
| `Bench2Drive/`, `vqa_evaluation/` | Upstream UniDriveVLA code, not used by the thesis. |

All commands below run from the repository root unless stated otherwise.

## Requirements

- Linux, NVIDIA GPU, no root access needed. Everything installs with
  [micromamba](https://mamba.readthedocs.io) into your home directory.
- **Inference and scoring (Steps 1–3, 5, 6):** one GPU. The scripts were run on an RTX PRO 6000
  Blackwell (96 GB) and on A100 80 GB.
- **Fine-tuning (Step 4, option B):** the thesis run used 8 × A100 80 GB for 7000 iterations.
  A single GPU works with the 1-GPU config but takes much longer.
- **Disk:** about 160 GB for NAVSIM (maps, warm-up, navhard and its camera images), about 15 GB
  for checkpoints, about 400 GB more for nuScenes trainval (only needed to fine-tune).
- **RAM:** 64 GB or more. Loading the nuScenes trainval annotations alone takes 20–40 GB.

Run long jobs inside `tmux` so that a dropped SSH connection does not kill them.

## Step 1 — Model environment (`unidrivevla`)

```bash
git clone https://github.com/abdo75/unidrivevla-8-mode.git
cd unidrivevla-8-mode
bash scripts/hpc/install_1_torch.sh      # micromamba, env 'unidrivevla' (Python 3.9), CUDA 12.8, gcc 13, PyTorch 2.7.0
bash scripts/hpc/install_2_packages.sh   # transformers 4.57.1 + Qwen3-VL patch, mmcv, mmdet, mmdet3d, flash-attn, deepspeed, plugin ops
```

Both scripts are re-runnable: finished parts are skipped. `install_1_torch.sh` ends with a GPU
test (bf16 matmul and attention). `install_2_packages.sh` ends with an import test of every
package and a flash-attn kernel run. If either test fails, fix the cause before continuing.

What the scripts assume, and what to change if your machine differs:

| Assumption | Where | Change |
|---|---|---|
| micromamba at `~/.local/bin/micromamba`, envs under `~/micromamba` | top of every script | Export `MAMBA_ROOT_PREFIX=/your/path` before running. If micromamba is already on your `PATH`, it is used. |
| GPU needs CUDA 12.8 + PyTorch 2.7 (required for Blackwell, sm_120; also works on A100/H100) | `install_1_torch.sh`: `CUDA_LABEL`, `torch==2.7.0 ... cu128` | For an older driver that does not support CUDA 12.8, use `CUDA_LABEL="nvidia/label/cuda-12.1.0"`, `gcc_linux-64=12 gxx_linux-64=12`, and `torch==2.5.1 torchvision==0.20.1 --index-url .../cu121`. Check the driver with `nvidia-smi`. |
| GPU architecture | `install_2_packages.sh` detects it with `torch.cuda.get_device_capability` | Nothing to change. Build on the machine whose GPU you will use: kernels built for another architecture fail with `no kernel image is available`. |
| Enough RAM for parallel compilation | `install_2_packages.sh`: `MAX_JOBS` (default 8; flash-attn uses 4) | If the build is killed (`Killed`, nvcc OOM), run `MAX_JOBS=2 bash scripts/hpc/install_2_packages.sh`. The flash-attn build from source takes 30–60 min. |
| numpy 1.22, opencv 4.8, networkx 3.2.1, setuptools < 80 | `install_2_packages.sh`: `CONSTRAINTS` file | Do not upgrade them. The mmlab stack uses `np.int` (removed in numpy 1.24) and `pkg_resources` (removed in setuptools 82). |
| Old glibc (< 2.32) | flash-attn wheel import fails with `GLIBC_2.32 not found` | Build from source: `FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn==2.8.3 --no-build-isolation`. |

Check the environment at any time:

```bash
micromamba run -n unidrivevla python -c "import torch, mmdet3d, flash_attn; print(torch.__version__, torch.cuda.get_device_name(0))"
```

## Step 2 — Checkpoints and nuScenes data

```bash
bash scripts/hpc/install_3_nuscenes_mini.sh
```

This downloads:

- the released UniDriveVLA checkpoints from HuggingFace:
  `owl10/UniDriveVLA_Nusc_Base_Stage1` (VLM weights) and `owl10/UniDriveVLA_Nusc_Base_Stage2`
  (full 2B model, `UniDriveVLA_Stage2_Nuscenes_2B.pt`), plus the OccWorld VAE;
- nuScenes v1.0-mini, the Occ3D labels, the map expansion and CAN bus data;

and then generates the info files and the k-means anchors (`nuScenes/data/kmeans/*.npy`). The
model configs read these anchors, so this step is needed even if you only run inference.

Where data lands, and how to change it:

| Variable | Default | Meaning |
|---|---|---|
| `CKPT_ROOT` | first existing of `/mnt/shared/$USER/UniDriveVLA/checkpoints`, `$HOME/UniDriveVLA/checkpoints`; else a new dir under `$HOME` | Checkpoint directory. Every later script looks for `$CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage1` in the same places and also in `<repo>/checkpoints`. |
| `DATA_ROOT` | `$HOME/nuscenes` (or `/mnt/shared/...` if it exists) | nuScenes data. The script symlinks it into `nuScenes/data/nuscenes/`. |

Example: `CKPT_ROOT=$PWD/checkpoints DATA_ROOT=/big/disk/nuscenes bash scripts/hpc/install_3_nuscenes_mini.sh`.
If you set `CKPT_ROOT` here, export the same value before every later script.

Some files are behind a login (map expansion, CAN bus) or a Google Drive quota (Occ3D). When a
download fails, the script prints the file it needs and waits. Either paste a working download
link, or download the file in a browser, copy it to the printed path, and press Enter. The script is
re-runnable; `FORCE_STEP3=1` forces a full re-run.

**Only for fine-tuning (Step 4, option B):** the full nuScenes trainval set is needed.

```bash
bash scripts/hpc/install_4_nuscenes_trainval.sh   # ~400 GB; set TRAINVAL_ROOT=/big/disk if the default has no space
```

## Step 3 — NAVSIM data and environment (`navsim`)

1. **Set the paths.** Edit the two lines in
   [`scripts/navsim/navsim_paths.sh`](scripts/navsim/navsim_paths.sh). All NAVSIM scripts read
   them; there is nothing else to export.

   ```bash
   : "${OPENSCENE_DATA_ROOT:=$HOME/navsim_dataset}"   # NAVSIM data (~160 GB, many small files)
   : "${NAVSIM_WS:=$HOME/navsim_ws}"                  # devkit clone + all outputs (exp/)
   ```

   If your home directory has a quota (common on clusters), point both to a scratch disk.

2. **Download the data** with the devkit's own download scripts: maps, `warmup_two_stage`,
   `navhard_two_stage`, the test-split logs, and the camera images of the navhard scenes.

   ```bash
   bash scripts/hpc/fetch_navsim_data.sh
   ```

   It checks for free space first (`REQUIRED_GB`, default 160) and skips splits that already
   exist. It needs `wget`, `unzip` and `tar`.

3. **Create the `navsim` environment.** It clones the
   [NAVSIM devkit](https://github.com/autonomousvision/navsim) into `$NAVSIM_WS/navsim`, creates
   its own micromamba env from the devkit's `environment.yml`, and writes
   `$NAVSIM_WS/navsim_env.sh` (sourced by the later scripts).

   ```bash
   bash scripts/hpc/setup_navsim_env.sh
   ```

4. **Check the setup** with NAVSIM's constant-velocity agent. It needs no model and no GPU.

   ```bash
   TRAIN_TEST_SPLIT=navhard_two_stage bash scripts/hpc/run_navsim_metric_cache.sh   # once per split, CPU
   TRAIN_TEST_SPLIT=navhard_two_stage bash scripts/hpc/run_navsim_warmup_cv.sh      # expect EPDMS ≈ 0.115
   ```

   The metric cache holds the ground truth that EPDMS scoring reads. Build it once per split.
   The `warmup_two_stage` split (default when `TRAIN_TEST_SPLIT` is unset) is small and useful
   for quick tests of every step below.

## Step 4 — Get the eight-mode model

### Option A — Use the released weights (recommended)

Only the trained parameters (mode embeddings and the action expert's input/output layers,
11 MB) are released. They are merged into the released UniDriveVLA Stage-2 checkpoint:

```bash
mkdir -p checkpoints
wget -O checkpoints/iter_7000_trained_delta.pt \
  https://github.com/abdo75/unidrivevla-8-mode/releases/download/thesis-v1/iter_7000_trained_delta.pt
micromamba run -n unidrivevla python scripts/hpc/apply_trainable_delta.py \
  $CKPT_ROOT/UniDriveVLA_Nusc_Base_Stage2/UniDriveVLA_Stage2_Nuscenes_2B.pt \
  checkpoints/iter_7000_trained_delta.pt \
  checkpoints/UniDriveVLA_NAVSIM_modes8_iter7000.pt
```

`checkpoints/UniDriveVLA_NAVSIM_modes8_iter7000.pt` is the default `CKPT` of the Stage B scripts.

### Option B — Train it yourself

Needs nuScenes trainval (Step 2). Thesis run, 8 GPUs, 7000 iterations:

```bash
tmux new -s modes
CFG=projects/configs/UniDriveVLA/unidrivevla_stage2_2b_modes8_ft_8gpu.py \
EXP_NAME=modes8_ft_8gpu \
  bash scripts/hpc/finetune_stage2_modes.sh
```

- One GPU: use `CFG=projects/configs/UniDriveVLA/unidrivevla_stage2_2b_modes8_ft.py`
  (5000-scene subset of trainval). The script switches to the `gloo` backend on one GPU.
- The training log is written to
  `nuScenes/work_dirs/<EXP_NAME>/logs/baseline/train-baseline-all-train.txt`, not to the terminal.
  `planning.mode_endpoint_std` should increase as the modes separate.
- Re-running the same command resumes from the latest checkpoint. Delete the work dir to start over.
- The config saves a checkpoint every 1000 iterations (`max_iters=17580`). The thesis uses
  iteration 7000, after which the nuScenes validation minADE no longer decreased. To evaluate a
  checkpoint: `bash scripts/hpc/eval_minade.sh nuScenes/work_dirs/modes8_ft_8gpu/iter_7000/global_step*/mp_rank_00_model_states.pt`.

Use your checkpoint in Step 5 with
`CKPT=$PWD/nuScenes/work_dirs/modes8_ft_8gpu/iter_7000`. To extract the trained parameters for
another machine: `python scripts/hpc/extract_trainable_delta.py <mp_rank_00_model_states.pt> <delta.pt>`.

## Step 5 — Score the candidates on navhard

Each stage writes files under `$NAVSIM_WS/exp/`, which the next stage reads. Set the split
once:

```bash
export TRAIN_TEST_SPLIT=navhard_two_stage
```

| Stage | Command | Env | GPU | Output |
|---|---|---|---|---|
| A. Model inputs (camera images, ego state, command) per scene | `bash scripts/navsim/run_navsim_stage_a.sh` | navsim | no | `exp/uni_inputs/<split>/` |
| Metric cache (skip if done in Step 3) | `bash scripts/hpc/run_navsim_metric_cache.sh` | navsim | no | `exp/metric_cache/<split>/` |
| B. Eight candidates per scene | `bash scripts/navsim/run_navsim_stage_b_modes.sh` | unidrivevla | yes | `exp/uni_trajectories/<split>_modes/trajectories.pkl` |
| C. Score every mode with EPDMS | `bash scripts/navsim/run_navsim_oracle_ceiling.sh` | navsim | no | `exp/uni_oracle_<split>_cand<k>/` (one CSV per mode) |

The scripts activate the right environment themselves. Run each first with `LIMIT=4` (Stage A,
B) on `warmup_two_stage` to check the chain in minutes.

Stage C converts each candidate to NAVSIM's format (axis swap, lateral sign −1, extension from
3 s to 4 s), builds one submission per mode and scores it with the official
`run_pdm_score_from_submission.py`. The `Final extended pdm score` printed for candidate *k* is
the EPDMS of mode *k* (thesis, Table "Scores of the eight candidates on navhard"). `oracle_aggregate.py` then prints the per-scene
statistics: mean stage score per mode, random candidate and oracle (best of 8).

Stage B reads `CONFIG`, `CKPT` and `FT_LOAD_FROM`; the defaults match Option A. Override any of
them on the command line, e.g. `CKPT=/path/to/ckpt bash scripts/navsim/run_navsim_stage_b_modes.sh`.

## Step 6 — Learned scorer

```bash
export TRAIN_TEST_SPLIT=navhard_two_stage
bash scripts/navsim/run_navsim_stage_b_scorer.sh   # Stage B again, also saving the perception tokens of each scene (GPU)
bash scripts/scorer/run_scorer_navhard_cv.sh       # dataset -> 5-fold CV -> submission -> official EPDMS
```

`run_scorer_navhard_cv.sh`:

1. joins the perception tokens, the eight candidates and the EPDMS terms from Step 5 into one
   dataset (`exp/uni_scorer/dataset_navhard_two_stage.pt`);
2. splits the scenes into 5 random groups, trains 5 scorers (each on 4 groups, 200 epochs) and
   selects a candidate for every scene with the scorer that did not see it
   (`exp/uni_scorer/selection_navhard_two_stage_cv.json`);
3. builds a submission from the selected candidates and scores it with the official EPDMS. This
   is the 12.7 in the results table.

The number of folds and epochs can be changed with `K=` and `EPOCHS=`. Candidates are generated
with fixed noise seeds, so the dump in this step matches the candidates scored in Step 5.

Further analysis scripts (all in `scripts/scorer/`, `unidrivevla` env):

| Script | Purpose |
|---|---|
| `aggregate_subscores.py <dataset.pt>` | Mean of each EPDMS term per mode. |
| `diagnose_signal.py <dataset.pt>` | Spread of scores between candidates of one scene. |
| `run_ablations.sh` | Scorer variants (loss, inputs, width). |
| `find_examples.py`, `../navsim/run_navsim_render_examples.sh` | Pick and render example scenes. |

## Tests

CPU only, `unidrivevla` env:

```bash
cd scripts/scorer
micromamba run -n unidrivevla python test_epdms.py              # EPDMS reconstruction vs NAVSIM CSV rows
micromamba run -n unidrivevla python test_scorer_model.py
micromamba run -n unidrivevla python test_build_dataset.py
micromamba run -n unidrivevla python test_cv_select.py
micromamba run -n unidrivevla python test_select_candidates.py
```

## Troubleshooting: what to change where

| Symptom | Cause | Fix |
|---|---|---|
| `micromamba: command not found` in a script | micromamba not at `~/.local/bin` | Put it on `PATH`, or run `install_1_torch.sh`, which installs it there. |
| `no kernel image is available for execution on the device` | CUDA extensions built for another GPU | Re-run `install_2_packages.sh` on the machine with the target GPU (delete `third_party/mmcv-1.7.2/build` first). |
| `FlashAttention only supports Ampere GPUs or newer` | GPU older than Ampere (e.g. V100) | The configs already use `visual_attn_impl="sdpa"`, and the perception decoder falls back to SDPA on such GPUs. The remaining flash-attn calls are in training only: train on Ampere or newer. |
| `CKPT_ROOT not found (UniDriveVLA_Nusc_Base_Stage1)` | Checkpoints are elsewhere | `export CKPT_ROOT=/path/to/checkpoints` before the script. |
| `navsim_env.sh not found` / NAVSIM data paths not found | `navsim_paths.sh` points elsewhere than where the data is | Edit the two lines in `scripts/navsim/navsim_paths.sh`, then re-run `setup_navsim_env.sh`. |
| `metric cache missing` | Cache not built for this split | `TRAIN_TEST_SPLIT=<split> bash scripts/hpc/run_navsim_metric_cache.sh`. |
| Training crashes at NCCL init or segfaults on one GPU | NCCL plugins of the cloud image, or NCCL on a single GPU | `finetune_stage2_modes.sh` already sets `NCCL_NET=Socket` and uses `DIST_BACKEND=gloo` on one GPU. Set `DIST_BACKEND=gloo` explicitly if it still fails. |
| `torchrun: error: argument --master-port: invalid int value` | `tools/dist_train.sh` reads `MLP_WORKER_0_PORT`, not `MASTER_PORT` | `export MLP_WORKER_0_PORT=28600` (the fine-tune script does this). |
| Process killed with `exitcode: -9` and no traceback | System RAM exhausted (annotation loading, data-loader workers) | Use a node with ≥ 64 GB RAM. The fine-tune configs set `workers_per_gpu=0` for this reason. |
| CUDA out of memory | Other jobs on the GPU, or fragmentation | Free the GPU (training needs ~40 GB free). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by the scripts. |
| `TypeError: 'function' object is not iterable` inside `torch.compile` | torch 2.7 inductor bug with mmcv ops | Handled in this fork: the code falls back to the uncompiled module. No change needed. |
| NCCL timeout at the end of evaluation | Ranks finish at very different times | `export TORCH_DIST_TIMEOUT=14400` (seconds). |
| Trajectories turn the wrong way in NAVSIM | Wrong lateral sign in the conversion | Keep `LATERAL_SIGN=-1` (default in `stage_c_build_submission.py`). `+1` mirrors trajectories left↔right. The straight-heavy warm-up split does not show the error; check on navhard. |
| `ModuleNotFoundError: qwenvl3_vla_planning_head_single_decoder` | Upstream dead import | Already fixed in this fork. |

Notes on the metric:

- **EPDMS** is NAVSIM v2's official score. Per real scene, it multiplies the first-stage score
  with a weighted mean of the scores of the second-stage (synthetic) scenes; the weights depend on
  how close each synthetic start is to where the planner's first-stage trajectory ends. The NAVSIM
  CSV reports it as `Final extended pdm score` (0–1; the thesis and the table above report × 100).
- **Stage score** is the score of one trajectory in one scene, before the two stages are
  combined. Its mean over the 5912 scenes, printed by `oracle_aggregate.py`, is used only to
  compare selection policies (thesis, Section "Evaluation metrics"). Its values differ from
  EPDMS.

Licensed under Apache 2.0 ([LICENSE.txt](LICENSE.txt)), as the upstream repository.
