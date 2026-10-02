#!/bin/bash
#
# Blackwell bootstrap — STEP 2 of N: the full ML + compiled stack on torch 2.7.
# ----------------------------------------------------------------------------
# Run AFTER step 1 (env + CUDA 12.8 + torch 2.7 verified) and AFTER cloning the
# repo. Run FROM the repo root:   bash scripts/hpc/install_2_packages.sh
#
# Ordered risk-first so the make-or-break builds fail fast:
#   A. numpy/setuptools pins      (foundation for old mmlab stack)
#   B. transformers 4.57.1 + Qwen3-VL 3-tuple patch
#   C. mmcv 1.7.2 build           <-- HIGHEST RISK (old mmlab vs torch 2.7)
#   D. mmdet 2.28.2 + mmseg 0.30.0
#   E. mmdet3d 1.0.0rc6 build
#   F. flash-attn source build for sm_120   <-- LONGEST (~30-60 min)
#   G. deepspeed/peft/timm/qwen-vl-utils + curated nuScenes reqs
#   H. nuScenes plugin custom ops build
#   I. import verification
#
# Idempotent: each phase checks if its artifact already imports and skips if so,
# so re-running after a fix resumes where it failed. set -eo pipefail (NO -u —
# conda/cuda activation hooks reference unset vars).

set -eo pipefail
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
ENV_NAME=unidrivevla
MM="$HOME/.local/bin/micromamba"
LOG="./install_2_packages_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

# Run from repo root — sanity-check we can see the repo layout.
if [ ! -d nuScenes ] || [ ! -d third_party/mmcv-1.7.2 ]; then
    echo "!! Run this from the repo root (need ./nuScenes and ./third_party). Aborting."
    exit 1
fi
REPO_ROOT="$(pwd)"

echo "=== Blackwell bootstrap step 2 — $(date -Is) on $(hostname -s) ==="
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
eval "$("$MM" shell hook --shell bash)"
micromamba activate "$ENV_NAME"

# Build env for nvcc-backed extensions (mmcv, mmdet3d, flash-attn, plugin ops).
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export PATH="$CUDA_HOME/bin:$PATH"
# Target the arch of the GPU that is actually present: ubix is Blackwell (sm_120),
# a GCP A100 is sm_80. Building for the wrong arch still compiles, then dies at
# runtime with "no kernel image is available", so detect it instead of hardcoding.
GPU_CAP="$(python -c 'import torch;c=torch.cuda.get_device_capability(0);print(f"{c[0]}.{c[1]}")')"
export TORCH_CUDA_ARCH_LIST="$GPU_CAP"
FA_ARCH="${GPU_CAP/./}"            # 8.0 -> 80, 12.0 -> 120 (flash-attn's format)
export MAX_JOBS="${MAX_JOBS:-8}"   # flash-attn/mmcv are RAM-hungry; 8*~8GB fits both boxes
echo "CUDA_HOME=$CUDA_HOME | nvcc=$(nvcc --version | tail -2 | head -1) | arch=$TORCH_CUDA_ARCH_LIST | MAX_JOBS=$MAX_JOBS"

py() { python -c "$1"; }
have() { python -c "import $1" 2>/dev/null; }

# Global pip constraints: numpy and opencv are pinned for the WHOLE script so
# that no transitive dependency (e.g. mmcv's unpinned opencv-python pulling
# opencv 4.13 -> numpy>=2.0) can clobber them. Old mmlab needs np.int (numpy
# <1.24); opencv 4.8 is the last numpy-1.x-compatible line.
CONSTRAINTS=/tmp/unidrivevla_constraints.txt
cat > "$CONSTRAINTS" <<EOF
numpy==1.22.4
opencv-python==4.8.0.76
networkx==3.2.1
EOF
export PIP_CONSTRAINT="$CONSTRAINTS"
echo "PIP_CONSTRAINT=$PIP_CONSTRAINT (numpy 1.22.4 + opencv 4.8 + networkx 3.2.1 pinned)"

# --- A. base pins -----------------------------------------------------------
echo "=== A. setuptools<80 + numpy 1.22.4 + opencv 4.8 (np.int + pkg_resources) ==="
pip install "setuptools<80" "numpy==1.22.4" "opencv-python==4.8.0.76"

# --- B. transformers + Qwen3-VL 3-tuple patch -------------------------------
echo "=== B. transformers 4.57.1 + Qwen3-VL patch ==="
pip install transformers==4.57.1
TRANSFORMERS_DIR="$(python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')"
echo "transformers dir: $TRANSFORMERS_DIR"
cp -v qwenvl3/transformers_replace/models/qwen3_vl/modeling_qwen3_vl.py \
      "$TRANSFORMERS_DIR/models/qwen3_vl/modeling_qwen3_vl.py"
py "from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel; print('transformers patch import OK')"

# --- C. core training deps EARLY --------------------------------------------
# This vendored mmcv imports deepspeed in its runner (base_runner.py:14), so
# `import mmdet3d` / `from mmcv import ops` need deepspeed present. Install the
# python deps before the mmlab builds. Skip networkx==2.5 / pytorch-lightning /
# torchmetrics from requirements_nusc.txt — those downgrade torch 2.7's deps
# and can break dynamo/FX.
echo "=== C. deepspeed/peft/timm/qwen-vl-utils + curated nuScenes reqs ==="
# Force networkx back to 3.2.1 (a stale 2.2 lingers in the env; torch 2.7 fx
# wants 3.x; 3.2.1 is the last 3.x that supports py3.9). Constraint reinforces.
pip install "networkx==3.2.1"
pip install deepspeed==0.14.4 peft timm==1.0.11 qwen-vl-utils gdown "huggingface_hub[cli]"
pip install einops==0.8.1 casadi==3.6.7 motmetrics yapf==0.40.1 \
    pandas==1.2.2 ipython nuscenes-devkit==1.1.11

# --- D. mmcv 1.7.2 build (CUDA ops) -----------------------------------------
echo "=== D. mmcv 1.7.2 build (CUDA ops) ==="
# Verify the compiled extension via mmcv._ext (NOT `from mmcv import ops`, which
# pulls cnn->runner->import deepspeed and would conflate a build failure with a
# missing dep).
if py "import mmcv, mmcv._ext; print('mmcv', mmcv.__version__, '_ext OK')" 2>/dev/null; then
    echo "mmcv._ext already built — skipping"
else
    pushd third_party/mmcv-1.7.2 >/dev/null
    export MMCV_WITH_OPS=1 FORCE_CUDA=1 MMCV_NO_Compiler_CHECK=1
    pip install -r requirements.txt
    python setup.py build_ext --inplace
    # --no-build-isolation: use the env's setuptools<80 (still ships
    # pkg_resources); the isolated build env grabs setuptools 82+ where
    # pkg_resources is gone, breaking mmcv's setup.py import.
    pip install -e . --no-build-isolation
    popd >/dev/null
    py "import mmcv, mmcv._ext; print('mmcv', mmcv.__version__, '_ext OK')"
fi

# --- E. mmdet + mmseg (pure-python wheels) ----------------------------------
echo "=== E. mmdet 2.28.2 + mmsegmentation 0.30.0 ==="
have mmdet || pip install mmdet==2.28.2
have mmseg || pip install mmsegmentation==0.30.0

# --- F. mmdet3d 1.0.0rc6 build ----------------------------------------------
echo "=== F. mmdet3d 1.0.0rc6 build ==="
# mmdet3d 1.0.0rc6 runtime.txt carries toxic pins (networkx<2.3, numba==0.53.0,
# trimesh<2.35.40) that deadlock pip's resolver against the modern env's
# networkx 3.2.1 / torch 2.7, so we install the real runtime deps it imports —
# unpinned and env-compatible — and install mmdet3d itself with --no-deps so pip
# never resolves the stale pins. lyft_dataset_sdk is NOT optional despite the
# "lyft" name: mmdet3d.datasets -> core.evaluation.lyft_eval imports it at package
# load, so converters and training break without it (--no-deps: its runtime deps
# are pure-python and already present from section C).
# These dep installs run UNCONDITIONALLY (idempotent; pip skips what's present) —
# they must NOT sit behind the "mmdet3d already built" guard, or a re-run on a box
# that already has mmdet3d silently skips them and mmdet3d.datasets fails to import.
# scikit-image and tensorboard are pinned to versions that respect the env's
# numpy 1.22.4 (skimage>=0.20 wants numpy>=1.23) and protobuf 3.19.6 (tensorboard
# >=2.13 pulls protobuf 4/5, which breaks onnx 1.12). tensorboard is imported by
# the training logger (torch.utils.tensorboard).
pip install numba plyfile trimesh "scikit-image==0.19.3" "tensorboard==2.11.2" future
pip install --no-deps lyft_dataset_sdk
if have "mmdet3d.datasets"; then
    echo "mmdet3d (incl. datasets) already importable — skipping build"
else
    pushd third_party/mmdetection3d-1.0.0rc6 >/dev/null
    pip install -e . --no-build-isolation --no-deps
    popd >/dev/null
fi

# --- G. flash-attn (LONGEST when it has to build from source) ----------------
echo "=== G. flash-attn 2.8.3 for sm_${FA_ARCH} ==="
if have flash_attn; then
    echo "flash_attn already importable — skipping"
elif [ "$FA_ARCH" = "120" ]; then
    # Blackwell: no released wheel carries sm_120, and a wheel that lacks it
    # imports fine but fails at kernel launch, so force a source build.
    # flash-attn ignores TORCH_CUDA_ARCH_LIST and defaults to building
    # sm_80/90/100/120 — 4x the memory, which OOM-killed nvcc (the 'Killed'
    # in the prior run) at MAX_JOBS=8. FLASH_ATTN_CUDA_ARCHS restricts to
    # sm_120 only (4x less work), and we throttle parallelism so the heavy
    # backward kernels stay within the 94 GB box.
    FLASH_ATTN_CUDA_ARCHS="120" MAX_JOBS=4 NVCC_THREADS=2 \
        FLASH_ATTENTION_FORCE_BUILD=TRUE \
        pip install flash-attn==2.8.3 --no-build-isolation
else
    # Mainstream archs (sm_80 A100, sm_90 H100) are covered by released wheels:
    # minutes instead of an hour. If pip finds no matching wheel it falls back
    # to a source build, which FLASH_ATTN_CUDA_ARCHS keeps to this arch alone.
    FLASH_ATTN_CUDA_ARCHS="$FA_ARCH" MAX_JOBS=4 NVCC_THREADS=2 \
        pip install flash-attn==2.8.3 --no-build-isolation
fi

# --- H. nuScenes plugin custom ops ------------------------------------------
echo "=== H. nuScenes plugin custom ops build ==="
pushd nuScenes/projects/mmdet3d_plugin/ops >/dev/null
pip install -e . --no-build-isolation
popd >/dev/null

# --- I. import verification --------------------------------------------------
echo "=== I. VERIFICATION ==="
python - <<'PY'
import importlib, sys
mods = ["torch", "torchvision", "numpy", "transformers", "mmcv", "mmdet",
        "mmseg", "mmdet3d", "mmdet3d.datasets", "flash_attn", "deepspeed",
        "peft", "timm", "nuscenes", "skimage",
        "torch.utils.tensorboard"]  # datasets->lyft_eval; tensorboard = training logger
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  [ok] {m:14s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  [FAIL] {m:14s} {type(e).__name__}: {e}")
        bad.append(m)
import torch
print("torch CUDA:", torch.version.cuda, "| device cap:", torch.cuda.get_device_capability(0))
# flash-attn kernel on sm_120 — the real test
try:
    import flash_attn, torch
    from flash_attn import flash_attn_func
    q = torch.randn(1, 256, 8, 64, device="cuda", dtype=torch.bfloat16)
    o = flash_attn_func(q, q, q); torch.cuda.synchronize()
    print("flash_attn kernel on sm_120: OK", tuple(o.shape))
except Exception as e:
    print(f"  [flash_attn kernel] {type(e).__name__}: {e}")
    bad.append("flash_attn-kernel")
if bad:
    print("!! FAILED:", bad); sys.exit(1)
print(">>> STEP 2 OK: full stack imports. Proceed to data + smoke (step 3).")
PY

echo "=== STEP 2 DONE — log: $LOG ==="
