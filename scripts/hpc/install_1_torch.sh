#!/bin/bash
#
# Blackwell bootstrap — STEP 1 of N: package manager + CUDA 12.8 + PyTorch 2.7,
# then PROVE the GPU actually runs kernels on sm_120 before we build anything else.
# ----------------------------------------------------------------------------
# Target host: ubix (Ubuntu 24.04, RTX PRO 6000 Blackwell sm_120, no root, no conda).
# What it does (all userspace, no sudo):
#   1. installs micromamba to ~/.local/bin if absent
#   2. creates env 'unidrivevla' (py3.9 to match iris) with CUDA 12.8 toolkit,
#      gcc/g++ 13, ninja, cmake
#   3. installs PyTorch 2.7.x cu128 (the first torch with Blackwell kernels)
#   4. runs a GPU smoke: capability check + bf16 matmul + SDPA on CUDA
# What it deliberately does NOT do: flash-attn, mmcv/mmdet3d, data, checkpoints.
#   Those come in step 2, only after this proves torch sees the GPU.
#
# Usage:  bash install_1_torch.sh
# Re-runnable: skips micromamba install and env creation if already present.

# NOTE: no `set -u`. conda/CUDA activation scripts (e.g. ~cuda-nvcc_activate.sh)
# reference vars like NVCC_PREPEND_FLAGS that may be unset, and micromamba
# re-runs activation after `install` — nounset turns that into a fatal error.
set -eo pipefail

# Defensive defaults so the cuda-nvcc activation hook is happy even under -u.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
ENV_NAME=unidrivevla
PY_VER=3.9
CUDA_LABEL="nvidia/label/cuda-12.8.0"
LOG="./install_1_torch_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

echo "=== Blackwell bootstrap step 1 — $(date -Is) on $(hostname -s) ==="
echo "mamba root: $MAMBA_ROOT   env: $ENV_NAME   python: $PY_VER   log: $LOG"

# --- 1. micromamba (rootless single binary) ---------------------------------
MM="$HOME/.local/bin/micromamba"
if [ ! -x "$MM" ]; then
    echo "--- installing micromamba to $MM ---"
    mkdir -p "$HOME/.local/bin"
    # Official static build; bin/micromamba is the only file we need.
    curl -Ls "https://micro.mamba.pm/api/micromamba/linux-64/latest" \
        | tar -xvj -C "$HOME/.local" bin/micromamba
else
    echo "--- micromamba already present: $MM ---"
fi
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
eval "$("$MM" shell hook --shell bash)"

# --- 2. env + CUDA toolkit + compilers + build tools ------------------------
if ! micromamba env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "--- creating env '$ENV_NAME' (python $PY_VER) ---"
    micromamba create -y -n "$ENV_NAME" "python=$PY_VER" -c conda-forge
else
    echo "--- env '$ENV_NAME' already exists ---"
fi
micromamba activate "$ENV_NAME"

echo "--- installing CUDA 12.8 toolkit + gcc/g++ 13 + ninja + cmake ---"
# CUDA 12.8 nvcc can target sm_120; gcc 13 is within CUDA 12.8's supported range.
micromamba install -y -n "$ENV_NAME" \
    -c "$CUDA_LABEL" cuda-toolkit \
    -c conda-forge gcc_linux-64=13 gxx_linux-64=13 ninja cmake

export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc: $("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | tail -2 | head -1)"

# --- 3. PyTorch 2.7 cu128 (first torch with Blackwell sm_120 kernels) -------
echo "--- installing torch 2.7 cu128 (HEAD 403 on the root URL is harmless) ---"
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# --- 4. GPU verification: the make-or-break test ----------------------------
echo "=== GPU VERIFICATION (sm_120 kernel execution) ==="
python - <<'PY'
import torch, sys
print("torch:", torch.__version__, "| built for CUDA:", torch.version.cuda)
ok = torch.cuda.is_available()
print("cuda.is_available:", ok)
if not ok:
    print("!! FAIL: torch cannot see the GPU"); sys.exit(1)
name = torch.cuda.get_device_name(0)
cap  = torch.cuda.get_device_capability(0)
print("device:", name, "| capability:", cap)
if cap[0] < 12:
    print(f"!! WARNING: expected sm_120, got sm_{cap[0]}{cap[1]}")
# Real kernel on the device — this is what fails with "no kernel image
# available" if torch lacks sm_120 support.
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
y = x @ x; torch.cuda.synchronize()
print("bf16 matmul on GPU: OK", tuple(y.shape))
import torch.nn.functional as F
q = torch.randn(1, 16, 1024, 128, device="cuda", dtype=torch.bfloat16)
o = F.scaled_dot_product_attention(q, q, q); torch.cuda.synchronize()
print("SDPA on GPU: OK", tuple(o.shape))
print("peak alloc (MiB):", round(torch.cuda.max_memory_allocated()/1024**2, 1))
print(">>> FOUNDATION OK: Blackwell runs torch kernels. Proceed to step 2.")
PY

echo "=== STEP 1 DONE — log saved to $LOG ==="
echo "To make this env available in new shells, add to ~/.bashrc:"
echo "    export MAMBA_ROOT_PREFIX=$MAMBA_ROOT"
echo "    eval \"\$($MM shell hook --shell bash)\""
echo "    micromamba activate $ENV_NAME"
