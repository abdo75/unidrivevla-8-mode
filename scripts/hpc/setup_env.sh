#!/bin/bash -l

#SBATCH -J unidrivevla_env
#SBATCH --mail-type=start,end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH -p gpu
#SBATCH -o slurm-env-%j.out
#SBATCH --qos=normal

set -eo pipefail

eval "$(micromamba shell hook --shell bash)"
micromamba activate unidrivevla

# --- Step 3: CUDA toolkit + compilers ---
echo ">>> [1/11] Installing CUDA toolkit and compilers..."
micromamba install -c nvidia/label/cuda-12.1.0 cuda-toolkit gcc_linux-64=12 gxx_linux-64=12 -y

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++

echo "============================================"
echo "  Python:    $(python --version)"
echo "  nvcc:      $(nvcc --version | grep release)"
echo "  GCC:       $($CXX --version | head -1)"
echo "  CUDA_HOME: $CUDA_HOME"
echo "============================================"

# --- Step 4: PyTorch ---
echo ""
echo ">>> [2/11] Installing PyTorch..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# --- Step 5: Transformers + Qwen3-VL patches ---
echo ""
echo ">>> [3/11] Installing transformers and patching Qwen3-VL..."
pip install transformers==4.57.1
TRANSFORMERS_DIR=${CONDA_PREFIX}/lib/python3.9/site-packages/transformers/
cp -r ~/UniDriveVLA/qwenvl3/transformers_replace/models ${TRANSFORMERS_DIR}

# --- Step 6: Pin setuptools ---
echo ""
echo ">>> [4/11] Pinning setuptools..."
pip install "setuptools<80"

# --- Step 7: Build MMCV ---
echo ""
echo ">>> [5/11] Building mmcv 1.7.2 from source..."
cd ~/UniDriveVLA/third_party/mmcv-1.7.2
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MMCV_NO_Compiler_CHECK=1
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install -e .

# --- Step 7b: Install mmdet (missing from requirements_nusc.txt) ---
echo ""
echo ">>> [6/11] Installing mmdet..."
pip install mmdet==2.28.2 mmsegmentation==0.30.0

# --- Step 8: Build mmdet3d ---
echo ""
echo ">>> [7/11] Building mmdet3d from source..."
cd ~/UniDriveVLA/third_party/mmdetection3d-1.0.0rc6
pip install -e .

# --- Step 9: Training deps ---
echo ""
echo ">>> [8/11] Installing training dependencies..."
pip install deepspeed==0.14.4 peft

echo ""
echo ">>> [9/11] Installing flash-attn (forced source build)..."
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn==2.8.3

echo ""
echo ">>> [10/11] Installing timm and qwen-vl-utils..."
pip install timm==1.0.11 qwen-vl-utils

# --- nuScenes steps ---
echo ""
echo ">>> [11/11] Installing nuScenes requirements and building custom ops..."
cd ~/UniDriveVLA
pip install -r requirements/requirements_nusc.txt
cd ~/UniDriveVLA/nuScenes/projects/mmdet3d_plugin/ops
pip install -e .

# --- Verification ---
cd ~/UniDriveVLA
echo ""
echo "============================================"
echo "Verification:"
python -c "
import torch
checks = [
    ('torch',        lambda: torch.__version__),
    ('CUDA avail',   lambda: str(torch.cuda.is_available())),
    ('GPU',          lambda: torch.cuda.get_device_name(0)),
    ('numpy',        lambda: __import__('numpy').__version__),
    ('mmcv',         lambda: __import__('mmcv').__version__),
    ('mmdet',        lambda: __import__('mmdet').__version__),
    ('mmdet3d',      lambda: __import__('mmdet3d').__version__),
    ('transformers', lambda: __import__('transformers').__version__),
    ('deepspeed',    lambda: __import__('deepspeed').__version__),
    ('peft',         lambda: __import__('peft').__version__),
    ('flash_attn',   lambda: __import__('flash_attn').__version__),
    ('timm',         lambda: __import__('timm').__version__),
    ('qwen_vl_utils',lambda: __import__('qwen_vl_utils').__version__),
]
for name, fn in checks:
    try:
        print(f'  {name:15s} {fn()}')
    except Exception as e:
        print(f'  {name:15s} FAILED: {e}')
"
echo "============================================"
echo "Environment build complete."
