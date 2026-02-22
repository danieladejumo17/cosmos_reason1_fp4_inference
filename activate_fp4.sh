#!/bin/bash
# =============================================================================
# FP4 Environment Activation Script
# =============================================================================
#
# Source this file to set up the runtime environment for FP4 inference on
# Blackwell GPUs inside vast.ai Docker containers.
#
# Usage:
#   source activate_fp4.sh           # Quick activation (no verification)
#   source activate_fp4.sh --verify  # Activate with package verification
#
# This script configures three things:
#   1. OpenMPI override (LD_PRELOAD + PATH) -- fixes MPI_Init hang in Docker
#   2. NVIDIA library paths (LD_LIBRARY_PATH) -- for TensorRT-LLM + CUDA 13
#   3. Virtual environment activation (if one exists)
#
# Run setup_fp4_vast.sh first to install all dependencies and build OpenMPI.
#
# =============================================================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for --verify flag
VERIFY_ENV=false
for arg in "$@"; do
    if [ "$arg" == "--verify" ]; then
        VERIFY_ENV=true
    fi
done

# =============================================================================
# 1. Fix MPI for Docker/vast.ai containers
# =============================================================================
# OpenMPI 4.1.6 shipped with Ubuntu 24.04 has a PMIx version mismatch
# (ext3x module vs PMIx 5.0.1) that causes MPI_Init to hang when orted
# can't initialize a PMIx server inside Docker containers.
#
# The fix (applied by setup_fp4_vast.sh / build_openmpi_docker.sh) is a
# custom OpenMPI build with --with-pmix=internal. We override the system
# OpenMPI by prepending our build to PATH/LD_LIBRARY_PATH and using
# LD_PRELOAD to ensure our libmpi.so is loaded instead of the system one.

CUSTOM_OMPI="/usr/local/openmpi-4.1.6"
if [ -d "$CUSTOM_OMPI/lib" ]; then
    export LD_LIBRARY_PATH="$CUSTOM_OMPI/lib:${LD_LIBRARY_PATH}"
    export LD_PRELOAD="$CUSTOM_OMPI/lib/libmpi.so"
    export PATH="$CUSTOM_OMPI/bin:$PATH"
    # Disable vader (shared memory) single-copy mechanism which can cause
    # issues in containerized environments without /dev/shm or with
    # restricted permissions.
    export OMPI_MCA_btl_vader_single_copy_mechanism=none
    echo -e "${GREEN}[FP4]${NC} OpenMPI with internal PMIx loaded (Docker/vast.ai fix)"
else
    echo -e "${YELLOW}[FP4]${NC} Custom OpenMPI not found at $CUSTOM_OMPI"
    echo "       MPI may hang. Run: bash setup_fp4_vast.sh"
fi

# =============================================================================
# 1b. Force decord for video reading (avoid torchcodec/FFmpeg errors)
# =============================================================================
# export FORCE_QWENVL_VIDEO_READER=decord
export FORCE_QWENVL_VIDEO_READER=torchvision

# =============================================================================
# 2. NVIDIA Library Paths (CUDA 13, TensorRT, TensorRT-LLM)
# =============================================================================
# TensorRT-LLM's C++ bindings need these libraries on LD_LIBRARY_PATH:
#   - CUDA 13 runtime (cublas, cudnn, nccl, etc.)
#   - TensorRT runtime (libnvinfer.so.10)
#   - TensorRT-LLM native libs

PYTHON_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
if [ -z "$PYTHON_SITE_PACKAGES" ]; then
    PYTHON_SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"
fi

NVIDIA_LIB_DIRS=(
    "$PYTHON_SITE_PACKAGES/nvidia/cu13/lib"
    "$PYTHON_SITE_PACKAGES/nvidia/cublas/lib"
    "$PYTHON_SITE_PACKAGES/nvidia/cudnn/lib"
    "$PYTHON_SITE_PACKAGES/nvidia/nccl/lib"
    "$PYTHON_SITE_PACKAGES/nvidia/cuda_runtime/lib"
    "$PYTHON_SITE_PACKAGES/nvidia/nvjitlink/lib"
    "$PYTHON_SITE_PACKAGES/nvidia/cufft/lib"
    "$PYTHON_SITE_PACKAGES/nvidia/cusparse/lib"
    "$PYTHON_SITE_PACKAGES/nvidia/cusolver/lib"
    "$PYTHON_SITE_PACKAGES/nvidia/cusparselt/lib"
    "$PYTHON_SITE_PACKAGES/nvidia/cuda_cupti/lib"
    "$PYTHON_SITE_PACKAGES/nvidia/cuda_nvrtc/lib"
    "$PYTHON_SITE_PACKAGES/tensorrt_cu13_libs/lib"
    "$PYTHON_SITE_PACKAGES/tensorrt_libs"
    "$PYTHON_SITE_PACKAGES/tensorrt_llm/libs"
)

ADDED_PATHS=0
for lib_dir in "${NVIDIA_LIB_DIRS[@]}"; do
    if [ -d "$lib_dir" ]; then
        export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH}"
        ADDED_PATHS=$((ADDED_PATHS + 1))
    fi
done

if [ "$ADDED_PATHS" -gt 0 ]; then
    echo -e "${GREEN}[FP4]${NC} Added $ADDED_PATHS NVIDIA library paths to LD_LIBRARY_PATH"
else
    echo -e "${YELLOW}[FP4]${NC} Warning: No NVIDIA library paths found"
    echo "       Run: bash setup_fp4_vast.sh"
fi

# =============================================================================
# 3. Virtual Environment (if present)
# =============================================================================

VENV_PATHS=(
    "/venv/main/bin/activate"
    "$SCRIPT_DIR/fp4_env/bin/activate"
    "$SCRIPT_DIR/venv/bin/activate"
    "$SCRIPT_DIR/.venv/bin/activate"
)

for venv_path in "${VENV_PATHS[@]}"; do
    if [ -f "$venv_path" ]; then
        source "$venv_path"
        echo -e "${GREEN}[FP4]${NC} Activated virtual environment: $(dirname $(dirname $venv_path))"
        break
    fi
done

# =============================================================================
# 4. Verification (optional: --verify flag)
# =============================================================================

if $VERIFY_ENV; then
    echo ""
    echo "FP4 Environment Verification:"

    # TensorRT-LLM (suppress its own version banner during import)
    TRTLLM_VERSION=$(python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)" 2>&1 | grep -v "TensorRT LLM version" | tail -1)
    if [ -n "$TRTLLM_VERSION" ]; then
        echo "  [OK] TensorRT-LLM: $TRTLLM_VERSION"
    else
        echo "  [!!] TensorRT-LLM: Not available (check LD_LIBRARY_PATH)"
    fi

    # Flash Attention
    FA_VERSION=$(python3 -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null)
    if [ -n "$FA_VERSION" ]; then
        echo "  [OK] flash-attn: $FA_VERSION"
    else
        echo "  [!!] flash-attn: Not installed (run: pip install flash-attn --no-build-isolation)"
    fi

    # ModelOpt
    MODELOPT_VERSION=$(python3 -c "import modelopt; print(modelopt.__version__)" 2>/dev/null)
    if [ -n "$MODELOPT_VERSION" ]; then
        echo "  [OK] ModelOpt: $MODELOPT_VERSION"
    else
        echo "  [!!] ModelOpt: Not available"
    fi

    # TRT-LLM Patches
    python3 -c "
from tensorrt_llm.functional import RotaryScalingType, PositionEmbeddingType
try:
    RotaryScalingType.from_string('default')
    PositionEmbeddingType.from_string('default')
    print('  [OK] TRT-LLM patches: Applied (rope_type default handled)')
except ValueError:
    print('  [!!] TRT-LLM patches: NOT applied (run setup_fp4_vast.sh)')
" 2>&1 | grep -v "TensorRT LLM version" || echo "  [!!] TRT-LLM patches: Could not verify"

    # PyTorch + CUDA + GPU
    python3 -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    print(f'  [OK] PyTorch: {torch.__version__}')
    print(f'  [OK] CUDA GPU: {gpu_name}')
    print(f'  [OK] Compute Capability: {compute_cap[0]}.{compute_cap[1]}')
    if compute_cap[0] >= 10:
        print('  [OK] Blackwell FP4 Tensor Cores: Available')
    else:
        print('  [!!] Blackwell FP4 Tensor Cores: Not available (SM 10.0+ required)')
else:
    print('  [!!] CUDA: Not available')
" 2>/dev/null || echo "  [!!] PyTorch/CUDA: Check failed"

    # HuggingFace login
    python3 -c "
from huggingface_hub import HfApi
try:
    user = HfApi().whoami()
    print(f'  [OK] HuggingFace: logged in as {user[\"name\"]}')
except:
    print('  [!!] HuggingFace: Not logged in (run: huggingface-cli login)')
" 2>/dev/null || echo "  [!!] HuggingFace: Check failed"
fi

echo ""
echo -e "${GREEN}[FP4]${NC} Environment activated. Ready for FP4 inference."
echo "       Run 'source activate_fp4.sh --verify' to check all packages."
