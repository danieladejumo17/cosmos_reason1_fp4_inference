#!/bin/bash
# =============================================================================
# FP4 Inference Setup for vast.ai Docker Instances (Blackwell GPUs)
# =============================================================================
#
# Single-script setup for TensorRT-LLM FP4 inference on vast.ai containers.
# Handles all dependencies, fixes, and patches needed to run NVFP4 VLM
# inference on RTX 5090 (Blackwell) inside Docker.
#
# Supports:
#   - nvidia/Cosmos-Reason1-7B (quantized to NVFP4 via quantize_cosmos_fp4.py)
#   - nvidia/Qwen2.5-VL-7B-Instruct-NVFP4 (pre-quantized by NVIDIA)
#
# What this script does:
#   1. Installs system build dependencies (gcc, wget, etc.)
#   2. Builds OpenMPI 4.1.6 from source with internal PMIx
#      (fixes MPI_Init hang in Docker/vast.ai containers)
#      NOTE: Must happen BEFORE TensorRT-LLM install because TRT-LLM's
#      C++ bindings link against libmpi.so.40 at import time.
#   3. Installs TensorRT-LLM, PyTorch, and Python dependencies (if not present)
#   4. Installs flash-attn (required by Qwen2.5-VL vision encoder in TRT-LLM)
#   5. Patches TensorRT-LLM to handle HuggingFace rope_type "default"
#      (fixes Qwen2.5-VL/Cosmos-Reason1 model loading in TRT-LLM v1.1.0)
#   6. Logs in to HuggingFace (for gated NVIDIA model access)
#   7. Verifies the full stack works end-to-end
#
# Prerequisites:
#   - vast.ai instance with RTX 5090 (Blackwell SM 10.0+)
#   - NVIDIA drivers + CUDA toolkit installed (nvidia-smi must work)
#   - Python 3.10+ with pip
#   - TensorRT-LLM and PyTorch will be installed automatically if not present
#
# Usage:
#   bash setup_fp4_vast.sh                          # Full setup (interactive HF login)
#   bash setup_fp4_vast.sh --hf-token <TOKEN>       # Provide HF token non-interactively
#   bash setup_fp4_vast.sh --skip-hf-login          # Skip HuggingFace login
#   bash setup_fp4_vast.sh --skip-build-openmpi     # Skip OpenMPI build (if already done)
#   bash setup_fp4_vast.sh --skip-verify            # Skip final verification
#
# After setup, activate the environment before running inference:
#   source activate_fp4.sh
#   python3 quantize_cosmos_fp4.py                                            # Quantize (once)
#   python3 cosmos_fp4_inference.py --video_dir stu_dataset/stu_videos        # Cosmos FP4
#   python3 trtllm_fp4_inference.py --video_dir stu_dataset/stu_videos        # Qwen FP4
#   python3 fp8_inference.py --video_dir stu_dataset/stu_videos               # INT8 comparison
#
# =============================================================================

set -e

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ────────────────────────────────────────────────────────────────
HF_TOKEN=""
SKIP_HF_LOGIN=false
SKIP_BUILD_OPENMPI=false
SKIP_VERIFY=false
OMPI_VERSION="4.1.6"
OMPI_INSTALL="/usr/local/openmpi-${OMPI_VERSION}"

# ── Parse Arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-token)       HF_TOKEN="$2"; shift 2 ;;
        --skip-hf-login)  SKIP_HF_LOGIN=true; shift ;;
        --skip-build-openmpi) SKIP_BUILD_OPENMPI=true; shift ;;
        --skip-verify)    SKIP_VERIFY=true; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --hf-token TOKEN       HuggingFace token for gated model access"
            echo "  --skip-hf-login        Skip HuggingFace login step"
            echo "  --skip-build-openmpi   Skip OpenMPI build (if already installed)"
            echo "  --skip-verify          Skip end-to-end verification"
            echo "  --help, -h             Show this help message"
            exit 0 ;;
        *)  echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# ── Helper Functions ────────────────────────────────────────────────────────
step()    { echo -e "\n${BLUE}══════════════════════════════════════════════════════════${NC}"; \
            echo -e "${BLUE}  $1${NC}"; \
            echo -e "${BLUE}══════════════════════════════════════════════════════════${NC}"; }
info()    { echo -e "${GREEN}[FP4]${NC} $1"; }
warn()    { echo -e "${YELLOW}[FP4]${NC} $1"; }
fail()    { echo -e "${RED}[FP4]${NC} $1"; exit 1; }

# =============================================================================
# 0. Pre-flight Checks
# =============================================================================
step "Pre-flight checks"

if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found. This script requires an NVIDIA GPU environment."
fi

if ! command -v python3 &>/dev/null; then
    fail "python3 not found. This script requires Python 3.10+."
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
info "GPU: $GPU_NAME (SM $COMPUTE_CAP)"

MAJOR_SM=$(echo "$COMPUTE_CAP" | cut -d. -f1)
if [ "$MAJOR_SM" -ge 10 ] 2>/dev/null; then
    info "Blackwell architecture confirmed -- native FP4 Tensor Cores available"
else
    warn "GPU compute capability $COMPUTE_CAP -- FP4 requires Blackwell (SM 10.0+)"
    warn "Continuing setup anyway, but FP4 inference may not use hardware acceleration"
fi

info "Pre-flight checks passed"

# =============================================================================
# 1. System Dependencies
# =============================================================================
step "Step 1/6: Installing system build dependencies"

if [ "$EUID" -ne 0 ]; then SUDO="sudo"; else SUDO=""; fi

$SUDO apt-get update -qq
$SUDO apt-get install -y -qq \
    build-essential \
    wget \
    python3-dev \
    > /dev/null 2>&1

info "System build dependencies installed"

# =============================================================================
# 2. Build OpenMPI with Internal PMIx (Docker/vast.ai MPI Fix)
# =============================================================================
step "Step 2/6: Building OpenMPI with internal PMIx (Docker/vast.ai fix)"

# Problem:
#   Ubuntu 24.04 ships OpenMPI 4.1.6 with an ext3x PMIx module that depends
#   on system PMIx 5.0.1. Inside Docker containers (especially vast.ai),
#   MPI_Init hangs because orted cannot initialize a PMIx server -- the
#   ext3x module expects PMIx v3.x wire protocol but finds PMIx 5.0.1.
#
# Why this must happen BEFORE TensorRT-LLM:
#   TRT-LLM's C++ bindings (tensorrt_llm.bindings) link against libmpi.so.40
#   at import time. Without a working OpenMPI on LD_LIBRARY_PATH, even
#   `import tensorrt_llm` fails with: "libmpi.so.40: cannot open shared object"
#
# Fix:
#   Rebuild OpenMPI 4.1.6 from source with --with-pmix=internal, which
#   bundles a compatible PMIx version and avoids the system mismatch.

if [ "$SKIP_BUILD_OPENMPI" = true ]; then
    info "Skipping OpenMPI build (--skip-build-openmpi)"
elif [ -f "$OMPI_INSTALL/lib/libmpi.so" ]; then
    info "OpenMPI ${OMPI_VERSION} already installed at ${OMPI_INSTALL}"
else
    BUILD_DIR="/tmp/openmpi-build"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Download
    if [ ! -f "openmpi-${OMPI_VERSION}.tar.gz" ]; then
        info "Downloading OpenMPI ${OMPI_VERSION}..."
        wget -q "https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OMPI_VERSION}.tar.gz"
    fi

    info "Extracting..."
    tar xf "openmpi-${OMPI_VERSION}.tar.gz"
    cd "openmpi-${OMPI_VERSION}"

    # Configure with internal PMIx/libevent/hwloc
    info "Configuring (internal PMIx, libevent, hwloc)..."
    ./configure \
        --prefix="$OMPI_INSTALL" \
        --with-pmix=internal \
        --with-libevent=internal \
        --with-hwloc=internal \
        --disable-mpi-fortran \
        --enable-mpi-cxx \
        --without-verbs \
        --disable-dlopen \
        --enable-static=no \
        --enable-shared=yes \
        > /tmp/openmpi-configure.log 2>&1

    # Build
    NCORES=$(nproc)
    info "Building with ${NCORES} cores (this takes ~5 minutes)..."
    make -j"$NCORES" > /tmp/openmpi-build.log 2>&1

    # Install
    info "Installing to ${OMPI_INSTALL}..."
    make install > /tmp/openmpi-install.log 2>&1

    # Test
    info "Testing MPI_Init..."
    cat > /tmp/test_mpi_init.c << 'MPIEOF'
#include <mpi.h>
#include <stdio.h>
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("MPI OK: rank=%d size=%d\n", rank, size);
    MPI_Finalize();
    return 0;
}
MPIEOF

    "$OMPI_INSTALL/bin/mpicc" /tmp/test_mpi_init.c -o /tmp/test_mpi_init 2>/dev/null
    if timeout 10 /tmp/test_mpi_init 2>/dev/null; then
        info "OpenMPI ${OMPI_VERSION} built and verified successfully"
    else
        fail "OpenMPI MPI_Init test failed. Check logs: /tmp/openmpi-*.log"
    fi

    # Cleanup
    rm -rf "$BUILD_DIR"
    cd "$SCRIPT_DIR"
    info "Build directory cleaned up"
fi

# Activate OpenMPI library paths so TensorRT-LLM can find libmpi.so.40
# during install-time import verification and all subsequent steps.
if [ -d "$OMPI_INSTALL/lib" ]; then
    export LD_LIBRARY_PATH="$OMPI_INSTALL/lib:${LD_LIBRARY_PATH}"
    export LD_PRELOAD="$OMPI_INSTALL/lib/libmpi.so"
    export PATH="$OMPI_INSTALL/bin:$PATH"
    export OMPI_MCA_btl_vader_single_copy_mechanism=none
    info "OpenMPI libraries activated (LD_LIBRARY_PATH + LD_PRELOAD)"
else
    warn "OpenMPI lib dir not found at $OMPI_INSTALL/lib -- TRT-LLM import may fail"
fi

# =============================================================================
# 3. Install TensorRT-LLM + PyTorch (if not already present)
# =============================================================================
step "Step 3/6: TensorRT-LLM and PyTorch"

NEED_TRTLLM=false
NEED_TORCH=false

if python3 -c "import tensorrt_llm" 2>/dev/null; then
    TRTLLM_VER=$(python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)" 2>&1 | grep -v "TensorRT LLM version" | tail -1)
    info "TensorRT-LLM already installed: v${TRTLLM_VER}"
else
    NEED_TRTLLM=true
fi

if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    info "PyTorch already installed: v${TORCH_VER} (CUDA available)"
else
    NEED_TORCH=true
fi

if [ "$NEED_TORCH" = true ] && [ "$NEED_TRTLLM" = true ]; then
    info "Installing TensorRT-LLM (this will also install PyTorch + TensorRT + CUDA deps)..."
    info "This may take several minutes on first install..."
    pip install --upgrade pip 2>&1 | tail -1 || true
    pip install "tensorrt-llm>=1.1.0" \
        --extra-index-url https://pypi.nvidia.com \
        2>&1 | tail -5
    info "TensorRT-LLM installed"
elif [ "$NEED_TRTLLM" = true ]; then
    info "Installing TensorRT-LLM..."
    pip install "tensorrt-llm>=1.1.0" \
        --extra-index-url https://pypi.nvidia.com \
        2>&1 | tail -5
    info "TensorRT-LLM installed"
elif [ "$NEED_TORCH" = true ]; then
    info "Installing PyTorch with CUDA support..."
    pip install "torch>=2.9.0" \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        2>&1 | tail -5
    info "PyTorch installed"
fi

# Install remaining Python dependencies from requirements file
if [ -f "$SCRIPT_DIR/requirements_fp4.txt" ]; then
    info "Installing Python dependencies from requirements_fp4.txt..."
    pip install -r "$SCRIPT_DIR/requirements_fp4.txt" \
        --extra-index-url https://pypi.nvidia.com \
        2>&1 | tail -5
    info "Python dependencies installed"
fi

# Verify critical packages are now available
python3 -c "import tensorrt_llm; print(f'TensorRT-LLM: {tensorrt_llm.__version__}')" 2>/dev/null \
    || fail "TensorRT-LLM installation failed. Check pip output above."
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
    || fail "PyTorch CUDA not available after installation. Check CUDA drivers."

info "TensorRT-LLM and PyTorch verified"

# =============================================================================
# 4. Install flash-attn (Required by Qwen2.5-VL Vision Encoder)
# =============================================================================
step "Step 4/6: Installing flash-attn"

# TensorRT-LLM's Qwen2.5-VL model sets _attn_implementation='flash_attention_2'
# for the vision encoder. Without flash-attn, model loading fails with:
#   ImportError: FlashAttention2 has been toggled on, but it cannot be used
#   due to the following error: the package flash_attn seems to be not installed.

if python3 -c "import flash_attn" 2>/dev/null; then
    FA_VER=$(python3 -c "import flash_attn; print(flash_attn.__version__)")
    info "flash-attn already installed: v${FA_VER}"
else
    info "Installing flash-attn (this takes ~1-2 minutes to compile)..."
    pip install flash-attn --no-build-isolation 2>&1 | tail -3
    info "flash-attn installed"
fi

# =============================================================================
# 5. Patch TensorRT-LLM for Qwen2.5-VL / Cosmos NVFP4 Compatibility
# =============================================================================
step "Step 5/6: Patching TensorRT-LLM for Qwen2.5-VL / Cosmos NVFP4 models"

# Problem:
#   The nvidia/Qwen2.5-VL-7B-Instruct-NVFP4 model has a config where:
#     - Top-level config.rope_scaling.type = "mrope" (handled by TRT-LLM)
#     - text_config.rope_scaling.type = "default" (NOT handled by TRT-LLM)
#     - text_config.rope_scaling.rope_type = "default" (NOT handled by TRT-LLM)
#
#   When TRT-LLM creates the language model sub-component from text_config, it
#   calls PositionEmbeddingType.from_string("default") and
#   RotaryScalingType.from_string("default"), both of which raise ValueError.
#
# Fix:
#   Add alias mappings in functional.py so that:
#     - PositionEmbeddingType: "default" -> "rope_gpt_neox" (standard RoPE)
#     - RotaryScalingType:     "default" -> "none" (no scaling)
#   These are semantically correct: HuggingFace "default" rope_type means
#   standard RoPE with no scaling, which maps to rope_gpt_neox / none.

TRTLLM_FUNCTIONAL=$(python3 -c "
import tensorrt_llm, os
print(os.path.join(os.path.dirname(tensorrt_llm.__file__), 'functional.py'))
" 2>/dev/null | tail -1)

if [ ! -f "$TRTLLM_FUNCTIONAL" ]; then
    fail "Cannot find tensorrt_llm/functional.py at: $TRTLLM_FUNCTIONAL"
fi

# ── Patch 1: RotaryScalingType.from_string ──────────────────────────────────
# Check if already patched
if grep -q "'default': 'none'" "$TRTLLM_FUNCTIONAL" 2>/dev/null; then
    info "RotaryScalingType patch already applied"
else
    info "Patching RotaryScalingType.from_string() ..."
    python3 -c "
path = '$TRTLLM_FUNCTIONAL'
with open(path, 'r') as f:
    content = f.read()

old = '''    @staticmethod
    def from_string(s):
        try:
            return RotaryScalingType[s]
        except KeyError:
            raise ValueError(f'Unsupported rotary scaling type: {s}')'''

new = '''    @staticmethod
    def from_string(s):
        # Map HuggingFace rope_type names to TRT-LLM rotary scaling types
        _aliases = {
            'default': 'none',
        }
        s = _aliases.get(s, s)
        try:
            return RotaryScalingType[s]
        except KeyError:
            raise ValueError(f'Unsupported rotary scaling type: {s}')'''

if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('  RotaryScalingType patched')
else:
    print('  RotaryScalingType: already patched or different TRT-LLM version')
"
fi

# ── Patch 2: PositionEmbeddingType.from_string ─────────────────────────────
if grep -q "'default': 'rope_gpt_neox'" "$TRTLLM_FUNCTIONAL" 2>/dev/null; then
    info "PositionEmbeddingType patch already applied"
else
    info "Patching PositionEmbeddingType.from_string() ..."
    python3 -c "
path = '$TRTLLM_FUNCTIONAL'
with open(path, 'r') as f:
    content = f.read()

old = '''    @staticmethod
    def from_string(s):
        try:
            return PositionEmbeddingType[s]
        except KeyError:
            raise ValueError(f'Unsupported position embedding type: {s}')'''

new = '''    @staticmethod
    def from_string(s):
        # Map HuggingFace rope_type names to TRT-LLM types
        _aliases = {
            'default': 'rope_gpt_neox',
            'linear': 'rope_gpt_neox',
        }
        s = _aliases.get(s, s)
        try:
            return PositionEmbeddingType[s]
        except KeyError:
            raise ValueError(f'Unsupported position embedding type: {s}')'''

if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('  PositionEmbeddingType patched')
else:
    print('  PositionEmbeddingType: already patched or different TRT-LLM version')
"
fi

info "TensorRT-LLM patches applied"

# =============================================================================
# 6. HuggingFace Login (for Gated NVIDIA Models)
# =============================================================================
step "Step 6/6: HuggingFace authentication"

# The NVIDIA models are gated on HuggingFace. You must:
#   1. Accept the license at https://huggingface.co/nvidia/Cosmos-Reason1-7B
#      (and/or https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-NVFP4)
#   2. Provide a HuggingFace token with at least "Read" permission

if [ "$SKIP_HF_LOGIN" = true ]; then
    info "Skipping HuggingFace login (--skip-hf-login)"
    warn "You must log in manually before running inference:"
    warn "  huggingface-cli login --token <YOUR_TOKEN>"
elif [ -n "$HF_TOKEN" ]; then
    info "Logging in to HuggingFace with provided token..."
    huggingface-cli login --token "$HF_TOKEN" 2>&1 | grep -v "^$"
    info "HuggingFace login complete"
else
    # Check if already logged in
    if python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
        HF_USER=$(python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
        info "Already logged in to HuggingFace as: $HF_USER"
    else
        warn "Not logged in to HuggingFace."
        warn "The NVFP4 model is gated -- you need a token to download it."
        warn ""
        warn "To log in, run:"
        warn "  huggingface-cli login --token <YOUR_TOKEN>"
        warn ""
        warn "Or re-run this script with:"
        warn "  bash setup_fp4_vast.sh --hf-token <YOUR_TOKEN>"
        warn ""
        warn "Get a token at: https://huggingface.co/settings/tokens"
        warn "Accept license at: https://huggingface.co/nvidia/Cosmos-Reason1-7B"
        warn "                    https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-NVFP4"
    fi
fi

# =============================================================================
# Verification
# =============================================================================

if [ "$SKIP_VERIFY" = false ]; then
    step "Verification: Testing TRT-LLM import with patched OpenMPI"

    # Activate the environment
    source "$SCRIPT_DIR/activate_fp4.sh" 2>/dev/null

    python3 -c "
import sys
print('Python:', sys.version.split()[0])

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
cap = torch.cuda.get_device_capability(0)
print(f'Compute Capability: {cap[0]}.{cap[1]}')

import tensorrt_llm
print(f'TensorRT-LLM: {tensorrt_llm.__version__}')

import flash_attn
print(f'flash-attn: {flash_attn.__version__}')

from tensorrt_llm.functional import RotaryScalingType, PositionEmbeddingType
# Verify patches work
assert RotaryScalingType.from_string('default') == RotaryScalingType.none, \
    'RotaryScalingType patch not working'
assert PositionEmbeddingType.from_string('default') == PositionEmbeddingType.rope_gpt_neox, \
    'PositionEmbeddingType patch not working'
print('TRT-LLM patches: verified')

print()
print('All checks passed. Environment is ready for FP4 inference.')
print()
print('Next steps:')
print('  source activate_fp4.sh')
print('  python3 quantize_cosmos_fp4.py                                         # Quantize Cosmos (once)')
print('  python3 cosmos_fp4_inference.py --video_dir stu_dataset/stu_videos     # Cosmos FP4')
print('  python3 trtllm_fp4_inference.py --video_dir stu_dataset/stu_videos     # Qwen FP4')
print('  python3 fp8_inference.py --video_dir stu_dataset/stu_videos            # INT8 comparison')
" 2>&1
fi

# =============================================================================
# Summary
# =============================================================================
step "Setup Complete"

echo -e "
${GREEN}FP4 environment is ready for Blackwell Tensor Core inference.${NC}

Setup applied:
  1. OpenMPI ${OMPI_VERSION} with internal PMIx  →  ${OMPI_INSTALL}/
     (fixes MPI_Init hang in Docker/vast.ai; built first so TRT-LLM can link libmpi.so)
  2. TensorRT-LLM + PyTorch + Python dependencies installed
  3. flash-attn installed
     (required by Qwen2.5-VL vision encoder)
  4. TensorRT-LLM patched: RotaryScalingType + PositionEmbeddingType
     (fixes 'default' rope_type in Qwen2.5-VL/Cosmos NVFP4 config)

Usage (Cosmos-Reason1-7B FP4 -- recommended):
  ${GREEN}source activate_fp4.sh${NC}
  ${GREEN}python3 quantize_cosmos_fp4.py${NC}                                         # Quantize (once, ~2 min)
  ${GREEN}python3 cosmos_fp4_inference.py --video_dir stu_dataset/stu_videos${NC}     # FP4 inference

Usage (Qwen2.5-VL FP4 -- pre-quantized):
  ${GREEN}source activate_fp4.sh${NC}
  ${GREEN}python3 trtllm_fp4_inference.py --video_dir stu_dataset/stu_videos${NC}

Usage (INT8 comparison):
  ${GREEN}source activate_fp4.sh${NC}
  ${GREEN}python3 fp8_inference.py --video_dir stu_dataset/stu_videos${NC}

Files:
  setup_fp4_vast.sh           ← This setup script (run once)
  activate_fp4.sh             ← Environment activation (source before each session)
  quantize_cosmos_fp4.py      ← Quantize Cosmos-Reason1-7B to NVFP4 (run once)
  cosmos_fp4_inference.py     ← Cosmos FP4 inference (recommended)
  trtllm_fp4_inference.py     ← Qwen FP4 inference
  fp8_inference.py            ← INT8 inference (bitsandbytes)
  requirements_fp4.txt        ← Python package requirements
"
