#!/bin/bash
# =============================================================================
# FP4 Quantization Environment Setup Script
# =============================================================================
# 
# This script sets up the environment for NVFP4 quantization with true FP4
# compute on NVIDIA Blackwell GPUs (RTX 5090, SM 10.0+)
#
# Usage:
#   bash setup_fp4_env.sh
#
# Or with options:
#   bash setup_fp4_env.sh --no-system-deps    # Skip system dependencies
#   bash setup_fp4_env.sh --venv              # Create a virtual environment
#   bash setup_fp4_env.sh --help              # Show help
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default options
INSTALL_SYSTEM_DEPS=true
CREATE_VENV=false
VENV_NAME="fp4_env"

# =============================================================================
# Functions
# =============================================================================

print_header() {
    echo -e "${BLUE}"
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

show_help() {
    echo "FP4 Quantization Environment Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --no-system-deps    Skip installation of system dependencies"
    echo "  --venv              Create a Python virtual environment"
    echo "  --venv-name NAME    Name for virtual environment (default: fp4_env)"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                      # Full installation"
    echo "  $0 --venv               # Install in virtual environment"
    echo "  $0 --no-system-deps     # Skip apt-get packages"
    echo ""
}

check_gpu() {
    print_step "Checking GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo "")
        
        if [ -n "$GPU_INFO" ]; then
            echo "  Detected GPU: $GPU_INFO"
            
            # Check for Blackwell (SM 10.0+)
            COMPUTE_CAP=$(echo "$GPU_INFO" | cut -d',' -f2 | tr -d ' ')
            MAJOR_VERSION=$(echo "$COMPUTE_CAP" | cut -d'.' -f1)
            
            if [ "$MAJOR_VERSION" -ge 10 ] 2>/dev/null; then
                print_success "Blackwell GPU detected - Native FP4 Tensor Cores available!"
            else
                print_warning "GPU compute capability $COMPUTE_CAP detected."
                print_warning "Native FP4 requires Blackwell (SM 10.0+). FP4 may be emulated."
            fi
        else
            print_warning "Could not query GPU information"
        fi
    else
        print_warning "nvidia-smi not found. Make sure NVIDIA drivers are installed."
    fi
}

install_system_deps() {
    print_step "Installing system dependencies..."
    
    # Check if running as root or with sudo
    if [ "$EUID" -ne 0 ]; then
        SUDO="sudo"
    else
        SUDO=""
    fi
    
    # Update package list
    $SUDO apt-get update
    
    # Install required packages
    $SUDO apt-get install -y \
        libopenmpi-dev \
        openmpi-bin \
        build-essential \
        python3-dev \
        python3-pip \
        python3-venv \
        git \
        wget \
        curl
    
    print_success "System dependencies installed"
}

setup_venv() {
    print_step "Creating virtual environment: $VENV_NAME"
    
    # Create virtual environment
    python3 -m venv "$SCRIPT_DIR/$VENV_NAME"
    
    # Activate it
    source "$SCRIPT_DIR/$VENV_NAME/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip wheel setuptools
    
    print_success "Virtual environment created and activated"
    echo "  To activate later: source $SCRIPT_DIR/$VENV_NAME/bin/activate"
}

install_pip_packages() {
    print_step "Installing Python packages..."
    
    # NVIDIA PyPI index URL
    NVIDIA_INDEX="https://pypi.nvidia.com"
    
    echo "  Installing from requirements_fp4.txt..."
    pip install -r "$SCRIPT_DIR/requirements_fp4.txt" \
        --extra-index-url "$NVIDIA_INDEX"
    
    # Explicitly install CUDA 13 cuBLAS (may not be in requirements)
    echo "  Ensuring CUDA 13 cuBLAS is installed..."
    pip install "nvidia-cublas>=13.0.0" --extra-index-url "$NVIDIA_INDEX" 2>/dev/null || true
    
    # Fix numpy version if open3d installed numpy>=2 (tensorrt-llm requires <2)
    echo "  Ensuring numpy<2 for TensorRT-LLM compatibility..."
    pip install "numpy<2" 2>/dev/null || true
    
    print_success "Python packages installed"
}

setup_environment_variables() {
    print_step "Setting up environment variables..."
    
    # Find the CUDA 13 library path
    PYTHON_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
    CUDA13_LIB_PATH="$PYTHON_SITE_PACKAGES/nvidia/cu13/lib"
    
    if [ -d "$CUDA13_LIB_PATH" ]; then
        echo "  Found CUDA 13 libraries at: $CUDA13_LIB_PATH"
        
        # Create activation script
        ACTIVATE_SCRIPT="$SCRIPT_DIR/activate_fp4.sh"
        
        cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# FP4 Environment Activation Script
# Source this file to set up the environment for FP4 quantization
#
# Usage: source activate_fp4.sh

# Add CUDA 13 libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CUDA13_LIB_PATH:\${LD_LIBRARY_PATH}"

# Activate virtual environment if it exists
if [ -f "$SCRIPT_DIR/$VENV_NAME/bin/activate" ]; then
    source "$SCRIPT_DIR/$VENV_NAME/bin/activate"
fi

echo "FP4 environment activated"
echo "  LD_LIBRARY_PATH includes: $CUDA13_LIB_PATH"

# Verify TensorRT-LLM can be imported
python3 -c "import tensorrt_llm; print(f'  TensorRT-LLM version: {tensorrt_llm.__version__}')" 2>/dev/null || echo "  Warning: TensorRT-LLM import check skipped"
EOF
        
        chmod +x "$ACTIVATE_SCRIPT"
        print_success "Created activation script: $ACTIVATE_SCRIPT"
    else
        print_warning "CUDA 13 library path not found at expected location"
        print_warning "You may need to manually set LD_LIBRARY_PATH"
    fi
}

verify_installation() {
    print_step "Verifying installation..."
    
    # Set LD_LIBRARY_PATH for verification
    PYTHON_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
    export LD_LIBRARY_PATH="$PYTHON_SITE_PACKAGES/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
    
    echo "  Checking Python packages..."
    
    # Check key packages
    PACKAGES=("torch" "transformers" "modelopt" "tensorrt_llm" "bitsandbytes" "open3d")
    ALL_OK=true
    
    for pkg in "${PACKAGES[@]}"; do
        if python3 -c "import $pkg" 2>/dev/null; then
            VERSION=$(python3 -c "import $pkg; print(getattr($pkg, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
            echo "    ✓ $pkg ($VERSION)"
        else
            echo "    ✗ $pkg - NOT INSTALLED"
            ALL_OK=false
        fi
    done
    
    # Check GPU
    echo ""
    echo "  Checking CUDA availability..."
    python3 -c "
import torch
if torch.cuda.is_available():
    print(f'    ✓ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'    ✓ Compute capability: {torch.cuda.get_device_capability(0)}')
else:
    print('    ✗ CUDA not available')
" 2>/dev/null || echo "    ✗ Could not check CUDA"
    
    if $ALL_OK; then
        print_success "All packages verified successfully!"
    else
        print_warning "Some packages may need manual installation"
    fi
}

# =============================================================================
# Parse command line arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-system-deps)
            INSTALL_SYSTEM_DEPS=false
            shift
            ;;
        --venv)
            CREATE_VENV=true
            shift
            ;;
        --venv-name)
            VENV_NAME="$2"
            CREATE_VENV=true
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# =============================================================================
# Main Installation
# =============================================================================

print_header "FP4 Quantization Environment Setup"

echo "Configuration:"
echo "  Script directory: $SCRIPT_DIR"
echo "  Install system deps: $INSTALL_SYSTEM_DEPS"
echo "  Create virtual env: $CREATE_VENV"
if $CREATE_VENV; then
    echo "  Virtual env name: $VENV_NAME"
fi
echo ""

# Check GPU first
check_gpu

# Install system dependencies
if $INSTALL_SYSTEM_DEPS; then
    install_system_deps
else
    print_warning "Skipping system dependencies (--no-system-deps)"
fi

# Create virtual environment if requested
if $CREATE_VENV; then
    setup_venv
fi

# Install pip packages
install_pip_packages

# Setup environment variables
setup_environment_variables

# Verify installation
verify_installation

# =============================================================================
# Final Instructions
# =============================================================================

print_header "Setup Complete!"

echo "To use the FP4 environment:"
echo ""
echo "  1. Activate the environment:"
echo "     source $SCRIPT_DIR/activate_fp4.sh"
echo ""
echo "  2. Run FP4 quantization (optional, for calibrated FP4):"
echo "     python3 fp4_quantization.py --output_dir ./cosmos-fp4-engine"
echo ""
echo "  3. Run FP4 inference (NVFP4 on Blackwell Tensor Cores):"
echo "     python3 fp4_inference.py --video_dir ./videos --engine_dir ./cosmos-fp4-engine"
echo ""
echo "  4. Run INT8 inference (bitsandbytes):"
echo "     python3 fp8_inference.py --video_dir ./videos"
echo ""

if $CREATE_VENV; then
    echo "Virtual environment location: $SCRIPT_DIR/$VENV_NAME"
    echo "Activate with: source $SCRIPT_DIR/$VENV_NAME/bin/activate"
    echo ""
fi

print_success "FP4 environment setup completed!"
