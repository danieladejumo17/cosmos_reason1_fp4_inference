#!/bin/bash
# =============================================================================
# FP16 Inference Environment Setup
# =============================================================================
#
# Installs Python dependencies for:
#   - batched_fp16_inference.py  (batched FP16 video anomaly detection)
#   - fp16_inference.py          (single-video FP16 inference)
#   - harzard_prcpt/generate_hpt_5s_videos.py  (HPT dataset preparation)
#   - metrics.py                 (classification metrics + confusion matrix)
#
# Usage:
#   bash setup_fp16.sh              # Install into current Python env
#   bash setup_fp16.sh --venv       # Create a venv first, then install
#
# Requires: Python 3.10+, CUDA-capable GPU with PyTorch support
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv_fp16"
USE_VENV=false

for arg in "$@"; do
    case "$arg" in
        --venv) USE_VENV=true ;;
        *) echo -e "${RED}Unknown argument: $arg${NC}"; exit 1 ;;
    esac
done

# --- Optional venv creation ---
if $USE_VENV; then
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}Creating virtual environment at $VENV_DIR ...${NC}"
        python3 -m venv "$VENV_DIR"
    fi
    echo -e "${GREEN}Activating venv: $VENV_DIR${NC}"
    source "$VENV_DIR/bin/activate"
fi

echo -e "${GREEN}Installing FP16 inference dependencies...${NC}"

pip install --upgrade pip

pip install \
    torch \
    transformers \
    accelerate \
    qwen-vl-utils \
    decord \
    opencv-python-headless \
    natsort \
    scikit-learn \
    matplotlib \
    seaborn \
    numpy

echo ""
echo -e "${GREEN}✅ All dependencies installed.${NC}"
echo ""
echo "Verify with:"
echo "  python -c \"import torch; print('PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available())\""
echo "  python -c \"import transformers; print('Transformers', transformers.__version__)\""
echo ""
echo "Run inference:"
echo "  python batched_fp16_inference.py --video_dir ./videos --batch_size 2"
echo "  python fp16_inference.py --video_dir ./videos"
echo ""
echo "Generate HPT clips:"
echo "  cd harzard_prcpt && python generate_hpt_5s_videos.py"
