#!/bin/bash
# =============================================================================
# DEPRECATED -- Use setup_fp4_vast.sh instead
# =============================================================================
#
# This script has been superseded by setup_fp4_vast.sh, which handles:
#   - OpenMPI rebuild for Docker/vast.ai (MPI_Init hang fix)
#   - flash-attn installation (Qwen2.5-VL vision encoder)
#   - TensorRT-LLM patches (rope_type "default" compatibility)
#   - HuggingFace authentication (gated NVIDIA model)
#   - End-to-end verification
#
# Usage:
#   bash setup_fp4_vast.sh                      # Full setup
#   bash setup_fp4_vast.sh --hf-token <TOKEN>   # With HF token
#
# =============================================================================

echo "This script has been replaced by setup_fp4_vast.sh"
echo ""
echo "Usage:"
echo "  bash setup_fp4_vast.sh                      # Full setup"
echo "  bash setup_fp4_vast.sh --hf-token <TOKEN>   # With HF token"
echo "  bash setup_fp4_vast.sh --help               # Show all options"
echo ""
echo "Run it now? (y/n)"
read -r answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_fp4_vast.sh" "$@"
fi
