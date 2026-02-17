# FP4 Vision-Language Model Inference on Blackwell GPUs

### Usage
"""
bash setup_fp4_vast.sh --hf-token hf_YOUR_TOKEN_HERE
source activate_fp4.sh
python3 trtllm_fp4_inference.py --video_dir stu_dataset/stu_videos
"""

Accelerated video anomaly detection using **NVIDIA TensorRT-LLM** with **native FP4 Tensor Core** inference on RTX 5090 (Blackwell architecture). This project runs a Vision-Language Model (VLM) to classify autonomous driving videos as normal or anomalous, achieving **0.47s per video** with NVFP4 quantization.

## Overview

This repo provides three quantization/inference approaches for the same VLM task, benchmarked on the [STU dataset](https://huggingface.co/datasets/danieladejumo/stu_dataset) (Semantic Traffic Understanding):

| Method | Script | Model | Avg Inference | Quantization |
|---|---|---|---|---|
| **FP4 TensorRT-LLM** | `trtllm_fp4_inference.py` | `nvidia/Qwen2.5-VL-7B-Instruct-NVFP4` | **0.47s/video** | NVFP4 (native Blackwell Tensor Cores) |
| INT8 bitsandbytes | `fp8_inference.py` | `nvidia/Cosmos-Reason1-7B` | 1.01s/video | INT8 (bitsandbytes `load_in_8bit`) |
| FP4 PyTorch+ModelOpt | `fp4_inference.py` | `nvidia/Cosmos-Reason1-7B` | ~1.5s/video | Simulated FP4 (ModelOpt quantizers) |

The FP4 TensorRT-LLM path is the fastest because it compiles FP4 GEMM kernels that execute directly on Blackwell's FP4 Tensor Cores, using NVIDIA's pre-quantized NVFP4 checkpoint calibrated on CNN/DailyMail data.

## Hardware Requirements

- **GPU**: NVIDIA RTX 5090 (Blackwell, SM 10.0+) with 32 GB VRAM
- **Environment**: [vast.ai](https://vast.ai) Docker instance (or any Docker environment with TensorRT-LLM)
- **Docker image**: Must include TensorRT-LLM >= 1.1.0 and PyTorch >= 2.9.0 with CUDA

## Quick Start (vast.ai)

### 1. Provision a vast.ai instance

Select an RTX 5090 instance with a Docker image that includes TensorRT-LLM (e.g., an NVIDIA NGC TensorRT-LLM container or a community image with the stack pre-installed).

### 2. Clone the repo

```bash
git clone <this-repo-url> /workspace/temp
cd /workspace/temp
```

### 3. Run the setup script

This single script handles everything: OpenMPI rebuild, flash-attn installation, TensorRT-LLM patches, and HuggingFace authentication.

```bash
bash setup_fp4_vast.sh --hf-token <YOUR_HUGGINGFACE_TOKEN>
```

You need a HuggingFace token because the NVFP4 model is gated. To get one:
1. Create an account at [huggingface.co](https://huggingface.co)
2. Go to [Settings > Tokens](https://huggingface.co/settings/tokens) and create a **Read** token
3. Accept the license at [nvidia/Qwen2.5-VL-7B-Instruct-NVFP4](https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-NVFP4)

### 4. Download the STU dataset

```bash
bash setup_stu_dataset.sh
```

### 5. Run FP4 inference

```bash
source activate_fp4.sh
python3 trtllm_fp4_inference.py --video_dir stu_dataset/stu_videos
```

The first run downloads the model (~6 GB) and builds the TensorRT-LLM engine (~55s). Subsequent runs reuse the cached model.

## What the Setup Script Does

`setup_fp4_vast.sh` applies five fixes required to make TensorRT-LLM FP4 inference work inside vast.ai Docker containers:

### Fix 1: OpenMPI with Internal PMIx

**Problem**: Ubuntu 24.04 ships OpenMPI 4.1.6 with a PMIx `ext3x` module that depends on system PMIx 5.0.1. Inside Docker containers, `MPI_Init` hangs because `orted` cannot initialize a PMIx server -- the `ext3x` module expects PMIx v3.x wire protocol but finds PMIx 5.0.1.

**Fix**: Rebuild OpenMPI 4.1.6 from source with `--with-pmix=internal`, which bundles a compatible PMIx version. Installed to `/usr/local/openmpi-4.1.6/`.

### Fix 2: Flash Attention

**Problem**: TensorRT-LLM sets `_attn_implementation='flash_attention_2'` for the Qwen2.5-VL vision encoder. Without the `flash-attn` package, model loading fails.

**Fix**: `pip install flash-attn --no-build-isolation`

### Fix 3: TensorRT-LLM Rope Type Patches

**Problem**: The `nvidia/Qwen2.5-VL-7B-Instruct-NVFP4` model config has `rope_scaling.rope_type = "default"` in its `text_config`. TensorRT-LLM v1.1.0's `PositionEmbeddingType.from_string()` and `RotaryScalingType.from_string()` don't recognize `"default"`, causing `ValueError` during model loading.

**Fix**: Patch `tensorrt_llm/functional.py` to map HuggingFace's `"default"` rope type to the correct TRT-LLM enum values:
- `PositionEmbeddingType`: `"default"` -> `"rope_gpt_neox"` (standard RoPE)
- `RotaryScalingType`: `"default"` -> `"none"` (no scaling)

### Fix 4: HuggingFace Authentication

**Problem**: The NVFP4 model is gated on HuggingFace and requires license acceptance + a token.

**Fix**: `huggingface-cli login --token <TOKEN>`

### Fix 5: Video Data Format Conversion

**Problem**: `qwen_vl_utils.process_vision_info()` returns video frames as a single tensor of shape `(N, C, H, W)`, but TRT-LLM's multimodal input processor expects each video as a Python list of individual `(C, H, W)` frame tensors.

**Fix**: Handled in `trtllm_fp4_inference.py` -- converts `tensor(N,C,H,W)` to `[tensor(C,H,W), ...]` before passing to `llm.generate()`.

## File Reference

### Scripts

| File | Description |
|---|---|
| `setup_fp4_vast.sh` | **One-time setup** for a fresh vast.ai instance. Run once. |
| `activate_fp4.sh` | **Environment activation**. Source before each session. |
| `build_openmpi_docker.sh` | Standalone OpenMPI build (called by `setup_fp4_vast.sh`). |
| `setup_stu_dataset.sh` | Downloads and extracts the STU dataset. |
| `setup_fp4_env.sh` | Deprecated. Redirects to `setup_fp4_vast.sh`. |

### Inference Scripts

| File | Description |
|---|---|
| `trtllm_fp4_inference.py` | **FP4 inference via TensorRT-LLM** (recommended, fastest). Uses `nvidia/Qwen2.5-VL-7B-Instruct-NVFP4`. |
| `fp8_inference.py` | INT8 inference via bitsandbytes. Uses `nvidia/Cosmos-Reason1-7B`. |
| `fp4_inference.py` | FP4 inference via PyTorch + ModelOpt (simulated quantization). |
| `fp4_quantization.py` | FP4 quantization calibration script (for PyTorch+ModelOpt path). |

### Configuration

| File | Description |
|---|---|
| `requirements_fp4.txt` | Python package requirements for FP4 inference. |
| `.gitignore` | Excludes model checkpoints, videos, engines, and caches. |

### Dataset

| File | Description |
|---|---|
| `stu_dataset/` | STU dataset module -- video processing, dataloader, metrics. |
| `stu_dataset/stu_video_dataset.py` | Dataset class and video dataloader. |
| `stu_dataset/stu_experiments.ipynb` | Experiment notebook. |
| `stu_dataset/requirements.txt` | Dataset-specific requirements. |

## Usage

### FP4 TensorRT-LLM Inference (Recommended)

```bash
source activate_fp4.sh

# Basic usage
python3 trtllm_fp4_inference.py --video_dir stu_dataset/stu_videos

# Save results to JSON
python3 trtllm_fp4_inference.py \
    --video_dir stu_dataset/stu_videos \
    --output_file results.json

# Custom settings
python3 trtllm_fp4_inference.py \
    --video_dir stu_dataset/stu_videos \
    --fps 8 \
    --max_tokens 15 \
    --target_resolution 512x512
```

### INT8 bitsandbytes Inference

```bash
source activate_fp4.sh
python3 fp8_inference.py --video_dir stu_dataset/stu_videos
```

### Verify Environment

```bash
source activate_fp4.sh --verify
```

This checks all components: TensorRT-LLM, flash-attn, ModelOpt, TRT-LLM patches, GPU, and HuggingFace login status.

## Architecture

The FP4 TensorRT-LLM pipeline:

```
Video File
    |
    v
qwen_vl_utils.process_vision_info()    # Extract frames at target FPS
    |
    v
AutoProcessor.apply_chat_template()    # Generate prompt with <video_pad> tokens
    |
    v
tensorrt_llm.LLM.generate()            # TRT-LLM PyTorch backend
    |-- Vision Encoder (Flash Attention 2)
    |-- NVFP4 Language Model (FP4 GEMM on Blackwell Tensor Cores)
    |-- Paged KV Cache (~17 GB)
    |-- M-RoPE positional encoding (temporal + spatial)
    |
    v
Classification: Normal / Anomaly
```

## Troubleshooting

### `MPI_Init` hangs (no output after "rank 0 using MpiPoolSession")

The OpenMPI fix hasn't been applied. Run:
```bash
bash setup_fp4_vast.sh --skip-hf-login
```

### `ImportError: flash_attn seems to be not installed`

```bash
pip install flash-attn --no-build-isolation
```

### `ValueError: Unsupported position embedding type: default`

The TRT-LLM patch hasn't been applied. Run:
```bash
bash setup_fp4_vast.sh --skip-build-openmpi --skip-hf-login
```

### CUDA out of memory

Kill leftover GPU processes from a previous run:
```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
```

Then retry. The model needs ~28 GB VRAM (FP4 weights + KV cache + vision encoder).

### HuggingFace model download fails / times out

Make sure you're logged in and have accepted the model license:
```bash
huggingface-cli login --token <YOUR_TOKEN>
# Then visit https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-NVFP4 and click "Submit"
```
