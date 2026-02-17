# FP4 Vision-Language Model Inference on Blackwell GPUs

Accelerated video anomaly detection using **NVIDIA TensorRT-LLM** with **native FP4 Tensor Core** inference on RTX 5090 (Blackwell architecture). This project runs Vision-Language Models (VLMs) to classify autonomous driving videos as normal or anomalous, achieving **0.47s per video** with NVFP4 quantization on Cosmos-Reason1-7B.

## Quick Start

```bash
# One-time setup (vast.ai Docker instance with TensorRT-LLM)
bash setup_fp4_vast.sh --hf-token hf_YOUR_TOKEN_HERE

# Download STU dataset
bash setup_stu_dataset.sh

# Activate environment
source activate_fp4.sh

# Quantize Cosmos-Reason1-7B to NVFP4 (once, ~2 minutes)
python3 quantize_cosmos_fp4.py

# Run FP4 inference
python3 cosmos_fp4_inference.py --video_dir stu_dataset/stu_videos --output_file cosmos_fp4_results.json

# Run INT8 inference (for comparison)
python3 fp8_inference.py --video_dir stu_dataset/stu_videos --output_json cosmos_int8_results.json
```

## Overview

This repo provides multiple quantization/inference approaches for VLM-based video anomaly detection, benchmarked on the [STU dataset](https://huggingface.co/datasets/danieladejumo/stu_dataset) (Semantic Traffic Understanding):

| Method | Script | Model | Avg Inference | Quantization |
|---|---|---|---|---|
| **FP4 TensorRT-LLM (Cosmos)** | `cosmos_fp4_inference.py` | `nvidia/Cosmos-Reason1-7B` | **0.47s/video** | NVFP4 (native Blackwell Tensor Cores) |
| FP4 TensorRT-LLM (Qwen) | `trtllm_fp4_inference.py` | `nvidia/Qwen2.5-VL-7B-Instruct-NVFP4` | 0.44s/video | NVFP4 (native Blackwell Tensor Cores) |
| INT8 bitsandbytes (Cosmos) | `fp8_inference.py` | `nvidia/Cosmos-Reason1-7B` | 1.04s/video | INT8 (bitsandbytes `load_in_8bit`) |
| FP4 PyTorch+ModelOpt | `fp4_inference.py` | `nvidia/Cosmos-Reason1-7B` | ~1.5s/video | Simulated FP4 (ModelOpt quantizers) |

The Cosmos-Reason1-7B model is NVIDIA's reasoning-enhanced VLM fine-tuned from Qwen2.5-VL-7B-Instruct on physical world reasoning data. The FP4 TensorRT-LLM path is the fastest because it compiles FP4 GEMM kernels that execute directly on Blackwell's FP4 Tensor Cores.

## FP4 vs INT8 Comparison (Cosmos-Reason1-7B)

Both methods use identical parameters (4 fps, 250x250 resolution, max 7 tokens) on the same 13 STU videos:

### Speed

| Metric | FP4 (TRT-LLM) | INT8 (bitsandbytes) | Delta |
|---|---|---|---|
| Model load time | 49.5s | 34.6s | INT8 faster (simpler loader) |
| Total inference (13 videos) | 6.11s | 13.50s | **FP4 2.2x faster** |
| Avg inference per video | **0.47s** | 1.04s | **FP4 2.2x faster** |
| Video load time | 9.94s | 10.16s | Equal |

### Classifications

| Video | FP4 Result | INT8 Result | Match |
|---|---|---|---|
| temp_display.mp4 | Normal | Normal | Yes |
| vid_30.mp4 | Normal | Normal | Yes |
| vid_50.mp4 | Normal | Normal | Yes |
| vid_70.mp4 | Normal | **Anomaly** ("The dog") | No |
| vid_90.mp4 | Normal | **Anomaly** ("The bicycle") | No |
| vid_110.mp4 | Normal | **Anomaly** ("The kick") | No |
| vid_130.mp4 | Normal | Normal | Yes |
| vid_150.mp4 | Normal | Normal | Yes |
| vid_170.mp4 | Normal | Normal | Yes |
| vid_190.mp4 | Normal | Normal | Yes |
| vid_210.mp4 | Normal | Normal | Yes |
| vid_230.mp4 | Normal | Normal | Yes |
| vid_250.mp4 | Normal | Normal | Yes |

**Agreement rate: 76.9% (10/13)**. INT8 flagged 3 additional anomalies that FP4 classified as Normal. The INT8 model (8-bit weights) appears more sensitive to edge cases involving animals and objects, while FP4 (4-bit weights) is more conservative. Without ground truth labels, neither can be declared more accurate, but the trade-off is clear: FP4 is 2.2x faster at the cost of potentially reduced sensitivity.

## Hardware Requirements

- **GPU**: NVIDIA RTX 5090 (Blackwell, SM 10.0+) with 32 GB VRAM
- **Environment**: [vast.ai](https://vast.ai) Docker instance (or any Linux environment with NVIDIA drivers)
- **Software**: Python 3.10+, NVIDIA drivers + CUDA toolkit (TensorRT-LLM and PyTorch are installed automatically by the setup script)

## Setup (vast.ai)

### 1. Provision a vast.ai instance

Select an RTX 5090 instance with a Docker image that includes NVIDIA drivers and CUDA toolkit (e.g., an NVIDIA NGC container or a community image with the CUDA stack). TensorRT-LLM and PyTorch are installed automatically by the setup script if not present.

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

You need a HuggingFace token because the models are gated. To get one:
1. Create an account at [huggingface.co](https://huggingface.co)
2. Go to [Settings > Tokens](https://huggingface.co/settings/tokens) and create a **Read** token
3. Accept the license for the models you want to use:
   - [nvidia/Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B)
   - [nvidia/Qwen2.5-VL-7B-Instruct-NVFP4](https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-NVFP4) (if using the Qwen path)

### 4. Download the STU dataset

```bash
bash setup_stu_dataset.sh
```

### 5. Quantize Cosmos-Reason1-7B to NVFP4

This creates a local NVFP4 checkpoint (~7 GB) using ModelOpt. Only needs to be run once.

```bash
source activate_fp4.sh
python3 quantize_cosmos_fp4.py
```

The quantization:
- Loads Cosmos-Reason1-7B in BF16 (~15 GB)
- Calibrates on 512 samples from CNN/DailyMail (text-only -- vision encoder is excluded)
- Exports packed FP4 weights (uint8 + fp8 scales) to `./cosmos-reason1-nvfp4/`
- Takes ~2 minutes on RTX 5090

### 6. Run inference

```bash
source activate_fp4.sh

# Cosmos-Reason1 FP4 (recommended)
python3 cosmos_fp4_inference.py --video_dir stu_dataset/stu_videos --output_file cosmos_fp4_results.json

# Cosmos-Reason1 INT8 (for comparison)
python3 fp8_inference.py --video_dir stu_dataset/stu_videos --output_json cosmos_int8_results.json

# Qwen2.5-VL FP4 (uses NVIDIA's pre-quantized checkpoint)
python3 trtllm_fp4_inference.py --video_dir stu_dataset/stu_videos --output_file qwen_fp4_results.json
```

## What the Setup Script Does

`setup_fp4_vast.sh` applies five fixes required to make TensorRT-LLM FP4 inference work inside vast.ai Docker containers:

### Fix 1: OpenMPI with Internal PMIx

**Problem**: Ubuntu 24.04 ships OpenMPI 4.1.6 with a PMIx `ext3x` module that depends on system PMIx 5.0.1. Inside Docker containers, `MPI_Init` hangs because `orted` cannot initialize a PMIx server -- the `ext3x` module expects PMIx v3.x wire protocol but finds PMIx 5.0.1.

**Fix**: Rebuild OpenMPI 4.1.6 from source with `--with-pmix=internal`, which bundles a compatible PMIx version. Installed to `/usr/local/openmpi-4.1.6/`.

### Fix 2: Flash Attention

**Problem**: TensorRT-LLM sets `_attn_implementation='flash_attention_2'` for the Qwen2.5-VL vision encoder. Without the `flash-attn` package, model loading fails.

**Fix**: `pip install flash-attn --no-build-isolation`

### Fix 3: TensorRT-LLM Rope Type Patches

**Problem**: Qwen2.5-VL-based model configs have `rope_scaling.rope_type = "default"` in their `text_config`. TensorRT-LLM v1.1.0's `PositionEmbeddingType.from_string()` and `RotaryScalingType.from_string()` don't recognize `"default"`, causing `ValueError` during model loading.

**Fix**: Patch `tensorrt_llm/functional.py` to map HuggingFace's `"default"` rope type to the correct TRT-LLM enum values:
- `PositionEmbeddingType`: `"default"` -> `"rope_gpt_neox"` (standard RoPE)
- `RotaryScalingType`: `"default"` -> `"none"` (no scaling)

### Fix 4: HuggingFace Authentication

**Problem**: The NVIDIA models are gated on HuggingFace and require license acceptance + a token.

**Fix**: `huggingface-cli login --token <TOKEN>`

### Fix 5: Video Data Format Conversion

**Problem**: `qwen_vl_utils.process_vision_info()` returns video frames as a single tensor of shape `(N, C, H, W)`, but TRT-LLM's multimodal input processor expects each video as a Python list of individual `(C, H, W)` frame tensors.

**Fix**: Handled in `cosmos_fp4_inference.py` and `trtllm_fp4_inference.py` -- converts `tensor(N,C,H,W)` to `[tensor(C,H,W), ...]` before passing to `llm.generate()`.

## File Reference

### Scripts

| File | Description |
|---|---|
| `setup_fp4_vast.sh` | **One-time setup** for a fresh vast.ai instance. Run once. |
| `activate_fp4.sh` | **Environment activation**. Source before each session. |
| `build_openmpi_docker.sh` | Standalone OpenMPI build (called by `setup_fp4_vast.sh`). |
| `setup_stu_dataset.sh` | Downloads and extracts the STU dataset. |
| `setup_fp4_env.sh` | Deprecated. Redirects to `setup_fp4_vast.sh`. |

### Inference & Quantization Scripts

| File | Description |
|---|---|
| `cosmos_fp4_inference.py` | **Cosmos-Reason1-7B FP4 inference via TensorRT-LLM** (recommended, fastest). |
| `quantize_cosmos_fp4.py` | Quantize Cosmos-Reason1-7B to NVFP4 using ModelOpt. Run once before `cosmos_fp4_inference.py`. |
| `trtllm_fp4_inference.py` | Qwen2.5-VL FP4 inference via TensorRT-LLM. Uses `nvidia/Qwen2.5-VL-7B-Instruct-NVFP4`. |
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

## Usage Examples

### Cosmos-Reason1 FP4 Inference (Recommended)

```bash
source activate_fp4.sh

# Basic usage
python3 cosmos_fp4_inference.py --video_dir stu_dataset/stu_videos

# Save results to JSON
python3 cosmos_fp4_inference.py \
    --video_dir stu_dataset/stu_videos \
    --output_file cosmos_fp4_results.json

# Custom settings
python3 cosmos_fp4_inference.py \
    --video_dir stu_dataset/stu_videos \
    --fps 8 \
    --max_tokens 15 \
    --target_resolution 512x512
```

### Qwen2.5-VL FP4 Inference

```bash
source activate_fp4.sh
python3 trtllm_fp4_inference.py --video_dir stu_dataset/stu_videos --output_file qwen_fp4_results.json
```

### INT8 bitsandbytes Inference

```bash
source activate_fp4.sh
python3 fp8_inference.py --video_dir stu_dataset/stu_videos --output_json cosmos_int8_results.json
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
    |-- Vision Encoder (Flash Attention 2, BF16)
    |-- NVFP4 Language Model (FP4 GEMM on Blackwell Tensor Cores)
    |-- Paged KV Cache (~17 GB)
    |-- M-RoPE positional encoding (temporal + spatial)
    |
    v
Classification: Normal / Anomaly
```

Cosmos-Reason1 NVFP4 quantization pipeline:

```
nvidia/Cosmos-Reason1-7B (BF16, ~15 GB)
    |
    v
ModelOpt NVFP4 quantization
    |-- Calibration: 512 CNN/DailyMail samples (text-only)
    |-- Algorithm: "max" (per-group-of-16 weight scaling)
    |-- Excluded: vision encoder (*visual*), lm_head
    |
    v
export_hf_checkpoint()
    |-- Packed uint8 weights + fp8 scales
    |-- Config: rope_scaling.type = "mrope", torch_dtype = "bfloat16"
    |
    v
./cosmos-reason1-nvfp4/ (~7 GB checkpoint)
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

Then retry. The FP4 model needs ~28 GB VRAM (FP4 weights + KV cache + vision encoder).

### HuggingFace model download fails / times out

Make sure you're logged in and have accepted the model license:
```bash
huggingface-cli login --token <YOUR_TOKEN>
# Then visit the model page and click "Agree and access repository"
# https://huggingface.co/nvidia/Cosmos-Reason1-7B
# https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-NVFP4
```

### Cosmos NVFP4 quantization produces wrong-size checkpoint

If the checkpoint is ~15 GB instead of ~7 GB, the export didn't pack weights correctly. Make sure you're using `modelopt.torch.export.export_hf_checkpoint` (not `model.save_pretrained`). Re-run:
```bash
rm -rf ./cosmos-reason1-nvfp4
python3 quantize_cosmos_fp4.py
```
