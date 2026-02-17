#!/usr/bin/env python3
"""
Quantize nvidia/Cosmos-Reason1-7B to NVFP4 for TensorRT-LLM inference.

Creates a HuggingFace-compatible NVFP4 checkpoint that TRT-LLM can load
directly via its LLM API. The quantization uses NVIDIA ModelOpt with the
same settings as nvidia/Qwen2.5-VL-7B-Instruct-NVFP4:

    - Algorithm: max (dynamic scaling per group)
    - Group size: 16
    - Excludes: vision encoder (visual*), lm_head
    - Calibration: 512 samples from cnn_dailymail (text-only, fast)

This takes ~5-10 minutes on a single RTX 5090 (32 GB VRAM).

Prerequisites:
    source activate_fp4.sh

Usage:
    python3 quantize_cosmos_fp4.py
    python3 quantize_cosmos_fp4.py --output_dir ./cosmos-reason1-nvfp4
    python3 quantize_cosmos_fp4.py --num_calib_samples 256 --calib_batch_size 4
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch

MODEL_NAME = "nvidia/Cosmos-Reason1-7B"
DEFAULT_OUTPUT = "./cosmos-reason1-nvfp4"


def get_calib_dataloader(tokenizer, num_samples=512, batch_size=2, max_length=512):
    """Create calibration dataloader from CNN/DailyMail (text-only, fast)."""
    from datasets import load_dataset

    print(f"  Loading calibration data: cnn_dailymail ({num_samples} samples)...")
    dataset = load_dataset(
        "cnn_dailymail", "3.0.0", split="validation", streaming=True
    )

    calib_texts = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        calib_texts.append(sample["article"][:max_length * 4])

    print(f"  Tokenizing {len(calib_texts)} calibration samples...")
    batches = []
    for i in range(0, len(calib_texts), batch_size):
        batch_texts = calib_texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        batches.append(encoded["input_ids"].cuda())

    return batches


def quantize_model(model, tokenizer, num_calib_samples, calib_batch_size):
    """Apply NVFP4 quantization with ModelOpt."""
    import modelopt.torch.quantization as mtq

    # Start from NVFP4_DEFAULT_CFG and add vision encoder exclusion
    quant_cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)

    # Exclude the vision encoder from quantization (same as NVIDIA's checkpoint).
    # The vision encoder is small (~0.6B params) and sensitive to aggressive
    # quantization. Keeping it in BF16 preserves image understanding quality.
    quant_cfg["quant_cfg"]["*visual*"] = {"enable": False}

    print(f"\nQuantization config:")
    print(f"  Algorithm: {quant_cfg['algorithm']}")
    print(f"  Bits: NVFP4 (2-bit mantissa, 1-bit exponent, group_size=16)")
    print(f"  Excluded: visual*, lm_head*, proj_out*, router*, gate*")
    print(f"  Calibration samples: {num_calib_samples}")

    # Get calibration data
    calib_batches = get_calib_dataloader(
        tokenizer,
        num_samples=num_calib_samples,
        batch_size=calib_batch_size,
    )

    # Define forward loop for calibration
    def forward_loop(model):
        print(f"  Running calibration ({len(calib_batches)} batches)...")
        for i, input_ids in enumerate(calib_batches):
            if i % 50 == 0:
                print(f"    Batch {i}/{len(calib_batches)}")
            with torch.no_grad():
                model(input_ids=input_ids)

    # Quantize
    print("\n  Applying NVFP4 quantization...")
    quant_start = time.time()
    model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    quant_time = time.time() - quant_start
    print(f"  Quantization complete in {quant_time:.1f}s")

    return model


def save_checkpoint(model, tokenizer, output_dir, source_model):
    """Save quantized model as HuggingFace-compatible NVFP4 checkpoint.

    Uses ModelOpt's export_hf_checkpoint to pack weights into the proper
    FP4 format (uint8 packed weights + fp8 scales) that TRT-LLM expects.
    """
    from modelopt.torch.export import export_hf_checkpoint

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Exporting packed NVFP4 checkpoint to {output_dir}...")
    save_start = time.time()

    # Export with ModelOpt -- this packs weights into uint8 FP4 format
    # with proper scale factors (weight_scale, weight_scale_2, input_scale)
    export_hf_checkpoint(
        model,
        dtype=torch.bfloat16,
        export_dir=output_dir,
    )

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Copy processor config from source model
    from huggingface_hub import hf_hub_download
    import shutil
    for fname in ["preprocessor_config.json", "video_preprocessor_config.json",
                   "chat_template.jinja"]:
        try:
            src = hf_hub_download(source_model, fname)
            shutil.copy2(src, output_dir / fname)
        except Exception:
            pass

    # Fix rope_scaling.type to 'mrope' (Cosmos has 'default' but uses mrope)
    config_path = output_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    if config.get("rope_scaling", {}).get("type") == "default":
        config["rope_scaling"]["type"] = "mrope"
    if config.get("torch_dtype") is None:
        config["torch_dtype"] = "bfloat16"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    save_time = time.time() - save_start
    print(f"  Saved in {save_time:.1f}s")

    # Print size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"  Checkpoint size: {total_size / 1e9:.2f} GB")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Cosmos-Reason1-7B to NVFP4 for TensorRT-LLM"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME,
        help=f"Source model (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--num_calib_samples", type=int, default=512,
        help="Number of calibration samples (default: 512)"
    )
    parser.add_argument(
        "--calib_batch_size", type=int, default=2,
        help="Calibration batch size (default: 2)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Cosmos-Reason1-7B NVFP4 Quantization")
    print("=" * 70)

    # Verify GPU
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu} ({mem:.1f} GB)")
    else:
        raise RuntimeError("CUDA not available")

    print(f"Source model: {args.model}")
    print(f"Output: {args.output_dir}")

    # Check if already quantized
    output_path = Path(args.output_dir)
    if (output_path / "hf_quant_config.json").exists():
        print(f"\nNVFP4 checkpoint already exists at {output_path}")
        print("Delete it to re-quantize: rm -rf", args.output_dir)
        return

    # Load model in BF16 (requires ~15 GB VRAM)
    print(f"\nStep 1/3: Loading {args.model} in BF16...")
    load_start = time.time()

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.1f}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"  GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Quantize
    print(f"\nStep 2/3: NVFP4 quantization with {args.num_calib_samples} calibration samples...")
    total_start = time.time()
    model = quantize_model(model, tokenizer, args.num_calib_samples, args.calib_batch_size)

    # Save
    print(f"\nStep 3/3: Saving NVFP4 checkpoint...")
    save_checkpoint(model, tokenizer, args.output_dir, args.model)

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"Quantization complete in {total_time:.1f}s")
    print(f"NVFP4 checkpoint saved to: {args.output_dir}")
    print(f"\nTo run inference:")
    print(f"  source activate_fp4.sh")
    print(f"  python3 cosmos_fp4_inference.py --video_dir stu_dataset/stu_videos")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
