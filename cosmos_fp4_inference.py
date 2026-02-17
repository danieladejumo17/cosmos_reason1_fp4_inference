#!/usr/bin/env python3
"""
Cosmos-Reason1-7B FP4 Inference via TensorRT-LLM (Blackwell GPUs)

Uses the NVFP4-quantized Cosmos-Reason1-7B model with TensorRT-LLM for
native FP4 Tensor Core inference on RTX 5090.

Cosmos-Reason1-7B is NVIDIA's reasoning-enhanced VLM fine-tuned from
Qwen2.5-VL-7B-Instruct on physical world reasoning data. The NVFP4
checkpoint is created by quantize_cosmos_fp4.py using NVIDIA ModelOpt
with the same quantization recipe as nvidia/Qwen2.5-VL-7B-Instruct-NVFP4.

Prerequisites:
    1. source activate_fp4.sh
    2. python3 quantize_cosmos_fp4.py   (creates the NVFP4 checkpoint)

Usage:
    source activate_fp4.sh
    python3 cosmos_fp4_inference.py --video_dir stu_dataset/stu_videos
    python3 cosmos_fp4_inference.py --video_dir stu_dataset/stu_videos --output_file cosmos_fp4_results.json
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import cv2
import torch

warnings.filterwarnings("ignore", category=UserWarning)

# Cosmos-Reason1-7B quantized to NVFP4 (local checkpoint)
DEFAULT_MODEL = "./cosmos-reason1-nvfp4"
# The original (unquantized) model for processor/tokenizer
SOURCE_MODEL = "nvidia/Cosmos-Reason1-7B"


def verify_gpu():
    """Verify Blackwell GPU is available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. RTX 5090 required for FP4.")

    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    print(f"Memory: {memory_gb:.1f} GB")

    if compute_cap[0] >= 10:
        print("Blackwell architecture detected - Native FP4 Tensor Cores available")
    else:
        print(f"SM {compute_cap[0]}.{compute_cap[1]} detected. FP4 requires Blackwell (SM 10.0+).")

    return gpu_name, compute_cap


def load_trtllm_model(model_path: str):
    """Load Cosmos-Reason1 NVFP4 model via TensorRT-LLM's LLM API."""
    from tensorrt_llm import LLM

    print(f"Loading TRT-LLM model: {model_path}")

    start = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
    )
    elapsed = time.time() - start
    print(f"  TRT-LLM model loaded in {elapsed:.1f}s")

    return llm


def load_video_frames(video_path: Path, target_fps: int, target_resolution: tuple):
    """Load video and extract frames for VLM processing."""
    import qwen_vl_utils

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        effective_fps = min(target_fps, native_fps) if native_fps > 0 else target_fps
        duration = frame_count / native_fps if native_fps > 0 else 0
        num_frames = max(1, int(duration * effective_fps))
        total_pixels = num_frames * target_resolution[0] * target_resolution[1]
    finally:
        cap.release()

    content = [{
        "role": "user",
        "content": [{"type": "video", "video": str(video_path), "total_pixels": total_pixels}]
    }]
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(content)
    return image_inputs, video_inputs


def get_analysis_prompt():
    """Safety analysis prompt for Cosmos-Reason1 (reasoning-enhanced)."""
    return (
        "You are an autonomous driving safety expert analyzing this video for "
        "EXTERNAL ANOMALIES that may impact safe AV operation.\n\n"
        "<think>\n"
        "- Obstacles, pedestrians, or vehicles violating rules\n"
        "- Roadwork, blocked lanes, poor visibility, or hazards\n"
        "- Reflections, shadows, or false visual cues confusing perception\n"
        "</think>\n\n"
        "<answer>\n"
        "Is there any external anomaly in this video? Reply with exactly one "
        "word of the following:\n"
        "Classification: Anomaly -- if any obstacle, obstruction, or unsafe "
        "condition is visible.\n"
        "Classification: Normal -- if no anomaly or obstruction is visible.\n"
        "</answer>"
    )


def analyze_video(llm, processor, video_path: Path, prefetched_data: tuple,
                  max_tokens: int, prompt_text: str):
    """Analyze a video using TRT-LLM's multimodal LLM API with Cosmos-Reason1."""
    from tensorrt_llm import SamplingParams

    image_inputs, video_inputs = prefetched_data

    content_parts = [
        {"type": "video", "video": str(video_path)},
        {"type": "text", "text": prompt_text},
    ]
    conversation = [{"role": "user", "content": content_parts}]
    text_with_tokens = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    # Convert video tensors from (N,C,H,W) to list of (C,H,W) for TRT-LLM
    mm_data = {}
    if video_inputs:
        converted_videos = []
        for v in video_inputs:
            if isinstance(v, torch.Tensor) and v.dim() == 4:
                converted_videos.append(list(v))
            elif isinstance(v, list):
                converted_videos.append(v)
            else:
                converted_videos.append([v])
        mm_data["video"] = converted_videos
    if image_inputs:
        mm_data["image"] = image_inputs

    prompt = {
        "prompt": text_with_tokens,
        "multi_modal_data": mm_data,
    }

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
    )

    output = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)

    if isinstance(output, list):
        output = output[0]

    return output.outputs[0].text.strip()


def parse_result(raw_output: str) -> str:
    """Parse model output to extract classification."""
    out = raw_output.lower()
    if "anomaly" in out:
        return "Anomaly"
    elif "normal" in out:
        return "Normal"
    return "Unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Cosmos-Reason1-7B FP4 Inference (TensorRT-LLM, Blackwell Tensor Cores)",
    )
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"NVFP4 model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--source_model", type=str, default=SOURCE_MODEL,
                        help=f"Original model for processor (default: {SOURCE_MODEL})")
    parser.add_argument("--fps", type=int, default=4,
                        help="Target FPS for video sampling (default: 4)")
    parser.add_argument("--max_tokens", type=int, default=7,
                        help="Max new tokens to generate (default: 7)")
    parser.add_argument("--target_resolution", type=str, default="250x250",
                        help="Target resolution WxH (default: 250x250)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    width, height = map(int, args.target_resolution.split('x'))
    target_resolution = (width, height)

    # Check NVFP4 checkpoint exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: NVFP4 checkpoint not found at '{model_path}'")
        print(f"Run quantization first: python3 quantize_cosmos_fp4.py")
        return

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Error: Video directory '{video_dir}' does not exist.")
        return

    video_files = sorted([
        f for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm")
        for f in video_dir.glob(ext)
    ])
    if not video_files:
        print(f"No video files found in '{video_dir}'.")
        return

    print("=" * 70)
    print("Cosmos-Reason1-7B FP4 Inference -- Blackwell Tensor Core Acceleration")
    print("=" * 70)

    gpu_name, compute_cap = verify_gpu()

    print(f"\nModel: Cosmos-Reason1-7B NVFP4 ({args.model})")
    print(f"Source: {args.source_model}")
    print(f"Video directory: {video_dir}")
    print(f"Videos found: {len(video_files)}")
    print(f"FPS: {args.fps}, Max tokens: {args.max_tokens}")
    print(f"Target resolution: {args.target_resolution}")
    print("=" * 70 + "\n")

    # Load TRT-LLM model
    print("Loading Cosmos-Reason1 NVFP4 model...")
    load_start = time.time()
    llm = load_trtllm_model(args.model)
    model_load_time = time.time() - load_start
    print(f"Model ready in {model_load_time:.1f}s\n")

    # Load processor from the original Cosmos model (has the right chat template)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.source_model, trust_remote_code=True)

    # Warmup
    print("Warming up TRT-LLM engine...")
    from tensorrt_llm import SamplingParams
    llm.generate(
        ["Hello, describe this scene briefly."],
        SamplingParams(max_tokens=5),
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    print("Warmup complete.\n")

    prompt_text = get_analysis_prompt()
    print(f"Processing {len(video_files)} videos with Cosmos-Reason1 FP4...\n" + "-" * 70)

    results = []
    total_inference_time = 0
    total_load_time = 0

    for idx, video_path in enumerate(video_files, 1):
        # Load video
        vload_start = time.time()
        try:
            prefetched = load_video_frames(video_path, args.fps, target_resolution)
        except Exception as e:
            print(f"[{idx}/{len(video_files)}] {video_path.name}: ERROR loading - {e}")
            results.append({"file": video_path.name, "result": "Error", "error": str(e)})
            continue
        vload_time = time.time() - vload_start
        total_load_time += vload_time

        # Inference
        infer_start = time.time()
        try:
            raw = analyze_video(llm, processor, video_path, prefetched, args.max_tokens, prompt_text)
            result = parse_result(raw)
            infer_time = time.time() - infer_start
            total_inference_time += infer_time

            print(f"[{idx}/{len(video_files)}] {video_path.name}: {result} "
                  f"(Load: {vload_time:.2f}s, FP4 Inference: {infer_time:.2f}s)")

            results.append({
                "file": video_path.name,
                "result": result,
                "raw_output": raw,
                "load_time_s": round(vload_time, 3),
                "inference_time_s": round(infer_time, 3),
            })
        except Exception as e:
            infer_time = time.time() - infer_start
            print(f"[{idx}/{len(video_files)}] {video_path.name}: ERROR - {e}")
            results.append({"file": video_path.name, "result": "Error", "error": str(e)})

    # Summary
    print("-" * 70)
    print("\nSUMMARY -- Cosmos-Reason1-7B FP4 Tensor Core Inference")
    print("=" * 70)

    anomalies = sum(1 for r in results if r["result"] == "Anomaly")
    normals = sum(1 for r in results if r["result"] == "Normal")
    errors = sum(1 for r in results if r["result"] == "Error")
    unknowns = sum(1 for r in results if r["result"] == "Unknown")
    successful = len(results) - errors

    print(f"Model: Cosmos-Reason1-7B NVFP4")
    print(f"Engine: TensorRT-LLM (FP4 Tensor Core GEMM)")
    print(f"Total videos: {len(video_files)}")
    print(f"  Anomaly: {anomalies}")
    print(f"  Normal: {normals}")
    print(f"  Unknown: {unknowns}")
    print(f"  Errors: {errors}")
    print(f"\nTotal load time: {total_load_time:.2f}s")
    print(f"Total FP4 inference time: {total_inference_time:.2f}s")

    avg_time = None
    if successful > 0:
        avg_time = total_inference_time / successful
        print(f"Average FP4 inference: {avg_time:.3f}s per video")

    if args.output_file:
        output_data = {
            "config": {
                "engine": "TensorRT-LLM",
                "model": "nvidia/Cosmos-Reason1-7B",
                "checkpoint": args.model,
                "quantization": "NVFP4",
                "compute_type": "FP4 Tensor Core GEMM (Blackwell native)",
                "fps": args.fps,
                "max_tokens": args.max_tokens,
                "target_resolution": args.target_resolution,
                "gpu": gpu_name,
                "compute_capability": f"{compute_cap[0]}.{compute_cap[1]}",
            },
            "summary": {
                "total_videos": len(video_files),
                "anomalies": anomalies,
                "normals": normals,
                "unknowns": unknowns,
                "errors": errors,
                "total_load_time_s": round(total_load_time, 3),
                "total_inference_time_s": round(total_inference_time, 3),
                "avg_inference_time_s": round(avg_time, 3) if avg_time else None,
                "model_load_time_s": round(model_load_time, 1),
            },
            "results": results,
        }

        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")

    print("\nCosmos-Reason1 FP4 inference complete.")


if __name__ == "__main__":
    main()
