#!/usr/bin/env python3
"""
True FP8 W8A8 inference using llm-compressor + vLLM.

Two-phase pipeline:
  1. Quantize: llm-compressor applies FP8_DYNAMIC to the model (one-time).
     - Weights:      FP8 E4M3, static per-channel scales (simple PTQ, no calibration data)
     - Activations:  FP8 E4M3, dynamic per-token scales at inference time
     - Excluded:     vision encoder (minimal savings, high accuracy cost) + lm_head
     Saves a compressed-tensors checkpoint that vLLM can load natively.

  2. Infer: vLLM loads the checkpoint and runs actual FP8 GEMM kernels on
     Ada Lovelace / Hopper / Blackwell tensor cores.

Prerequisites:
    pip install llmcompressor vllm qwen-vl-utils

Usage:
    # Quantize + infer (first run creates the FP8 checkpoint automatically)
    python3 fp8_vllm_inference.py --video_dir harzard_prcpt/data/hpt_1.5s_videos

    # Quantize only (skip inference)
    python3 fp8_vllm_inference.py --quantize-only

    # Infer with an existing checkpoint
    python3 fp8_vllm_inference.py --video_dir stu_dataset/stu_videos --fp8_dir ./my-fp8-checkpoint
"""

import argparse
import gc
import json
import re
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "nvidia/Cosmos-Reason1-7B"
DEFAULT_FP8_DIR = str(Path(__file__).resolve().parent / "cosmos-reason1-fp8-dynamic")


# ============================================================
# 1. FP8 Quantization (one-time, no calibration data needed)
# ============================================================
def quantize_model(source_model: str, output_dir: str):
    """Quantize model to FP8 W8A8 via llm-compressor and save checkpoint."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.transformers.compression.compressed_tensors_utils import get_model_compressor
    from safetensors.torch import save_file
    import shutil

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading {source_model} for quantization...")
    load_start = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        source_model, dtype="auto"
    )
    processor = AutoProcessor.from_pretrained(source_model)
    print(f"  Loaded in {time.time() - load_start:.1f}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head", "re:visual.*", "re:model.visual.*"],
    )

    print("\nApplying FP8_DYNAMIC quantization...")
    quant_start = time.time()
    oneshot(model=model, recipe=recipe)
    print(f"  Quantization complete in {time.time() - quant_start:.1f}s")

    print(f"\nSaving compressed-tensors checkpoint to {output_dir}...")
    save_start = time.time()

    compressor = get_model_compressor(
        model=model, save_compressed=True,
        skip_sparsity_compression_stats=True,
    )
    if compressor is not None:
        compressor.compress_model(model)

    state_dict = model.state_dict()
    state_dict = {k: v.contiguous().cpu() for k, v in state_dict.items()}

    max_shard_bytes = 5 * 1024**3
    shards = []
    current_shard = {}
    current_size = 0
    for key, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor_size
    if current_shard:
        shards.append(current_shard)

    weight_map = {}
    for i, shard in enumerate(shards, 1):
        shard_name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(output_path / shard_name))
        for key in shard:
            weight_map[key] = shard_name

    index = {
        "metadata": {"total_parameters": sum(p.numel() for p in model.parameters())},
        "weight_map": weight_map,
    }
    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    model.config.save_pretrained(output_dir)
    if compressor is not None:
        compressor.update_config(output_dir)
    processor.save_pretrained(output_dir)

    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"  Saved in {time.time() - save_start:.1f}s ({total_size / 1e9:.2f} GB)")

    del model, state_dict, shards, compressor
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

    return output_dir


# ============================================================
# 2. Video Loading
# ============================================================
def load_video(video_path: Path, target_fps: int, target_resolution: tuple[int, int]):
    import cv2
    import qwen_vl_utils

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        effective_fps = min(target_fps, native_fps) if native_fps > 0 else target_fps
        duration = frame_count / native_fps if native_fps > 0 else 0
        num_frames = max(1, int(duration * effective_fps))
        total_pixels = num_frames * target_resolution[0] * target_resolution[1]
    finally:
        if "cap" in locals() and cap.isOpened():
            cap.release()

    message = [{"role": "user", "content": [
        {"type": "video", "video": str(video_path), "total_pixels": total_pixels}
    ]}]
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(message)
    return image_inputs, video_inputs


# ============================================================
# 3. Prompt
# ============================================================
def get_analysis_prompt():
    return (
        "Analyze this driving video for external anomalies that may impact "
        "safe autonomous vehicle operation.\n\n"
        "Look carefully for:\n"
        "- Obstacles, pedestrians, or vehicles violating traffic rules\n"
        "- Roadwork, blocked lanes, poor visibility, or road hazards\n"
        "- Any unusual or unsafe condition\n\n"
        "<think>\n"
        "Describe what you observe in the video step by step.\n"
        "</think>\n\n"
        "<answer>\n"
        "Based on your analysis, classify this video with exactly one of:\n"
        "Classification: Anomaly\n"
        "Classification: Normal\n"
        "</answer>"
    )


# ============================================================
# 4. Result Parsing (robust CoT parser from FP4 pipeline)
# ============================================================
def parse_result(raw_output: str) -> str:
    out = raw_output.lower()

    matches = list(re.finditer(
        r"classification[:\s]+(?:is\s+)?[\"']?(anomaly|normal)", out
    ))
    if matches:
        return "Anomaly" if matches[-1].group(1) == "anomaly" else "Normal"

    answer_idx = out.rfind("<answer>")
    if answer_idx != -1:
        answer_text = out[answer_idx:]
        if re.search(r"\banomal(?:y|ies)\b", answer_text):
            return "Anomaly"
        if "normal" in answer_text:
            return "Normal"

    lines = [l.strip() for l in out.strip().split("\n")
             if l.strip() and not l.strip().startswith("</")]
    if lines:
        last_line = lines[-1]
        has_anomaly = bool(re.search(r"\banomal(?:y|ies)\b", last_line))
        negated = bool(re.search(r"\b(?:no|not|without)\b.*\banomal", last_line))
        if has_anomaly and not negated:
            return "Anomaly"
        if "normal" in last_line or (has_anomaly and negated):
            return "Normal"

    if re.search(r"\b(?:no|not|without|doesn.t|don.t|aren.t)\b.*\banomal(?:y|ies)\b", out):
        return "Normal"
    if re.search(r"\banomal(?:y|ies)\b", out):
        return "Anomaly"
    if "normal" in out:
        return "Normal"

    return "Unknown"


# ============================================================
# 5. Single-Video Analysis via vLLM
# ============================================================
def analyze_video(llm, processor, video_path: Path, prefetched_data, max_tokens: int, prompt_text: str):
    from vllm import SamplingParams

    image_inputs, video_inputs = prefetched_data

    conversation = [{"role": "user", "content": [
        {"type": "video", "video": str(video_path)},
        {"type": "text", "text": prompt_text},
    ]}]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    mm_data = {}
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    if image_inputs is not None:
        mm_data["image"] = image_inputs

    outputs = llm.generate(
        {"prompt": text, "multi_modal_data": mm_data},
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0),
        use_tqdm=False,
    )

    return outputs[0].outputs[0].text.strip()


# ============================================================
# 6. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="FP8 W8A8 Inference — llm-compressor quantization + vLLM FP8 GEMM kernels"
    )
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Directory containing video files")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help=f"Source HF model to quantize (default: {MODEL_NAME})")
    parser.add_argument("--fp8_dir", type=str, default=DEFAULT_FP8_DIR,
                        help=f"FP8 checkpoint directory (default: {DEFAULT_FP8_DIR})")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--target_resolution", type=str, default="250x250")
    parser.add_argument("--gpu_mem_utilization", type=float, default=0.60,
                        help="vLLM GPU memory utilization (default: 0.60)")
    parser.add_argument("--quantize-only", action="store_true",
                        help="Only quantize the model; skip inference")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save JSON results (default: <video_dir>/fp8_vllm_results.json)")
    args = parser.parse_args()

    if not args.quantize_only and args.video_dir is None:
        parser.error("--video_dir is required unless --quantize-only is set")

    # GPU info
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print("=" * 70)
    print("FP8 W8A8 Inference — llm-compressor + vLLM")
    print("=" * 70)
    print(f"GPU: {gpu_name} (SM {compute_cap[0]}.{compute_cap[1]}, {mem_gb:.1f} GB)")
    if compute_cap[0] < 9:
        print(f"WARNING: FP8 tensor cores require SM >= 8.9 (Ada Lovelace+). "
              f"Detected SM {compute_cap[0]}.{compute_cap[1]}.")

    # --- Phase 1: Quantize (if needed) ---
    fp8_path = Path(args.fp8_dir)
    if not (fp8_path / "config.json").exists():
        print(f"\nFP8 checkpoint not found at {fp8_path}")
        print("Running one-time FP8_DYNAMIC quantization...\n")
        quantize_model(args.model, args.fp8_dir)
        print(f"\nFP8 checkpoint ready at {args.fp8_dir}")
    else:
        print(f"\nFP8 checkpoint found at {fp8_path}")

    if args.quantize_only:
        print("--quantize-only set; skipping inference.")
        return

    # --- Phase 2: Inference via vLLM ---
    width, height = map(int, args.target_resolution.split("x"))
    target_resolution = (width, height)

    video_dir = Path(args.video_dir)
    video_files = sorted([
        f for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm")
        for f in video_dir.glob(ext)
    ])
    if not video_files:
        print(f"No video files found in {video_dir}")
        return

    output_json_path = Path(args.output_json) if args.output_json else video_dir / "fp8_vllm_results.json"

    from vllm import LLM
    from transformers import AutoProcessor

    print(f"\nLoading FP8 model in vLLM from {args.fp8_dir}...")
    load_start = time.time()
    llm = LLM(
        model=args.fp8_dir,
        limit_mm_per_prompt={"video": 1},
        gpu_memory_utilization=args.gpu_mem_utilization,
        trust_remote_code=True,
        dtype="auto",
    )
    model_load_time = time.time() - load_start
    print(f"  vLLM engine ready in {model_load_time:.1f}s")

    processor = AutoProcessor.from_pretrained(args.fp8_dir)

    # Warmup
    from vllm import SamplingParams
    print("Warming up vLLM engine...")
    llm.generate(
        [{"prompt": "Hello, describe this scene briefly."}],
        SamplingParams(max_tokens=5),
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    print("Warmup complete.\n")

    prompt_text = get_analysis_prompt()
    print(f"Processing {len(video_files)} videos with FP8 W8A8 (vLLM)\n" + "-" * 70)

    from metrics import Metrics
    results = []
    total_load_time = 0.0
    total_inference_time = 0.0
    metrics = Metrics()
    counts = {"Anomaly": 0, "Normal": 0, "Unknown": 0, "Error": 0}

    batch_start = time.time()

    for i, video_path in enumerate(video_files, 1):
        fname = video_path.stem
        if fname.startswith("Anom"):
            true_label = 1
        elif fname.startswith("Norm"):
            true_label = 0
        else:
            true_label = None

        vload_start = time.time()
        try:
            prefetched = load_video(video_path, args.fps, target_resolution)
            vload_time = time.time() - vload_start
            total_load_time += vload_time

            infer_start = time.time()
            raw = analyze_video(llm, processor, video_path, prefetched, args.max_tokens, prompt_text)
            infer_time = time.time() - infer_start
            total_inference_time += infer_time

            result = parse_result(raw)
            counts[result] += 1

            pred_label = 1 if result == "Anomaly" else 0
            if true_label is not None:
                metrics.update([pred_label], [true_label], [infer_time])

            results.append({
                "file": video_path.name,
                "result": result,
                "raw_output": raw,
                "true_label": "Anomaly" if true_label == 1 else ("Normal" if true_label == 0 else "Unknown"),
                "load_time_s": round(vload_time, 3),
                "inference_time_s": round(infer_time, 3),
            })

            print(
                f"[{i}/{len(video_files)}] {video_path.name}: {result} "
                f"(Load: {vload_time:.2f}s, FP8 Inference: {infer_time:.2f}s)"
            )

        except Exception as e:
            counts["Error"] += 1
            results.append({
                "file": video_path.name,
                "result": "Error",
                "raw_output": str(e),
                "load_time_s": 0.0,
                "inference_time_s": 0.0,
            })
            print(f"[{i}/{len(video_files)}] {video_path.name}: ERROR - {e}")

    total_time = time.time() - batch_start
    successful = len(video_files) - counts["Error"]

    # --- Output ---
    output_data = {
        "config": {
            "engine": "vLLM",
            "model": args.model,
            "checkpoint": args.fp8_dir,
            "quantization": "FP8 W8A8 (llm-compressor FP8_DYNAMIC)",
            "weight_format": "FP8 E4M3, static per-channel",
            "activation_format": "FP8 E4M3, dynamic per-token",
            "compute_type": "FP8 GEMM kernels (vLLM)",
            "fps": args.fps,
            "max_tokens": args.max_tokens,
            "target_resolution": args.target_resolution,
            "gpu": gpu_name,
            "compute_capability": f"{compute_cap[0]}.{compute_cap[1]}",
        },
        "summary": {
            "total_videos": len(video_files),
            "anomalies": counts["Anomaly"],
            "normals": counts["Normal"],
            "unknowns": counts["Unknown"],
            "errors": counts["Error"],
            "total_load_time_s": round(total_load_time, 3),
            "total_inference_time_s": round(total_inference_time, 3),
            "total_time_s": round(total_time, 3),
            "avg_inference_time_s": round(total_inference_time / successful, 3) if successful else None,
            "model_load_time_s": round(model_load_time, 1),
        },
        "metrics": metrics.compute() if metrics.count > 0 else None,
        "results": results,
    }

    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print("-" * 70)
    print("\nSUMMARY — FP8 W8A8 Inference (llm-compressor + vLLM)")
    print("=" * 70)
    print(f"Engine: vLLM with FP8 GEMM kernels")
    print(f"Quantization: FP8_DYNAMIC (weights=per-channel, activations=per-token)")
    print(f"Total videos: {len(video_files)}")
    print(f"  - Anomaly: {counts['Anomaly']}")
    print(f"  - Normal:  {counts['Normal']}")
    print(f"  - Unknown: {counts['Unknown']}")
    print(f"  - Errors:  {counts['Error']}")
    print(f"\nTotal load time: {total_load_time:.2f}s")
    print(f"Total FP8 inference time: {total_inference_time:.2f}s")
    if successful > 0:
        print(f"Average FP8 inference: {total_inference_time / successful:.3f}s per video")

    if metrics.count > 0:
        m = metrics.compute()
        print(f"\nCLASSIFICATION METRICS")
        print("=" * 70)
        print(f"  TP: {m['TP']}  TN: {m['TN']}  FP: {m['FP']}  FN: {m['FN']}")
        print(f"  Accuracy:  {m['Accuracy']:.4f}")
        print(f"  Precision: {m['Precision']:.4f}")
        print(f"  Recall:    {m['Recall']:.4f}")
        print(f"  F1-Score:  {m['F1-Score']:.4f}")
        print(f"  Avg Inference Time: {m['Avg Inference Time']:.3f}s")

    print(f"\nResults saved to: {output_json_path}")
    print("FP8 W8A8 inference complete.")


if __name__ == "__main__":
    main()
