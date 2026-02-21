#!/usr/bin/env python3

import argparse
import json
import time
import warnings
from pathlib import Path

import torch
import transformers
import qwen_vl_utils
import cv2

warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'

# ============================================================
# 1. Model Loader
# ============================================================
def load_model(model_name: str):
    print("🔧 Loading and compiling model... This may take a few seconds.")
    start = time.time()

    bnb_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
    ).eval()

    processor = transformers.AutoProcessor.from_pretrained(model_name)
    model.gradient_checkpointing_disable()
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)

    print(f"✅ Model ready in {time.time() - start:.2f}s\n")
    return model, processor


# ============================================================
# 2. Prompt Caching
# ============================================================
def build_cached_prompt(processor):
    base_text = (
        "You are an autonomous driving safety expert analyzing this video for EXTERNAL ANOMALIES that may impact safe AV operation.\n\n"
        "<think>\n"
        "- Obstacles, pedestrians, or vehicles violating rules\n"
        "- Roadwork, blocked lanes, poor visibility, or hazards\n"
        "- Reflections, shadows, or false visual cues confusing perception\n"
        "</think>\n\n"
        "<answer>\n"
        "Is there any external anomaly in this video? Reply with exactly one word of the following:\n"
        "Classification: Anomaly — if any obstacle, obstruction, or unsafe condition is visible.\n"
        "Classification: Normal — if no anomaly or obstruction is visible.\n"
        "</answer>"
    )
    # TODO
    conversation_template = [{"role": "user", "content": [{"type": "text", "text": base_text}]}]
    _ = processor.apply_chat_template(conversation_template, tokenize=False, add_generation_prompt=True)
    return base_text


# ============================================================
# 3. Warmup
# ============================================================
def warmup_model(model, processor):
    print("🔥 Warming up model (compiling kernels)...")
    dummy_conv = [{"role": "user", "content": [{"type": "text", "text": "Is this scene safe?"}]}]
    text = processor.apply_chat_template(dummy_conv, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=7)
    torch.cuda.synchronize()
    print("✅ Warmup complete.\n")


# ============================================================
# 4. Result Parsing
# ============================================================
def parse_result(raw_output: str) -> str:
    out = raw_output.lower()
    if "anomaly" in out:
        return "Anomaly"
    elif "normal" in out:
        return "Normal"
    return "Unknown"


# ============================================================
# 5. Video Prefetch (Now Just Decoder — Sequential)
# ============================================================
def load_video(video_path: Path, target_fps: int, target_resolution: tuple[int, int]):
    """
    Sequential (non-prefetch) version of video decoding.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        effective_fps = min(target_fps, native_fps) if native_fps > 0 else target_fps
        duration_seconds = frame_count / native_fps if native_fps > 0 else 0
        num_frames_to_sample = max(1, int(duration_seconds * effective_fps))

        total_pixels = num_frames_to_sample * target_resolution[0] * target_resolution[1]
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()

    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(
        [{"role": "user", "content": [{"type": "video", "video": str(video_path), "total_pixels": total_pixels}]}]
    )
    return image_inputs, video_inputs


# ============================================================
# 6. Video Analysis
# ============================================================
def analyze_video(model, processor, video_path: Path, prefetched_data, max_tokens: int, base_text: str):
    image_inputs, video_inputs = prefetched_data

    content = [
        {"type": "video", "video": str(video_path)},
        {"type": "text", "text": base_text},
    ]
    conversation = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    new_tokens = output[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


# ============================================================
# 7. Main — Sequential Video Processing
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="INT8 Inference with bitsandbytes")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="nvidia/Cosmos-Reason1-7B")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=7)
    parser.add_argument("--target_resolution", type=str, default="250x250")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save JSON results (default: <video_dir>/fp8_results.json)")
    args = parser.parse_args()

    width, height = map(int, args.target_resolution.split('x'))
    target_resolution = (width, height)

    video_dir = Path(args.video_dir)
    video_files = sorted([f for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv") for f in video_dir.glob(ext)])
    if not video_files:
        print("No video files found.")
        return

    # Determine output JSON path
    if args.output_json:
        output_json_path = Path(args.output_json)
    else:
        output_json_path = video_dir / "fp8_results.json"

    model, processor = load_model(args.model)
    warmup_model(model, processor)
    base_text = build_cached_prompt(processor)

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
    compute_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)

    print(f"📂 Found {len(video_files)} videos — running INT8 inference (bitsandbytes)\n" + "=" * 50)

    # Track results
    results = []
    total_load_time = 0.0
    total_inference_time = 0.0
    counts = {"Anomaly": 0, "Normal": 0, "Unknown": 0, "Error": 0}

    batch_start_time = time.time()

    for i, video_path in enumerate(video_files, 1):
        load_start = time.time()
        try:
            prefetched_data = load_video(video_path, args.fps, target_resolution)
            load_time = time.time() - load_start
            total_load_time += load_time

            analysis_start = time.time()
            raw = analyze_video(model, processor, video_path, prefetched_data, args.max_tokens, base_text)
            result = parse_result(raw)
            inference_time = time.time() - analysis_start
            total_inference_time += inference_time

            counts[result] += 1

            results.append({
                "file": video_path.name,
                "result": result,
                "raw_output": raw,
                "load_time_s": round(load_time, 3),
                "inference_time_s": round(inference_time, 3),
            })

            print(
                f"[{i}/{len(video_files)}] {video_path.name}: {result} "
                f"(Load: {load_time:.2f}s, INT8 Inference: {inference_time:.2f}s)"
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

    total_time = time.time() - batch_start_time

    # Build output JSON
    output_data = {
        "config": {
            "model": args.model,
            "inference_mode": "bitsandbytes INT8",
            "fps": args.fps,
            "max_tokens": args.max_tokens,
            "target_resolution": args.target_resolution,
            "quantization": "INT8 (bitsandbytes load_in_8bit)",
            "compute_type": "INT8 matrix multiplication",
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
            "avg_inference_time_s": round(total_inference_time / len(video_files), 3) if video_files else 0,
        },
        "results": results,
    }

    # Save JSON
    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("=" * 50)
    print("\nSUMMARY — INT8 Inference (bitsandbytes)")
    print("=" * 50)
    print(f"Total videos: {len(video_files)}")
    print(f"  - Anomaly: {counts['Anomaly']}")
    print(f"  - Normal: {counts['Normal']}")
    print(f"  - Unknown: {counts['Unknown']}")
    print(f"  - Errors: {counts['Error']}")
    print(f"\nTotal load time: {total_load_time:.2f}s")
    print(f"Total INT8 inference time: {total_inference_time:.2f}s")
    print(f"Average INT8 inference time: {total_inference_time / len(video_files):.2f}s per video")
    print(f"\nResults saved to: {output_json_path}")
    print("✅ INT8 inference complete.")


if __name__ == "__main__":
    main()
