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

from metrics import Metrics

warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'

# ============================================================
# 1. Model Loader
# ============================================================
def load_model(model_name: str):
    print("🔧 Loading and compiling model... This may take a few seconds.")
    start = time.time()

    # bnb_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        # quantization_config=bnb_config,
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
        "Classification: Anomaly — if any obstacle, obstruction, or unsafe condition is visible.\n" # TODO
        "Classification: Normal — if no anomaly or obstruction is visible.\n" # TODO
        "</answer>"
    )
    # TODO apply_chat_template is unused here
    conversation_template = [{"role": "user", "content": [{"type": "text", "text": base_text}]}]
    _ = processor.apply_chat_template(conversation_template, tokenize=False, add_generation_prompt=True)
    return base_text


# ============================================================
# 3. Warmup
# ============================================================
def warmup_model(model, processor, batch_size: int = 1):
    print(f"🔥 Warming up model (compiling kernels, batch_size={batch_size})...")
    dummy_conv = [{"role": "user", "content": [{"type": "text", "text": "Is this scene safe?"}]}]
    text = processor.apply_chat_template(dummy_conv, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text] * batch_size, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=7)
    torch.cuda.synchronize()
    print("✅ Warmup complete.\n")


# ============================================================
# 4. Result Parsing
# ============================================================
def parse_result(raw_output: str) -> str:
    out = raw_output.lower()
    if "classification: an" in out: # 3 tokens
        return "Anomaly"
    elif "normal" in out: # 1 token
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

    # TODO: We should be able to batch this
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(
        [{"role": "user", "content": [{"type": "video", "video": str(video_path), "total_pixels": total_pixels}]}]
    )
    return image_inputs, video_inputs


# ============================================================
# 6. Batched Video Analysis
# ============================================================
def analyze_video_batch(
    model, processor, batch_items: list[tuple[Path, tuple]], max_tokens: int, base_text: str
) -> list[str]:
    """
    Process a batch of videos in a single forward pass.
    batch_items: list of (video_path, (image_inputs, video_inputs)) tuples.
    Returns a list of raw output strings, one per video.
    """
    texts = []
    all_image_inputs = []
    all_video_inputs = []

    for video_path, (image_inputs, video_inputs) in batch_items:
        content = [
            {"type": "video", "video": str(video_path)},
            {"type": "text", "text": base_text},
        ]
        conversation = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        texts.append(text)

        if image_inputs is not None:
            all_image_inputs.extend(image_inputs)
        if video_inputs is not None:
            all_video_inputs.extend(video_inputs)

    inputs = processor(
        text=texts,
        images=all_image_inputs or None,
        videos=all_video_inputs or None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    new_tokens = output[:, inputs.input_ids.shape[1]:]
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)
    return [d.strip() for d in decoded]


# ============================================================
# 7. Main — Batched Video Processing
# ============================================================
def get_true_label(video_path: Path):
    fname = video_path.stem
    if fname.startswith("Anom"):
        return 1
    elif fname.startswith("Norm"):
        return 0
    return None


def main():
    parser = argparse.ArgumentParser(description="FP16 Batched Inference")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="nvidia/Cosmos-Reason1-7B")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of videos to process per batch (increase for GPU utilization, decrease if OOM)")
    parser.add_argument("--target_resolution", type=str, default="250x250")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save JSON results (default: <video_dir>/fp16_batch_results.json)")
    args = parser.parse_args()

    width, height = map(int, args.target_resolution.split('x'))
    target_resolution = (width, height)

    video_dir = Path(args.video_dir)
    video_files = sorted([f for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv") for f in video_dir.glob(ext)])
    if not video_files:
        print("No video files found.")
        return

    if args.output_json:
        output_json_path = Path(args.output_json)
    else:
        output_json_path = video_dir / "fp16_batch_results.json"

    model, processor = load_model(args.model)
    warmup_model(model, processor, batch_size=args.batch_size)
    base_text = build_cached_prompt(processor)

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
    compute_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)

    print(
        f"📂 Found {len(video_files)} videos — running FP16 batched inference "
        f"(batch_size={args.batch_size})\n" + "=" * 50
    )

    # Track results
    results = []
    total_load_time = 0.0
    total_inference_time = 0.0
    counts = {"Anomaly": 0, "Normal": 0, "Unknown": 0, "Error": 0}
    metrics = Metrics()
    overall_start = time.time()

    for batch_start_idx in range(0, len(video_files), args.batch_size):
        batch_paths = video_files[batch_start_idx : batch_start_idx + args.batch_size]
        batch_num = batch_start_idx // args.batch_size + 1
        global_offset = batch_start_idx + 1

        # --- Load videos for this batch ---
        batch_items = []
        batch_load_errors = {}
        load_start = time.time()
        for vp in batch_paths:
            try:
                prefetched = load_video(vp, args.fps, target_resolution)
                batch_items.append((vp, prefetched))
            except Exception as e:
                batch_load_errors[vp] = e
        batch_load_time = time.time() - load_start
        total_load_time += batch_load_time

        # Record errors from loading
        for vp, err in batch_load_errors.items():
            counts["Error"] += 1
            results.append({
                "file": vp.name,
                "result": "Error",
                "raw_output": str(err),
                "true_label": "Unknown",
                "load_time_s": 0.0,
                "inference_time_s": 0.0,
            })
            print(f"  {vp.name}: LOAD ERROR - {err}")

        if not batch_items:
            continue

        # --- Run batched inference ---
        inference_start = time.time()
        try:
            raw_outputs = analyze_video_batch(
                model, processor, batch_items, args.max_tokens, base_text
            )
        except Exception as e:
            # If the whole batch fails, record errors for all items
            for vp, _ in batch_items:
                counts["Error"] += 1
                results.append({
                    "file": vp.name,
                    "result": "Error",
                    "raw_output": str(e),
                    "true_label": "Unknown",
                    "load_time_s": round(batch_load_time / len(batch_items), 3),
                    "inference_time_s": 0.0,
                })
            print(f"  Batch {batch_num} INFERENCE ERROR - {e}")
            continue
        batch_inference_time = time.time() - inference_start
        total_inference_time += batch_inference_time
        per_video_inference = batch_inference_time / len(batch_items)

        # --- Process results ---
        for idx, ((vp, _), raw) in enumerate(zip(batch_items, raw_outputs)):
            true_label = get_true_label(vp)
            result = parse_result(raw)
            counts[result] += 1

            pred_label = 1 if result == "Anomaly" else 0
            if true_label is not None:
                metrics.update([pred_label], [true_label], [per_video_inference])

            results.append({
                "file": vp.name,
                "result": result,
                "raw_output": raw,
                "true_label": "Anomaly" if true_label == 1 else ("Normal" if true_label == 0 else "Unknown"),
                "load_time_s": round(batch_load_time / len(batch_items), 3),
                "inference_time_s": round(per_video_inference, 3),
            })

            global_idx = global_offset + idx
            print(
                f"[{global_idx}/{len(video_files)}] {vp.name}: {result} "
                f"(raw: {raw!r})"
            )

        print(
            f"  Batch {batch_num} ({len(batch_items)} videos): "
            f"Load {batch_load_time:.2f}s, Inference {batch_inference_time:.2f}s "
            f"({per_video_inference:.2f}s/video)"
        )

    total_time = time.time() - overall_start

    output_data = {
        "config": {
            "model": args.model,
            "inference_mode": "FP16 batched",
            "batch_size": args.batch_size,
            "fps": args.fps,
            "max_tokens": args.max_tokens,
            "target_resolution": args.target_resolution,
            "compute_type": "bfloat16",
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
            "avg_inference_time_s": round(total_inference_time / max(len(video_files), 1), 3),
        },
        "metrics": metrics.compute() if metrics.count > 0 else None,
        "results": results,
    }

    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print("=" * 50)
    print(f"\nSUMMARY — FP16 Batched Inference (batch_size={args.batch_size})")
    print("=" * 50)
    print(f"Total videos: {len(video_files)}")
    print(f"  - Anomaly: {counts['Anomaly']}")
    print(f"  - Normal: {counts['Normal']}")
    print(f"  - Unknown: {counts['Unknown']}")
    print(f"  - Errors: {counts['Error']}")
    print(f"\nTotal load time: {total_load_time:.2f}s")
    print(f"Total FP16 inference time: {total_inference_time:.2f}s")
    print(f"Average inference time: {total_inference_time / max(len(video_files), 1):.2f}s per video")

    if metrics.count > 0:
        m = metrics.compute()
        print(f"\nCLASSIFICATION METRICS")
        print("=" * 50)
        print(f"  TP: {m['TP']}  TN: {m['TN']}  FP: {m['FP']}  FN: {m['FN']}")
        print(f"  Accuracy:  {m['Accuracy']:.4f}")
        print(f"  Precision: {m['Precision']:.4f}")
        print(f"  Recall:    {m['Recall']:.4f}")
        print(f"  F1-Score:  {m['F1-Score']:.4f}")
        print(f"  Avg Inference Time: {m['Avg Inference Time']:.3f}s")

    print(f"\nResults saved to: {output_json_path}")
    print("FP16 batched inference complete.")


if __name__ == "__main__":
    main()
