#!/usr/bin/env python3

import argparse
import json
import time
import warnings
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
    LlavaNextVideoVideoProcessor,
)

from metrics import Metrics

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 1. Model Loader
# ============================================================
def load_model(model_name: str):
    print("Loading LLaVA-Video model... This may take a few minutes.")
    start = time.time()

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()

    from transformers import LlavaNextVideoConfig
    model_config = LlavaNextVideoConfig.from_pretrained(model_name)
    patch_size = model_config.vision_config.patch_size
    video_size = model_config.vision_config.image_size
    vision_feature_select_strategy = getattr(model_config, "vision_feature_select_strategy", "default")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    video_processor = LlavaNextVideoVideoProcessor(
        size={"shortest_edge": video_size},
        crop_size={"height": video_size, "width": video_size},
    )

    processor = LlavaNextVideoProcessor(
        video_processor=video_processor,
        image_processor=image_processor,
        tokenizer=tokenizer,
        patch_size=patch_size,
        vision_feature_select_strategy=vision_feature_select_strategy,
    )

    try:
        import json as _json
        from huggingface_hub import hf_hub_download
        chat_tpl_path = hf_hub_download(model_name, "chat_template.json")
        with open(chat_tpl_path) as f:
            chat_tpl = _json.load(f)["chat_template"]
        processor.chat_template = chat_tpl
    except Exception:
        pass

    torch.set_float32_matmul_precision("high")

    print(f"Model ready in {time.time() - start:.2f}s\n")
    return model, processor


# ============================================================
# 2. Prompt Builder (identical to Cosmos pipeline)
# ============================================================
def build_prompt():
    base_text = (
        "You are an autonomous driving safety expert analyzing this video for EXTERNAL ANOMALIES that may impact safe AV operation:\n\n"
        "- Obstacles, pedestrians, or vehicles violating rules\n"
        "- Roadwork, blocked lanes, poor visibility, or hazards\n"
        "- Reflections, shadows, or false visual cues confusing perception\n"

        "Describe what you see in this dashcam video. Is there any external anomaly in this video? Reply with:\n"
        "Classification: Anomaly (if any obstacle, obstruction, or unsafe condition is visible.)\n"
        "Classification: Normal (if no anomaly or obstruction is visible.)\n"
    )
    return base_text


# ============================================================
# 3. Warmup
# ============================================================
def warmup_model(model, processor):
    print("Warming up model...")
    dummy_conv = [{"role": "user", "content": [{"type": "text", "text": "Is this scene safe?"}]}]
    text = processor.apply_chat_template(dummy_conv, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=7)
    torch.cuda.synchronize()
    print("Warmup complete.\n")


# ============================================================
# 4. Result Parsing (identical to Cosmos pipeline)
# ============================================================
def parse_result(raw_output: str) -> str:
    print(raw_output)
    out = raw_output.lower()
    if "classification: an" in out or "classification: anomaly" in out:
        return "Anomaly"
    elif "classification: normal" in out:
        return "Normal"
    elif "anomaly" in out:
        return "Anomaly"
    elif "normal" in out:
        return "Normal"
    return "Unknown"


# ============================================================
# 5. Video Frame Extraction via PyAV
# ============================================================
def read_video_pyav(container, indices):
    """Decode specific frame indices from a PyAV container."""
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def load_video(video_path: Path, target_fps: int, target_resolution: tuple[int, int]):
    """
    Load video and sample frames at target_fps, matching the Cosmos pipeline's
    frame sampling strategy (effective_fps * duration frames, uniformly sampled).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    effective_fps = min(target_fps, native_fps) if native_fps > 0 else target_fps
    duration_seconds = frame_count / native_fps if native_fps > 0 else 0
    num_frames_to_sample = max(1, int(duration_seconds * effective_fps))

    container = av.open(str(video_path))
    total_frames = container.streams.video[0].frames
    if total_frames <= 0:
        total_frames = frame_count

    indices = np.arange(0, total_frames, max(1, total_frames / num_frames_to_sample)).astype(int)
    indices = indices[:num_frames_to_sample]

    clip = read_video_pyav(container, indices)
    container.close()
    return clip


# ============================================================
# 6. Video Analysis
# ============================================================
def analyze_video(model, processor, clip, max_tokens: int, base_text: str):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": base_text},
                {"type": "video"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=prompt,
        videos=clip,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    new_tokens = output[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


# ============================================================
# 7. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="LLaVA-NeXT-Video-7B Inference for Hazard Perception")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="llava-hf/LLaVA-NeXT-Video-7B-hf")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=150)
    parser.add_argument("--target_resolution", type=str, default="250x250")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save JSON results (default: <video_dir>/llava_video_results.json)")
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
        output_json_path = video_dir / "llava_video_results.json"

    model, processor = load_model(args.model)
    warmup_model(model, processor)
    base_text = build_prompt()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
    compute_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)

    print(f"Found {len(video_files)} videos — running LLaVA-Video FP16 inference\n" + "=" * 60)

    results = []
    total_load_time = 0.0
    total_inference_time = 0.0
    counts = {"Anomaly": 0, "Normal": 0, "Unknown": 0, "Error": 0}
    metrics = Metrics()

    batch_start_time = time.time()

    for i, video_path in enumerate(video_files, 1):
        fname = video_path.stem
        if fname.startswith("Anom"):
            true_label = 1
        elif fname.startswith("Norm"):
            true_label = 0
        else:
            true_label = None

        load_start = time.time()
        try:
            clip = load_video(video_path, args.fps, target_resolution)
            load_time = time.time() - load_start
            total_load_time += load_time

            analysis_start = time.time()
            raw = analyze_video(model, processor, clip, args.max_tokens, base_text)
            inference_time = time.time() - analysis_start
            result = parse_result(raw)
            total_inference_time += inference_time

            counts[result] += 1

            pred_label = 1 if result == "Anomaly" else 0
            if true_label is not None:
                metrics.update([pred_label], [true_label], [inference_time])

            results.append({
                "file": video_path.name,
                "result": result,
                "raw_output": raw,
                "true_label": "Anomaly" if true_label == 1 else ("Normal" if true_label == 0 else "Unknown"),
                "load_time_s": round(load_time, 3),
                "inference_time_s": round(inference_time, 3),
            })

            print(
                f"[{i}/{len(video_files)}] {video_path.name}: {result} "
                f"(Load: {load_time:.2f}s, LLaVA Inference: {inference_time:.2f}s)"
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

    output_data = {
        "config": {
            "model": args.model,
            "inference_mode": "LLaVA-Video FP16",
            "fps": args.fps,
            "max_tokens": args.max_tokens,
            "target_resolution": args.target_resolution,
            "quantization": "None (FP16)",
            "compute_type": "FP16",
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
        "metrics": metrics.compute() if metrics.count > 0 else None,
        "results": results,
    }

    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print("=" * 60)
    print("\nSUMMARY — LLaVA-Video FP16 Inference")
    print("=" * 60)
    print(f"Model:        {args.model}")
    print(f"GPU:          {gpu_name}")
    print(f"Total videos: {len(video_files)}")
    print(f"  - Anomaly:  {counts['Anomaly']}")
    print(f"  - Normal:   {counts['Normal']}")
    print(f"  - Unknown:  {counts['Unknown']}")
    print(f"  - Errors:   {counts['Error']}")
    print(f"\nTotal load time:               {total_load_time:.2f}s")
    print(f"Total LLaVA inference time:    {total_inference_time:.2f}s")
    print(f"Average LLaVA inference time:  {total_inference_time / len(video_files):.2f}s per video")
    print(f"Total wall-clock time:         {total_time:.2f}s")

    if metrics.count > 0:
        m = metrics.compute()
        print(f"\nCLASSIFICATION METRICS")
        print("=" * 60)
        print(f"  TP: {m['TP']}  TN: {m['TN']}  FP: {m['FP']}  FN: {m['FN']}")
        print(f"  Accuracy:  {m['Accuracy']:.4f}")
        print(f"  Precision: {m['Precision']:.4f}")
        print(f"  Recall:    {m['Recall']:.4f}")
        print(f"  F1-Score:  {m['F1-Score']:.4f}")
        print(f"  Avg Inference Time: {m['Avg Inference Time']:.3f}s")

    print(f"\nResults saved to: {output_json_path}")
    print("LLaVA-Video inference complete.")


if __name__ == "__main__":
    main()
