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

VIEW_DIRS = ("front_view", "left_view", "right_view", "back_view")
VIEW_LABELS = {
    "front_view": "Front view",
    "left_view": "Left view",
    "right_view": "Right view",
    "back_view": "Rear view",
}

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

    print(f"Model ready in {time.time() - start:.2f}s\n")
    return model, processor


# ============================================================
# 2. Prompt Caching
# ============================================================
def build_cached_prompt(processor):
    # TODO: refine multiview prompt
    base_text = (
        "You are an autonomous driving safety expert. The four videos above show "
        "synchronized front, left, right, and rear views from an ego vehicle.\n\n"
        "Analyze all views for EXTERNAL ANOMALIES that may impact safe AV operation.\n\n"
        "<think>\n"
        "- Obstacles, pedestrians, or vehicles violating rules\n"
        "- Roadwork, blocked lanes, poor visibility, or hazards\n"
        "- Reflections, shadows, or false visual cues confusing perception\n"
        "</think>\n\n"
        "<answer>\n"
        "Is there any external anomaly across any of the four views? Reply with exactly one word of the following:\n"
        "Classification: Anomaly — if any obstacle, obstruction, or unsafe condition is visible.\n"
        "Classification: Normal — if no anomaly or obstruction is visible.\n"
        "</answer>"
    )
    conversation_template = [{"role": "user", "content": [{"type": "text", "text": base_text}]}]
    _ = processor.apply_chat_template(conversation_template, tokenize=False, add_generation_prompt=True)
    return base_text


# ============================================================
# 3. Warmup
# ============================================================
def warmup_model(model, processor):
    print("Warming up model (compiling kernels)...")
    dummy_conv = [{"role": "user", "content": [{"type": "text", "text": "Is this scene safe?"}]}]
    text = processor.apply_chat_template(dummy_conv, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=7)
    torch.cuda.synchronize()
    print("Warmup complete.\n")


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
# 5. Multiview Video Loading
# ============================================================
def _compute_total_pixels(video_path: Path, target_fps: int, target_resolution: tuple[int, int]) -> int:
    cap = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        effective_fps = min(target_fps, native_fps) if native_fps > 0 else target_fps
        duration_seconds = frame_count / native_fps if native_fps > 0 else 0
        num_frames_to_sample = max(1, int(duration_seconds * effective_fps))

        return num_frames_to_sample * target_resolution[0] * target_resolution[1]
    finally:
        if cap is not None and cap.isOpened():
            cap.release()


def load_multiview_videos(
    view_paths: dict[str, Path], target_fps: int, target_resolution: tuple[int, int]
):
    """
    Load 4 synchronized view videos and process them together via qwen_vl_utils.
    view_paths: dict mapping view dir name -> video Path (ordered front, left, right, back).
    Content is interleaved: text label, then video, for each view.
    Returns (image_inputs, video_inputs) covering all views.
    """
    content = []
    for view_dir in VIEW_DIRS:
        vp = view_paths[view_dir]
        total_pixels = _compute_total_pixels(vp, target_fps, target_resolution)
        content.append({"type": "text", "text": f"{VIEW_LABELS[view_dir]}:"})
        content.append({"type": "video", "video": str(vp), "total_pixels": total_pixels})

    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(
        [{"role": "user", "content": content}]
    )
    return image_inputs, video_inputs


# ============================================================
# 6. Multiview Analysis (single conversation, 4 views)
# ============================================================
def analyze_multiview(
    model, processor, view_paths: dict[str, Path],
    prefetched_data: tuple, max_tokens: int, base_text: str,
) -> str:
    """
    Process 4 synchronized view videos as a single conversation.
    Returns the raw model output string.
    """
    image_inputs, video_inputs = prefetched_data

    content = []
    for vd in VIEW_DIRS:
        content.append({"type": "text", "text": f"{VIEW_LABELS[vd]}:"})
        content.append({"type": "video", "video": str(view_paths[vd])})
    content.append({"type": "text", "text": base_text})

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
# 7. Main — Multiview Sequential Processing
# ============================================================
def get_true_label(clip_name: str):
    #TODO: this gets label only from the front view
    stem = Path(clip_name).stem
    if stem.startswith("Anom"):
        return 1
    elif stem.startswith("Norm"):
        return 0
    return None


def discover_clips(video_dir: Path) -> list[str]:
    """
    Discover clip names from front_view/ and verify they exist in all view dirs.
    Returns sorted list of clip filenames present across all 4 views.
    """
    missing_dirs = [vd for vd in VIEW_DIRS if not (video_dir / vd).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"Expected view directories not found: {missing_dirs}")

    front_dir = video_dir / VIEW_DIRS[0]

    clip_names = sorted(
        f.name for f in front_dir.iterdir()
        if f.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")
    )

    valid_clips = []
    for clip in clip_names:
        missing = [vd for vd in VIEW_DIRS if not (video_dir / vd / clip).is_file()] # check if a view is missing for each video
        if missing:
            print(f"  WARNING: {clip} missing in {missing}, skipping.")
        else:
            valid_clips.append(clip)

    return valid_clips


def main():
    parser = argparse.ArgumentParser(description="FP16 Multiview Inference")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Root dir containing front_view/, left_view/, right_view/, back_view/")
    parser.add_argument("--model", type=str, default="nvidia/Cosmos-Reason1-7B")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=3)
    parser.add_argument("--target_resolution", type=str, default="250x250")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save JSON results (default: <video_dir>/fp16_multiview_results.json)")
    args = parser.parse_args()

    width, height = map(int, args.target_resolution.split('x'))
    target_resolution = (width, height)

    video_dir = Path(args.video_dir)
    clip_names = discover_clips(video_dir)
    if not clip_names:
        print("No synchronized clip sets found.")
        return

    output_json_path = Path(args.output_json) if args.output_json else video_dir / "fp16_multiview_results.json"

    model, processor = load_model(args.model)
    warmup_model(model, processor)
    base_text = build_cached_prompt(processor)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
    compute_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)

    print(
        f"Found {len(clip_names)} synchronized clip sets (4 views each) "
        f"— running FP16 multiview inference\n" + "=" * 50
    )

    results = []
    total_load_time = 0.0
    total_inference_time = 0.0
    counts = {"Anomaly": 0, "Normal": 0, "Unknown": 0, "Error": 0}
    metrics = Metrics()
    overall_start = time.time()

    for i, clip_name in enumerate(clip_names, 1):
        view_paths = {vd: video_dir / vd / clip_name for vd in VIEW_DIRS}

        try:
            load_start = time.time()
            prefetched = load_multiview_videos(view_paths, args.fps, target_resolution)
            load_time = time.time() - load_start
            total_load_time += load_time

            inference_start = time.time()
            raw = analyze_multiview(
                model, processor, view_paths, prefetched, args.max_tokens, base_text
            )
            inference_time = time.time() - inference_start
            total_inference_time += inference_time

            result = parse_result(raw)
            counts[result] += 1

            true_label = get_true_label(clip_name) #TODO: this gets label only from the front view
            if (true_label is not None) and (result != "Unknown"):
                pred_label = 1 if result == "Anomaly" else 0
                metrics.update([pred_label], [true_label], [inference_time])

            results.append({
                "clip": clip_name,
                "result": result,
                "raw_output": raw,
                "true_label": "Anomaly" if true_label == 1 else ("Normal" if true_label == 0 else "Unknown"),
                "load_time_s": round(load_time, 3),
                "inference_time_s": round(inference_time, 3),
            })

            print(
                f"[{i}/{len(clip_names)}] {clip_name}: {result} "
                f"(Load: {load_time:.2f}s, Inference: {inference_time:.2f}s) "
                f"(raw: {raw!r})"
            )

        except Exception as e:
            counts["Error"] += 1
            results.append({
                "clip": clip_name,
                "result": "Error",
                "raw_output": str(e),
                "true_label": "Unknown",
                "load_time_s": 0.0,
                "inference_time_s": 0.0,
            })
            print(f"[{i}/{len(clip_names)}] {clip_name}: ERROR - {e}")

    total_time = time.time() - overall_start

    output_data = {
        "config": {
            "model": args.model,
            "inference_mode": "FP16 multiview",
            "views": len(VIEW_DIRS),
            "view_dirs": list(VIEW_DIRS),
            "fps": args.fps,
            "max_tokens": args.max_tokens,
            "target_resolution": args.target_resolution,
            "compute_type": "bfloat16",
            "gpu": gpu_name,
            "compute_capability": f"{compute_cap[0]}.{compute_cap[1]}",
        },
        "summary": {
            "total_clips": len(clip_names),
            "anomalies": counts["Anomaly"],
            "normals": counts["Normal"],
            "unknowns": counts["Unknown"],
            "errors": counts["Error"],
            "total_load_time_s": round(total_load_time, 3),
            "total_inference_time_s": round(total_inference_time, 3),
            "total_time_s": round(total_time, 3),
            "avg_inference_time_s": round(total_inference_time / max(len(clip_names), 1), 3),
        },
        "metrics": metrics.compute() if metrics.count > 0 else None,
        "results": results,
    }

    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print("=" * 50)
    print(f"\nSUMMARY — FP16 Multiview Inference ({len(VIEW_DIRS)} views)")
    print("=" * 50)
    print(f"Total clips: {len(clip_names)}")
    print(f"  - Anomaly: {counts['Anomaly']}")
    print(f"  - Normal: {counts['Normal']}")
    print(f"  - Unknown: {counts['Unknown']}")
    print(f"  - Errors: {counts['Error']}")
    print(f"\nTotal load time: {total_load_time:.2f}s")
    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Average inference time: {total_inference_time / max(len(clip_names), 1):.2f}s per clip")

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
    print("FP16 multiview inference complete.")


if __name__ == "__main__":
    main()
