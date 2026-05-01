#!/usr/bin/env python3

import argparse
import asyncio
import json
import numpy as np
import os
import re
import subprocess
import time
import warnings
from pathlib import Path

os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"

import torch
import transformers
import qwen_vl_utils
import cv2

from fastmcp import Client as MCPClient

from metrics import Metrics

warnings.filterwarnings("ignore", category=UserWarning)

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
# 4b. Stage-2 MRM selector (runs only on Anomaly clips)
# ============================================================
MRM_CHOICES = ("EMERGENCY_STOP", "COMFORTABLE_STOP", "PULL_OVER")

MRM_TO_TOOL = {
    "EMERGENCY_STOP":   "emergency_stop",
    "COMFORTABLE_STOP": "comfortable_stop",
    "PULL_OVER":        "pull_over",
}

_MRM_RE = re.compile(r"MRM\s*[:\-]?\s*(EMERGENCY_STOP|COMFORTABLE_STOP|PULL_OVER)",
                     re.IGNORECASE)


def build_mrm_prompt() -> str:
    return (
        "An anomaly has been detected in the synchronized multiview footage above.\n"
        "Choose the single most appropriate Minimal Risk Maneuver (MRM) for the ego vehicle.\n\n"
        "<think>\n"
        "- EMERGENCY_STOP: imminent collision risk, hard brake required now.\n"
        "- COMFORTABLE_STOP: hazard visible ahead but it is safe to decelerate smoothly.\n"
        "- PULL_OVER: persistent hazard or degraded driving condition; ego should leave the travel lane and stop.\n"
        "</think>\n\n"
        "<answer>\n"
        "Reply with exactly one line of the form:\n"
        "MRM: EMERGENCY_STOP\n"
        "MRM: COMFORTABLE_STOP\n"
        "MRM: PULL_OVER\n"
        "</answer>"
    )


def parse_mrm(raw_output: str) -> str | None:
    """Extract the MRM choice from stage-2 output. Returns None if unrecognized."""
    m = _MRM_RE.search(raw_output or "")
    if m:
        return m.group(1).upper()
    upper = (raw_output or "").upper()
    for choice in MRM_CHOICES:
        if choice in upper:
            return choice
    return None


def analyze_mrm(
    model, processor, view_paths: dict[str, Path],
    prefetched_data: tuple, max_tokens: int, mrm_prompt: str,
) -> str:
    """Stage-2 inference: reuse the already-decoded video tensors from stage 1
    and ask the VLM to pick one of the three MRMs.

    Returns the raw model output string.
    """
    image_inputs, video_inputs = prefetched_data

    content = []
    for vd in VIEW_DIRS:
        content.append({"type": "text", "text": f"{VIEW_LABELS[vd]}:"})
        content.append({"type": "video", "video": str(view_paths[vd])})
    content.append({"type": "text", "text": mrm_prompt})

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
# 4c. MCP client helper (stdio to zenoh_autoware_mcp_server.py)
# ============================================================
async def call_mcp_tool(client: MCPClient, tool_name: str, args: dict) -> str:
    """Invoke a tool on the MCP server and return the textual content."""
    result = await client.call_tool(tool_name, args)
    if result.content:
        return result.content[0].text
    return "{}"


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
def get_true_label(view_paths: dict[str, Path]):
    """
    Determine ground-truth label from filenames across all 4 views.
    Anomaly if ANY view file starts with 'Anom_', Normal only if ALL are 'Norm_'.
    Returns None if no label prefix is found on any view.
    """
    has_anom = False
    has_label = False
    for vp in view_paths.values():
        stem = vp.stem
        if stem.startswith("Anom"):
            has_anom = True
            has_label = True
        elif stem.startswith("Norm"):
            has_label = True

    if not has_label:
        return None
    return 1 if has_anom else 0


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def _clip_numeric_id(filename: str) -> str:
    """Extract the numeric ID suffix from e.g. 'Anom_0001.mp4' -> '0001'."""
    return Path(filename).stem.rsplit("_", 1)[-1]


def discover_clips(video_dir: Path) -> list[tuple[str, dict[str, Path]]]:
    """
    Match clips across all 4 view dirs by their numeric ID (the four-digit
    suffix), regardless of Anom_/Norm_ label prefix.

    Returns a sorted list of (numeric_id, {view_dir: full_path}) tuples
    for IDs that are present in every view.
    """
    missing_dirs = [vd for vd in VIEW_DIRS if not (video_dir / vd).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"Expected view directories not found: {missing_dirs}")

    id_to_paths: dict[str, dict[str, Path]] = {}
    for vd in VIEW_DIRS:
        vd_path = video_dir / vd
        for f in vd_path.iterdir():
            if f.suffix.lower() in VIDEO_EXTS:
                nid = _clip_numeric_id(f.name)
                id_to_paths.setdefault(nid, {})[vd] = f

    valid = []
    for nid in sorted(id_to_paths):
        paths = id_to_paths[nid]
        missing = [vd for vd in VIEW_DIRS if vd not in paths]
        if missing:
            print(f"  WARNING: clip ID {nid} missing in {missing}, skipping.")
        else:
            valid.append((nid, paths))

    return valid


# ============================================================
# 8. Collage Video Generation
# ============================================================

COLLAGE_LAYOUT = [
    ("left_view",  0, 0),
    ("front_view", 1, 0),
    ("right_view", 0, 1),
    ("back_view",  1, 1),
]

HEADER_HEIGHT = 60
GAP_H = 10
GAP_V = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_ANOMALY = (0, 0, 255)
COLOR_NORMAL = (0, 200, 0)
COLOR_UNKNOWN = (180, 180, 180)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


def _label_color(label: str) -> tuple[int, int, int]:
    if label == "Anomaly":
        return COLOR_ANOMALY
    if label == "Normal":
        return COLOR_NORMAL
    return COLOR_UNKNOWN


def _view_gt_label(view_path: Path) -> str:
    return "Anomaly" if view_path.stem.startswith("Anom") else "Normal"


def _read_video_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def generate_collage_video(
    clips: list[tuple[str, dict[str, Path]]],
    results: list[dict],
    output_path: Path,
    native_fps: int,
    window_sec: int,
    step_sec: int,
):
    """
    Build a single 2x2 collage video from the sliding-window clip set.

    For the first clip all frames are written; for subsequent clips only the
    last `step_sec` seconds of new frames are appended, producing one
    continuous output video with per-window label overlays.
    """
    if not clips:
        return

    step_frames = native_fps * step_sec
    window_frames = native_fps * window_sec
    overlap_frames = window_frames - step_frames

    sample_path = next(iter(clips[0][1].values()))
    sample = _read_video_frames(sample_path)
    if not sample:
        print("WARNING: could not read sample clip for collage; skipping video output.")
        return
    clip_h, clip_w = sample[0].shape[:2]

    collage_w = 2 * clip_w + GAP_H
    collage_h = HEADER_HEIGHT + 2 * clip_h + GAP_V

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{collage_w}x{collage_h}",
        "-r", str(native_fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-loglevel", "error",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    print(f"\nGenerating collage video ({collage_w}x{collage_h} @ {native_fps}fps) ...")

    for idx, ((clip_id, view_paths), result_entry) in enumerate(zip(clips, results)):
        gt_int = get_true_label(view_paths)
        gt_label = "Anomaly" if gt_int == 1 else ("Normal" if gt_int == 0 else "Unknown")
        pred_label = result_entry["result"]
        mrm_choice = result_entry.get("mrm_choice")

        per_view_gt = {vd: _view_gt_label(vp) for vd, vp in view_paths.items()}

        all_view_frames: dict[str, list[np.ndarray]] = {}
        for vd in VIEW_DIRS:
            all_view_frames[vd] = _read_video_frames(view_paths[vd])

        n_frames = min(len(f) for f in all_view_frames.values())
        if idx == 0:
            frame_start = 0
        else:
            frame_start = overlap_frames

        for fi in range(frame_start, n_frames):
            canvas = np.zeros((collage_h, collage_w, 3), dtype=np.uint8)

            header_text = f"GT: {gt_label}  |  Pred: {pred_label}"
            if mrm_choice:
                header_text += f"  |  MRM: {mrm_choice}"
            text_size = cv2.getTextSize(header_text, FONT, 0.9, 2)[0]
            tx = (collage_w - text_size[0]) // 2
            ty = (HEADER_HEIGHT + text_size[1]) // 2
            cv2.putText(canvas, "GT: ", (tx, ty), FONT, 0.9, COLOR_WHITE, 2, cv2.LINE_AA)
            gt_offset = cv2.getTextSize("GT: ", FONT, 0.9, 2)[0][0]
            cv2.putText(canvas, gt_label, (tx + gt_offset, ty), FONT, 0.9,
                        _label_color(gt_label), 2, cv2.LINE_AA)
            sep_offset = gt_offset + cv2.getTextSize(gt_label, FONT, 0.9, 2)[0][0]
            cv2.putText(canvas, "  |  ", (tx + sep_offset, ty), FONT, 0.9,
                        COLOR_WHITE, 2, cv2.LINE_AA)
            pred_prefix_offset = sep_offset + cv2.getTextSize("  |  ", FONT, 0.9, 2)[0][0]
            cv2.putText(canvas, "Pred: ", (tx + pred_prefix_offset, ty), FONT, 0.9,
                        COLOR_WHITE, 2, cv2.LINE_AA)
            pred_text_offset = pred_prefix_offset + cv2.getTextSize("Pred: ", FONT, 0.9, 2)[0][0]
            cv2.putText(canvas, pred_label, (tx + pred_text_offset, ty), FONT, 0.9,
                        _label_color(pred_label), 2, cv2.LINE_AA)

            if mrm_choice:
                mrm_text = f"  |  MRM: {mrm_choice}"
                mrm_offset = pred_text_offset + cv2.getTextSize(pred_label, FONT, 0.9, 2)[0][0]
                cv2.putText(canvas, mrm_text, (tx + mrm_offset, ty), FONT, 0.9,
                            COLOR_ANOMALY, 2, cv2.LINE_AA)

            for vd, col, row in COLLAGE_LAYOUT:
                x0 = col * (clip_w + GAP_H)
                y0 = HEADER_HEIGHT + row * (clip_h + GAP_V)

                if fi < len(all_view_frames[vd]):
                    frame = all_view_frames[vd][fi]
                    canvas[y0:y0 + clip_h, x0:x0 + clip_w] = frame

                view_label = f"{VIEW_LABELS[vd]} - GT: {per_view_gt[vd]}"
                cv2.putText(canvas, view_label, (x0 + 10, y0 + 30), FONT, 0.7,
                            _label_color(per_view_gt[vd]), 2, cv2.LINE_AA)

            proc.stdin.write(canvas.tobytes())

        if (idx + 1) % 10 == 0 or idx == len(clips) - 1:
            print(f"  Collage progress: {idx + 1}/{len(clips)} clips processed")

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode} for {output_path}")

    print(f"Collage video saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="FP16 Multiview Inference")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Root dir containing front_view/, left_view/, right_view/, back_view/")
    parser.add_argument("--model", type=str, default="nvidia/Cosmos-Reason1-7B")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=3)
    parser.add_argument("--target_resolution", type=str, default="250x250")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save JSON results (default: <video_dir>/fp16_multiview_results.json)")
    parser.add_argument("--output_video", type=str, default=None,
                        help="Path for collage output video (default: <video_dir>/collage_output.mp4)")
    parser.add_argument("--native_fps", type=int, default=20,
                        help="Native FPS of the source clip videos (for collage output)")
    parser.add_argument("--window_sec", type=int, default=5,
                        help="Duration of each sliding-window clip in seconds")
    parser.add_argument("--step_sec", type=int, default=2,
                        help="Sliding window step size in seconds")
    parser.add_argument("--mcp_server", type=str, default=None,
                        help="Path to the MCP server script (default: "
                             "zenoh_autoware_mcp_server.py next to this file)")
    parser.add_argument("--no_dispatch", action="store_true",
                        help="Run stage-2 MRM selection but skip MCP calls "
                             "(offline eval; no Zenoh/Autoware required)")
    parser.add_argument("--mrm_max_tokens", type=int, default=16,
                        help="Max generation tokens for stage-2 MRM selection")
    parser.add_argument("--mcp_timeout_s", type=float, default=2.0,
                        help="Per-call MCP timeout in seconds")
    args = parser.parse_args()

    width, height = map(int, args.target_resolution.split('x'))
    target_resolution = (width, height)

    video_dir = Path(args.video_dir)
    clips = discover_clips(video_dir)
    if not clips:
        print("No synchronized clip sets found.")
        return

    output_json_path = Path(args.output_json) if args.output_json else video_dir / "fp16_multiview_results.json"

    model, processor = load_model(args.model)
    warmup_model(model, processor)
    base_text = build_cached_prompt(processor)
    mrm_prompt = build_mrm_prompt()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
    compute_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)

    server_script = args.mcp_server or str(Path(__file__).parent / "zenoh_autoware_mcp_server.py")
    if args.no_dispatch:
        mcp_client: MCPClient | None = None
        print("MCP dispatch disabled (--no_dispatch); stage-2 MRM selection will still run.")
    else:
        mcp_client = MCPClient(["python3", server_script])
        print(f"MCP client will spawn: {server_script}")

    print(
        f"Found {len(clips)} synchronized clip sets (4 views each) "
        f"— running FP16 multiview inference\n" + "=" * 50
    )

    results = []
    total_load_time = 0.0
    total_inference_time = 0.0
    total_stage2_time = 0.0
    counts = {"Anomaly": 0, "Normal": 0, "Unknown": 0, "Error": 0}
    mrm_counts = {k: 0 for k in MRM_CHOICES}
    mcp_failures = 0
    metrics = Metrics()
    overall_start = time.time()

    async def run_clips() -> None:
        nonlocal total_load_time, total_inference_time, total_stage2_time, mcp_failures

        for i, (clip_id, view_paths) in enumerate(clips, 1):
            clip_name = f"clip_{clip_id}"

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

                true_label = get_true_label(view_paths)
                if (true_label is not None) and (result != "Unknown"):
                    pred_label = 1 if result == "Anomaly" else 0
                    metrics.update([pred_label], [true_label], [inference_time])

                mrm_choice: str | None = None
                mrm_raw: str | None = None
                mcp_resp: str | None = None
                stage2_time = 0.0

                if result == "Anomaly":
                    stage2_start = time.time()
                    mrm_raw = analyze_mrm(
                        model, processor, view_paths, prefetched,
                        args.mrm_max_tokens, mrm_prompt,
                    )
                    stage2_time = time.time() - stage2_start
                    total_stage2_time += stage2_time

                    parsed = parse_mrm(mrm_raw)
                    mrm_choice = parsed or "EMERGENCY_STOP"
                    if parsed is None:
                        print(f"  [warn] MRM parser could not match {mrm_raw!r}; "
                              f"falling back to EMERGENCY_STOP")
                    mrm_counts[mrm_choice] += 1

                    tool_name = MRM_TO_TOOL[mrm_choice]
                    if mcp_client is not None:
                        try:
                            mcp_resp = await asyncio.wait_for(
                                call_mcp_tool(mcp_client, tool_name, {}),
                                timeout=args.mcp_timeout_s,
                            )
                        except asyncio.TimeoutError:
                            mcp_failures += 1
                            mcp_resp = f"TIMEOUT after {args.mcp_timeout_s:.1f}s"
                        except Exception as e:
                            mcp_failures += 1
                            mcp_resp = f"ERROR: {e}"
                    else:
                        mcp_resp = "skipped (--no_dispatch)"

                results.append({
                    "clip": clip_name,
                    "result": result,
                    "raw_output": raw,
                    "true_label": "Anomaly" if true_label == 1 else ("Normal" if true_label == 0 else "Unknown"),
                    "load_time_s": round(load_time, 3),
                    "inference_time_s": round(inference_time, 3),
                    "mrm_choice": mrm_choice,
                    "mrm_raw": mrm_raw,
                    "mrm_mcp_response": mcp_resp,
                    "stage2_inference_time_s": round(stage2_time, 3),
                })

                extra = ""
                if mrm_choice:
                    extra = f" MRM={mrm_choice} ({stage2_time:.2f}s)"
                    if mcp_resp is not None:
                        extra += f" mcp={mcp_resp[:80]!r}"

                print(
                    f"[{i}/{len(clips)}] {clip_name}: {result} "
                    f"(Load: {load_time:.2f}s, Inference: {inference_time:.2f}s) "
                    f"(raw: {raw!r}){extra}"
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
                    "mrm_choice": None,
                    "mrm_raw": None,
                    "mrm_mcp_response": None,
                    "stage2_inference_time_s": 0.0,
                })
                print(f"[{i}/{len(clips)}] {clip_name}: ERROR - {e}")

    if mcp_client is not None:
        async with mcp_client:
            await run_clips()
    else:
        await run_clips()

    total_time = time.time() - overall_start

    output_data = {
        "config": {
            "model": args.model,
            "inference_mode": "FP16 multiview + MRM dispatch",
            "views": len(VIEW_DIRS),
            "view_dirs": list(VIEW_DIRS),
            "fps": args.fps,
            "max_tokens": args.max_tokens,
            "mrm_max_tokens": args.mrm_max_tokens,
            "target_resolution": args.target_resolution,
            "compute_type": "bfloat16",
            "gpu": gpu_name,
            "compute_capability": f"{compute_cap[0]}.{compute_cap[1]}",
            "mcp_server": server_script,
            "mcp_dispatch_enabled": mcp_client is not None,
        },
        "summary": {
            "total_clips": len(clips),
            "anomalies": counts["Anomaly"],
            "normals": counts["Normal"],
            "unknowns": counts["Unknown"],
            "errors": counts["Error"],
            "emergency_stops": mrm_counts["EMERGENCY_STOP"],
            "comfortable_stops": mrm_counts["COMFORTABLE_STOP"],
            "pull_overs": mrm_counts["PULL_OVER"],
            "mcp_failures": mcp_failures,
            "total_load_time_s": round(total_load_time, 3),
            "total_inference_time_s": round(total_inference_time, 3),
            "total_stage2_inference_time_s": round(total_stage2_time, 3),
            "total_time_s": round(total_time, 3),
            "avg_inference_time_s": round(total_inference_time / max(len(clips), 1), 3),
        },
        "metrics": metrics.compute() if metrics.count > 0 else None,
        "results": results,
    }

    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print("=" * 50)
    print(f"\nSUMMARY — FP16 Multiview Inference ({len(VIEW_DIRS)} views)")
    print("=" * 50)
    print(f"Total clips: {len(clips)}")
    print(f"  - Anomaly: {counts['Anomaly']}")
    print(f"  - Normal: {counts['Normal']}")
    print(f"  - Unknown: {counts['Unknown']}")
    print(f"  - Errors: {counts['Error']}")
    print(f"\nMRM DISPATCH")
    print(f"  - EMERGENCY_STOP:   {mrm_counts['EMERGENCY_STOP']}")
    print(f"  - COMFORTABLE_STOP: {mrm_counts['COMFORTABLE_STOP']}")
    print(f"  - PULL_OVER:        {mrm_counts['PULL_OVER']}")
    if mcp_client is not None:
        print(f"  - MCP failures:     {mcp_failures}")
    print(f"\nTotal load time: {total_load_time:.2f}s")
    print(f"Total stage-1 inference time: {total_inference_time:.2f}s")
    print(f"Total stage-2 (MRM) inference time: {total_stage2_time:.2f}s")
    print(f"Average stage-1 inference time: {total_inference_time / max(len(clips), 1):.2f}s per clip")

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

    output_video_path = Path(args.output_video) if args.output_video else video_dir / "collage_output.mp4"
    generate_collage_video(
        clips, results, output_video_path,
        native_fps=args.native_fps,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
    )

    print("FP16 multiview inference complete.")


if __name__ == "__main__":
    asyncio.run(main())
