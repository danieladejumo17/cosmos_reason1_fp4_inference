#!/usr/bin/env python3
"""
Real-time VLM inference loop that subscribes to Autoware camera topics,
analyzes scenes with Qwen2.5-VL, and controls the ego vehicle via the
autoware_mcp_server.

Architecture:
    Camera topics ──▸ FrameBuffer ──▸ VLM inference ──▸ action parser
                                                              │
                                                        SafetyGate
                                                              │
                                                  MCP client (stdio) ──▸ autoware_mcp_server

Usage:
    python vlm_autoware_loop.py \\
        --camera_topics /sensing/camera/camera0/image_rect_color \\
        --mode advisory

    python vlm_autoware_loop.py \\
        --camera_topics /sensing/camera/front/image_rect_color,/sensing/camera/rear/image_rect_color \\
        --mode autonomous --auto_estop
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import transformers
import qwen_vl_utils

from fastmcp import Client

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"

# ROS 2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge


# ===================================================================
# Model loading (mirrors multiview_batch_fp16_inference.py)
# ===================================================================
def load_model(model_name: str):
    print("Loading and compiling model...")
    start = time.time()
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    model.gradient_checkpointing_disable()
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)
    print(f"Model ready in {time.time() - start:.2f}s\n")
    return model, processor


def warmup_model(model, processor):
    print("Warming up model (compiling kernels)...")
    dummy_conv = [{"role": "user", "content": [{"type": "text", "text": "Is this scene safe?"}]}]
    text = processor.apply_chat_template(dummy_conv, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=7)
    torch.cuda.synchronize()
    print("Warmup complete.\n")


# ===================================================================
# Action-oriented prompt
# ===================================================================
ACTION_PROMPT_TEMPLATE = (
    "You are an autonomous driving safety system analyzing the ego vehicle's "
    "{view} camera feed. Based on what you observe, recommend ONE action.\n\n"
    "Available actions:\n"
    "- NONE -- No intervention needed, driving is safe\n"
    "- SET_VELOCITY <speed_mps> -- Adjust speed (e.g. slow down for caution)\n"
    "- EMERGENCY_STOP -- Immediate stop for critical danger\n"
    "- STOP_OVERRIDE <decel_mps2> -- Controlled stop with given deceleration\n"
    "- STEER_LEFT <angle_deg> -- Steer left by given degrees\n"
    "- STEER_RIGHT <angle_deg> -- Steer right by given degrees\n"
    "- LANE_SWITCH_LEFT -- Switch to the left lane\n"
    "- LANE_SWITCH_RIGHT -- Switch to the right lane\n\n"
    "<think>\n"
    "[Your reasoning about what you observe]\n"
    "</think>\n\n"
    "<action>\n"
    "[Exactly one action from the list above with parameters]\n"
    "</action>"
)


def build_action_prompt(view: str = "front") -> str:
    return ACTION_PROMPT_TEMPLATE.format(view=view)


# ===================================================================
# Action parser
# ===================================================================
_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)

ACTION_MAP: dict[str, str] = {
    "NONE": "none",
    "SET_VELOCITY": "set_velocity",
    "EMERGENCY_STOP": "emergency_stop",
    "STOP_OVERRIDE": "stop_override",
    "STEER_LEFT": "steer",
    "STEER_RIGHT": "steer",
    "LANE_SWITCH_LEFT": "lane_switch",
    "LANE_SWITCH_RIGHT": "lane_switch",
}


def parse_action(raw: str) -> dict:
    """Parse VLM output into an action dict.

    Returns:
        {"tool": "...", "args": {...}, "reasoning": "...", "raw_action": "..."}
        or {"tool": "none", ...} when no intervention is needed.
    """
    reasoning = ""
    think_m = _THINK_RE.search(raw)
    if think_m:
        reasoning = think_m.group(1).strip()

    action_m = _ACTION_RE.search(raw)
    if not action_m:
        return {"tool": "none", "args": {}, "reasoning": reasoning, "raw_action": ""}

    action_str = action_m.group(1).strip()
    tokens = action_str.split()
    if not tokens:
        return {"tool": "none", "args": {}, "reasoning": reasoning, "raw_action": action_str}

    keyword = tokens[0].upper()

    if keyword == "NONE":
        return {"tool": "none", "args": {}, "reasoning": reasoning, "raw_action": action_str}

    if keyword == "SET_VELOCITY" and len(tokens) >= 2:
        try:
            speed = float(tokens[1])
        except ValueError:
            speed = 5.0
        return {"tool": "set_velocity", "args": {"velocity_mps": speed},
                "reasoning": reasoning, "raw_action": action_str}

    if keyword == "EMERGENCY_STOP":
        return {"tool": "emergency_stop", "args": {},
                "reasoning": reasoning, "raw_action": action_str}

    if keyword == "STOP_OVERRIDE":
        decel = 2.0
        if len(tokens) >= 2:
            try:
                decel = float(tokens[1])
            except ValueError:
                pass
        return {"tool": "stop_override", "args": {"deceleration_mps2": decel},
                "reasoning": reasoning, "raw_action": action_str}

    if keyword == "STEER_LEFT":
        angle = 10.0
        if len(tokens) >= 2:
            try:
                angle = float(tokens[1])
            except ValueError:
                pass
        return {"tool": "steer", "args": {"direction": "left", "angle_deg": angle},
                "reasoning": reasoning, "raw_action": action_str}

    if keyword == "STEER_RIGHT":
        angle = 10.0
        if len(tokens) >= 2:
            try:
                angle = float(tokens[1])
            except ValueError:
                pass
        return {"tool": "steer", "args": {"direction": "right", "angle_deg": angle},
                "reasoning": reasoning, "raw_action": action_str}

    if keyword == "LANE_SWITCH_LEFT":
        return {"tool": "lane_switch", "args": {"direction": "left"},
                "reasoning": reasoning, "raw_action": action_str}

    if keyword == "LANE_SWITCH_RIGHT":
        return {"tool": "lane_switch", "args": {"direction": "right"},
                "reasoning": reasoning, "raw_action": action_str}

    return {"tool": "none", "args": {}, "reasoning": reasoning, "raw_action": action_str}


# ===================================================================
# Frame buffer — ring buffer per camera view
# ===================================================================
class FrameBuffer:
    """Thread-safe ring buffer for camera frames keyed by view name."""

    def __init__(self, max_frames: int = 8):
        self._buffers: dict[str, collections.deque[np.ndarray]] = {}
        self._max = max_frames
        self._lock = threading.Lock()

    def push(self, view: str, frame: np.ndarray) -> None:
        with self._lock:
            if view not in self._buffers:
                self._buffers[view] = collections.deque(maxlen=self._max)
            self._buffers[view].append(frame)

    def snapshot(self, view: str) -> list[np.ndarray]:
        with self._lock:
            buf = self._buffers.get(view)
            if buf is None:
                return []
            return list(buf)

    def views(self) -> list[str]:
        with self._lock:
            return list(self._buffers.keys())


# ===================================================================
# ROS 2 camera subscriber node
# ===================================================================
def _view_name_from_topic(topic: str) -> str:
    """Derive a short view name from a camera topic path."""
    parts = topic.rstrip("/").split("/")
    for kw in ("front", "rear", "left", "right"):
        if kw in topic.lower():
            return kw
    if len(parts) >= 3:
        return parts[-2]
    return "cam"


class CameraSubscriberNode(Node):
    def __init__(
        self,
        topics: list[str],
        frame_buffer: FrameBuffer,
        target_resolution: tuple[int, int],
    ):
        super().__init__("vlm_camera_sub")
        self._fb = frame_buffer
        self._bridge = CvBridge()
        self._res = target_resolution

        cam_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        for topic in topics:
            view = _view_name_from_topic(topic)
            self.create_subscription(
                ROSImage,
                topic,
                lambda msg, v=view: self._on_image(msg, v),
                cam_qos,
            )
            self.get_logger().info(f"Subscribed to {topic} as view '{view}'")

    def _on_image(self, msg: ROSImage, view: str) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self._res:
            frame = cv2.resize(frame, self._res, interpolation=cv2.INTER_LINEAR)
        self._fb.push(view, frame)


# ===================================================================
# Video encoding helper
# ===================================================================
def frames_to_temp_video(
    frames: list[np.ndarray], fps: int = 4
) -> str:
    """Write frames to a temp .mp4 file and return the path.

    Caller is responsible for deleting the file after use.
    """
    if not frames:
        raise ValueError("No frames to encode")
    h, w = frames[0].shape[:2]
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return path


# ===================================================================
# Inference
# ===================================================================
def run_inference(
    model,
    processor,
    video_path: str,
    view: str,
    max_tokens: int,
) -> str:
    """Run a single VLM inference on a video clip and return raw text."""
    prompt = build_action_prompt(view)
    content = [
        {"type": "video", "video": video_path},
        {"type": "text", "text": prompt},
    ]
    conversation = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(
        [{"role": "user", "content": content}]
    )

    inputs = processor(
        text=[text],
        images=image_inputs or None,
        videos=video_inputs or None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    new_tokens = output[:, inputs.input_ids.shape[1] :]
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)
    return decoded[0].strip()


# ===================================================================
# Safety gate
# ===================================================================
class SafetyGate:
    """Cooldown-based gate to prevent rapid repeated commands."""

    def __init__(self, cooldown_s: float = 5.0):
        self._cooldown = cooldown_s
        self._last_action: dict[str, float] = {}

    def allow(self, tool_name: str) -> bool:
        if tool_name == "none":
            return True
        now = time.time()
        last = self._last_action.get(tool_name, 0.0)
        if now - last < self._cooldown:
            return False
        self._last_action[tool_name] = now
        return True


# ===================================================================
# MCP client helper
# ===================================================================
async def call_mcp_tool(client: Client, tool_name: str, args: dict) -> str:
    """Call a tool on the MCP server and return the result text."""
    result = await client.call_tool(tool_name, args)
    if result.content:
        return result.content[0].text
    return "{}"


# ===================================================================
# Main loop
# ===================================================================
async def main_loop(args: argparse.Namespace) -> None:
    # --- Load VLM ---
    model, processor = load_model(args.model)
    warmup_model(model, processor)

    width, height = map(int, args.target_resolution.split("x"))
    target_resolution = (width, height)

    # --- Frame buffer & ROS 2 camera subscriber ---
    max_frames = args.fps * max(1, int(args.inference_interval))
    frame_buffer = FrameBuffer(max_frames=max_frames)

    rclpy.init()
    camera_topics = [t.strip() for t in args.camera_topics.split(",")]
    cam_node = CameraSubscriberNode(camera_topics, frame_buffer, target_resolution)

    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(cam_node), daemon=True
    )
    spin_thread.start()
    print(f"Camera subscriber started for {camera_topics}")

    # --- MCP client (stdio to autoware_mcp_server.py) ---
    server_script = str(Path(__file__).parent / "autoware_mcp_server.py")
    mcp_client = Client(["python3", server_script])

    safety_gate = SafetyGate(cooldown_s=5.0)
    cycle = 0

    async with mcp_client:
        print(f"\nVLM loop running — mode={args.mode}, interval={args.inference_interval}s\n")

        while True:
            cycle += 1
            cycle_start = time.time()

            views = frame_buffer.views()
            if not views:
                await asyncio.sleep(0.5)
                continue

            for view in views:
                frames = frame_buffer.snapshot(view)
                if len(frames) < 2:
                    continue

                # Encode frames to temp video
                tmp_path = frames_to_temp_video(frames, fps=args.fps)
                try:
                    inf_start = time.time()
                    raw_output = run_inference(
                        model, processor, tmp_path, view, args.max_tokens
                    )
                    inf_time = time.time() - inf_start

                    action = parse_action(raw_output)
                    tool = action["tool"]

                    print(f"[cycle {cycle}] {view}: action={action['raw_action']!r} "
                          f"({inf_time:.2f}s)")
                    if action["reasoning"]:
                        print(f"  reasoning: {action['reasoning'][:200]}")

                    if tool == "none":
                        continue

                    # Safety gate
                    if not safety_gate.allow(tool):
                        print(f"  [cooldown] {tool} blocked by safety gate")
                        continue

                    # Mode handling
                    execute = False
                    if args.mode == "autonomous":
                        execute = True
                    elif args.mode == "advisory":
                        if tool == "emergency_stop" and args.auto_estop:
                            print("  [auto_estop] executing emergency_stop automatically")
                            execute = True
                        else:
                            try:
                                resp = input(
                                    f"  Execute {tool}({action['args']})? "
                                    f"[Enter=yes / s=skip]: "
                                )
                                execute = resp.strip().lower() != "s"
                            except EOFError:
                                execute = False

                    if execute:
                        try:
                            result = await asyncio.wait_for(
                                call_mcp_tool(mcp_client, tool, action["args"]),
                                timeout=2.0,
                            )
                            print(f"  -> MCP result: {result}")
                        except asyncio.TimeoutError:
                            print(f"  -> MCP call timed out for {tool}")
                        except Exception as e:
                            print(f"  -> MCP error: {e}")
                    else:
                        print(f"  [skipped] {tool}")

                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            # Pace the loop
            elapsed = time.time() - cycle_start
            sleep_time = max(0, args.inference_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)


# ===================================================================
# CLI
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time VLM loop controlling Autoware via MCP"
    )
    parser.add_argument(
        "--model", type=str, default="nvidia/Cosmos-Reason1-7B",
        help="HuggingFace model name (default: nvidia/Cosmos-Reason1-7B)",
    )
    parser.add_argument(
        "--camera_topics", type=str,
        default="/sensing/camera/camera0/image_rect_color",
        help="Comma-separated ROS 2 image topics",
    )
    parser.add_argument(
        "--fps", type=int, default=4,
        help="Frame sampling rate for video clips (default: 4)",
    )
    parser.add_argument(
        "--inference_interval", type=float, default=2.0,
        help="Seconds between inference cycles (default: 2.0)",
    )
    parser.add_argument(
        "--mode", type=str, choices=["autonomous", "advisory"], default="advisory",
        help="autonomous: act immediately; advisory: ask before executing (default: advisory)",
    )
    parser.add_argument(
        "--auto_estop", action="store_true",
        help="In advisory mode, auto-execute emergency stops without confirmation",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=128,
        help="Max generation tokens (default: 128)",
    )
    parser.add_argument(
        "--target_resolution", type=str, default="250x250",
        help="Frame resize resolution WxH (default: 250x250)",
    )
    args = parser.parse_args()
    asyncio.run(main_loop(args))


if __name__ == "__main__":
    main()
