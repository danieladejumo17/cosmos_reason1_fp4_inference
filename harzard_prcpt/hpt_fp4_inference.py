"""
Run Cosmos-Reason1-7B FP4 inference on the Hazard Perception Test dataset.

Uses CosmosFP4Runner (TensorRT-LLM, Blackwell FP4 Tensor Cores) with the
same HarzardPerceptionTestDataLoader used by the FP8 pipeline.

Prerequisites:
    1. source ../cosmos_reason1_fp4_inference/activate_fp4.sh
    2. Ensure the NVFP4 checkpoint exists (run quantize_cosmos_fp4.py first)

Usage:
    python hpt_fp4_inference.py
    python hpt_fp4_inference.py --video_dir ./videos --labels labels.csv
    python hpt_fp4_inference.py --model ../cosmos_reason1_fp4_inference/cosmos-reason1-nvfp4
"""

import argparse
import sys
from pathlib import Path

import cv2

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cosmos_reason1_fp4_inference import CosmosFP4Runner
from hpt_dataLoader import HarzardPerceptionTestDataLoader


def main():
    parser = argparse.ArgumentParser(
        description="Hazard Perception Test -- Cosmos-Reason1-7B FP4 Inference",
    )
    parser.add_argument("--video_dir", type=str, default="./videos",
                        help="Directory containing test videos (default: ./videos)")
    parser.add_argument("--labels", type=str, default="labels.csv",
                        help="CSV file with video labels (default: labels.csv)")
    parser.add_argument("--min_hazard_frames", type=int, default=10,
                        help="Min overlapping hazard frames for positive label (default: 10)")
    parser.add_argument("--output_dir", type=str, default="./output_videos_fp4",
                        help="Directory for output videos (default: ./output_videos_fp4)")
    parser.add_argument("--separate_videos", action="store_true", default=True,
                        help="Write each window as a separate video file")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to NVFP4 checkpoint (default: cosmos-reason1-nvfp4 in FP4 module dir)")
    parser.add_argument("--source_model", type=str, default=None,
                        help="HuggingFace model for the processor (default: nvidia/Cosmos-Reason1-7B)")
    parser.add_argument("--fps", type=int, default=4,
                        help="Target FPS for video sampling (default: 4)")
    parser.add_argument("--max_tokens", type=int, default=7,
                        help="Max new tokens to generate (default: 7)")
    parser.add_argument("--target_resolution", type=str, default="250x250",
                        help="Target resolution WxH for video frames (default: 250x250)")
    args = parser.parse_args()

    width, height = map(int, args.target_resolution.split("x"))

    user_prompt = (
        "You are an autonomous driving safety expert analyzing this video for "
        "EXTERNAL ANOMALIES that may impact safe AV operation.\n\n"
        "<think>\n"
        "Focus on:\n"
        "- Obstacles, pedestrians, or vehicles violating rules\n"
        "- Roadwork, blocked lanes, poor visibility, or hazards\n"
        "- Reflections, shadows, or false visual cues confusing perception\n"
        "</think>\n\n"
        "<answer>\n"
        "Is there any external anomaly in this video? Reply with exactly one of the following:\n"
        "Classification: Anomaly \u2014 if any obstacle, obstruction, or unsafe condition is visible.\n"
        "Classification: Normal \u2014 if no anomaly or obstruction is visible.\n"
        "</answer>"
    )

    runner_kwargs = dict(
        user_prompt=user_prompt,
        target_fps=args.fps,
        max_tokens=args.max_tokens,
        target_resolution=(width, height),
    )
    if args.model is not None:
        runner_kwargs["model_path"] = args.model
    if args.source_model is not None:
        runner_kwargs["source_model"] = args.source_model

    runner = CosmosFP4Runner(**runner_kwargs)

    output_root = Path(args.output_dir)
    output_root.mkdir(exist_ok=True)

    # Probe the first video window to get frame dimensions for the output video
    probe_loader = HarzardPerceptionTestDataLoader(
        args.video_dir, args.labels,
        min_hazard_frames=args.min_hazard_frames,
        out_video_folder=str(output_root),
        separate_videos=args.separate_videos,
    )
    first_video_path, _, _ = next(iter(probe_loader))
    cap = cv2.VideoCapture(str(first_video_path))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"\nOutput video params -- FPS: {vid_fps}, Width: {vid_width}, Height: {vid_height}\n")

    # Fresh dataloader for the full run
    dataloader = HarzardPerceptionTestDataLoader(
        args.video_dir, args.labels,
        min_hazard_frames=args.min_hazard_frames,
        out_video_folder=str(output_root),
        separate_videos=args.separate_videos,
    )

    metrics = runner.run_inference_dataloader(
        dataloader,
        vid_output=str(output_root / "fp4_inference_results.mp4"),
        vid_fps=vid_fps,
        vid_width=vid_width,
        vid_height=vid_height,
    )

    print("\n" + "=" * 50)
    print("Final Metrics (Cosmos-Reason1-7B FP4)")
    print("=" * 50)
    for key, value in metrics.compute().items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
