#!/usr/bin/env python3
"""
Convert sequential CARLA camera frames into a sliding-window video dataset
compatible with the multiview inference script.

Source layout:
    camera_clean_individualframe/{front,left,right,rear}/000000.png ...

Output layout:
    carla_vid_ds/{front_view,left_view,right_view,back_view}/Anom_0001.mp4 ...
"""

import argparse
from pathlib import Path

import cv2
from natsort import natsorted

VIEW_MAP = {
    "front": "front_view",
    "left": "left_view",
    "right": "right_view",
    "rear": "back_view",
}
SOURCE_VIEWS = list(VIEW_MAP.keys())


def discover_frames(input_dir: Path) -> list[str]:
    """List canonical frame filenames from the front/ directory, naturally sorted."""
    front_dir = input_dir / "front"
    if not front_dir.is_dir():
        raise FileNotFoundError(f"Expected source directory not found: {front_dir}")

    frames = natsorted(
        f.name for f in front_dir.iterdir()
        if f.suffix.lower() == ".png" and not f.name.endswith(":Zone.Identifier")
    )
    return frames


def has_anomaly(frame_names: list[str]) -> bool:
    return any(name.startswith("Anom_") for name in frame_names)


def write_video(frames_dir: Path, frame_names: list[str], output_path: Path, fps: int):
    first_frame = cv2.imread(str(frames_dir / frame_names[0]))
    if first_frame is None:
        raise IOError(f"Could not read frame: {frames_dir / frame_names[0]}")

    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    writer.write(first_frame)
    for name in frame_names[1:]:
        frame = cv2.imread(str(frames_dir / name))
        if frame is None:
            print(f"  WARNING: skipping unreadable frame {frames_dir / name}")
            continue
        writer.write(frame)

    writer.release()


def main():
    parser = argparse.ArgumentParser(
        description="Generate sliding-window video dataset from CARLA camera frames"
    )
    parser.add_argument("--input_dir", type=str, default="carla_ds/camera_clean_individualframe",
                        help="Root dir with front/, left/, right/, rear/ frame folders")
    parser.add_argument("--output_dir", type=str, default="carla_ds/carla_vid_ds",
                        help="Output dir for front_view/, left_view/, right_view/, back_view/")
    parser.add_argument("--fps", type=int, default=20,
                        help="Capture framerate of the original images")
    parser.add_argument("--window_sec", type=int, default=5,
                        help="Duration of each output video in seconds")
    parser.add_argument("--step_sec", type=int, default=2,
                        help="Sliding window step size in seconds")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    missing_src = [v for v in SOURCE_VIEWS if not (input_dir / v).is_dir()]
    if missing_src:
        raise FileNotFoundError(f"Missing source view directories: {missing_src}")

    for out_view in VIEW_MAP.values():
        (output_dir / out_view).mkdir(parents=True, exist_ok=True)

    frame_names = discover_frames(input_dir)
    total_frames = len(frame_names)
    print(f"Discovered {total_frames} frames in {input_dir / 'front'}")

    window_size = args.fps * args.window_sec
    step_size = args.fps * args.step_sec

    if total_frames < window_size:
        print(f"Not enough frames ({total_frames}) for a {args.window_sec}s window "
              f"({window_size} frames needed). Exiting.")
        return

    video_idx = 1
    anom_count = 0
    norm_count = 0

    for start in range(0, total_frames - window_size + 1, step_size):
        window_frames = frame_names[start : start + window_size]

        if has_anomaly(window_frames):
            label = "Anom_"
            anom_count += 1
        else:
            label = "Norm_"
            norm_count += 1

        clip_name = f"{label}{video_idx:04d}.mp4"

        for src_view, out_view in VIEW_MAP.items():
            src_dir = input_dir / src_view
            out_path = output_dir / out_view / clip_name
            write_video(src_dir, window_frames, out_path, args.fps)

        print(f"[{video_idx}] {clip_name}  (frames {start}-{start + window_size - 1})")
        video_idx += 1

    total_clips = video_idx - 1
    print(f"\nDone. Generated {total_clips} clips ({anom_count} Anomaly, {norm_count} Normal)")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
