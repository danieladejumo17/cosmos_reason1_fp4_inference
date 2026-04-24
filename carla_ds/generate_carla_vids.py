#!/usr/bin/env python3
"""
Convert sequential CARLA camera frames into a sliding-window video dataset
compatible with the multiview inference script.

Each source subfolder is processed independently.  Files are sorted by their
four-digit numeric ID so that output clip N across every view always
corresponds to the same range of image IDs.

Source layout:
    camera_clean_individualframe/{front,left,right,rear}/{Norm,Anom}_XXXX.jpg

Output layout:
    carla_vid_ds/{front_view,left_view,right_view,back_view}/{Anom,Norm}_XXXX.mp4
"""

import argparse
import subprocess
from pathlib import Path

import cv2

VIEW_MAP = {
    "front": "front_view",
    "left": "left_view",
    "right": "right_view",
    "rear": "back_view",
}


def _frame_id(filename: str) -> int:
    """Extract the four-digit numeric ID from e.g. 'Norm_0042.jpg' -> 42."""
    return int(Path(filename).stem.rsplit("_", 1)[-1])


def discover_frames(view_dir: Path) -> list[str]:
    """List frame filenames from a view directory, sorted by numeric ID."""
    if not view_dir.is_dir():
        raise FileNotFoundError(f"Expected source directory not found: {view_dir}")

    frames = [
        f.name for f in view_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg")
        and not f.name.endswith(":Zone.Identifier")
    ]
    frames.sort(key=_frame_id)
    return frames


def has_anomaly(frame_names: list[str]) -> bool:
    return any(name.startswith("Anom_") for name in frame_names)


def write_video(frames_dir: Path, frame_names: list[str], output_path: Path, fps: int):
    first_frame = cv2.imread(str(frames_dir / frame_names[0]))
    if first_frame is None:
        raise IOError(f"Could not read frame: {frames_dir / frame_names[0]}")
    h, w = first_frame.shape[:2]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-loglevel", "error",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    proc.stdin.write(first_frame.tobytes())
    for name in frame_names[1:]:
        frame = cv2.imread(str(frames_dir / name))
        if frame is None:
            print(f"  WARNING: skipping unreadable frame {frames_dir / name}")
            continue
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode} for {output_path}")


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

    missing_src = [v for v in VIEW_MAP if not (input_dir / v).is_dir()]
    if missing_src:
        raise FileNotFoundError(f"Missing source view directories: {missing_src}")

    for out_view in VIEW_MAP.values():
        (output_dir / out_view).mkdir(parents=True, exist_ok=True)

    window_size = args.fps * args.window_sec
    step_size = args.fps * args.step_sec

    for src_view, out_view in VIEW_MAP.items():
        src_dir = input_dir / src_view
        frame_names = discover_frames(src_dir)
        total_frames = len(frame_names)
        print(f"\n=== {src_view} -> {out_view} ({total_frames} frames) ===")

        if total_frames < window_size:
            print(f"  Not enough frames ({total_frames}) for a {args.window_sec}s window "
                  f"({window_size} frames needed). Skipping.")
            continue

        video_idx = 1
        anom_count = 0
        norm_count = 0

        for start in range(0, total_frames - window_size + 1, step_size):
            window_frames = frame_names[start : start + window_size]

            if has_anomaly(window_frames):
                label = "Anom"
                anom_count += 1
            else:
                label = "Norm"
                norm_count += 1

            clip_name = f"{label}_{video_idx:04d}.mp4"
            out_path = output_dir / out_view / clip_name
            write_video(src_dir, window_frames, out_path, args.fps)

            id_start = _frame_id(window_frames[0])
            id_end = _frame_id(window_frames[-1])
            print(f"  [{video_idx}] {clip_name}  (IDs {id_start:04d}-{id_end:04d})")
            video_idx += 1

        total_clips = video_idx - 1
        print(f"  {out_view}: {total_clips} clips ({anom_count} Anomaly, {norm_count} Normal)")

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
