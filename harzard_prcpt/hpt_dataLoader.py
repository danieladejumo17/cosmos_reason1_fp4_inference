import os
import sys
from pathlib import Path
from typing import Generator
from natsort import natsorted

import cv2

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.dataloader import CosmosInferenceDataloader


class HarzardPerceptionTestDataLoader(CosmosInferenceDataloader):
    def __init__(
        self,
        video_folder: str | Path,
        labels_csv_file: str | Path,
        min_hazard_frames: int = 10,
        out_video_folder: str | Path = None,
        separate_videos: bool = False,
    ):
        self.video_folder = Path(video_folder).resolve()
        self.labels_csv_file = Path(labels_csv_file).resolve()
        self.min_hazard_frames = min_hazard_frames
        self.separate_videos = separate_videos

        if out_video_folder is None:
            self.out_video_folder = self.video_folder.parent / "output_videos"
        else:
            self.out_video_folder = Path(out_video_folder).resolve()
        self.out_video_folder.mkdir(exist_ok=True, parents=True)

        self.labels_dict = {}
        with open(self.labels_csv_file, 'r') as file:
            for line in file:
                file_name, start_sec, end_sec = line.strip().split(',')
                self.labels_dict[file_name] = (float(start_sec), float(end_sec))

    def __iter__(self) -> Generator[tuple[Path, bool, list], None, None]:
        print(f"[HPT_DL] Output videos will be saved to: {self.out_video_folder}")

        for video_file in natsorted(os.listdir(self.video_folder)):
            if video_file.endswith('.mp4') or video_file.endswith('.avi') or video_file.endswith('.webm'):
                video_path = self.video_folder / video_file
                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    print(f"Error opening video file {video_path}")
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)
                # print(f"[HPT_DL] FPS for {video_path}: {fps}")
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_sec = total_frames / fps

                if video_path.stem in self.labels_dict:
                    start_sec, end_sec = self.labels_dict[video_path.stem]
                    hazard_start_frame = int(start_sec * fps)
                    hazard_end_frame = int(end_sec * fps)

                    print(f"[HPT_DL] Processing {video_path}:")
                    print(f"[HPT_DL]  - Duration: {duration_sec:.2f} seconds")
                    print(f"[HPT_DL]  - Hazard frames: {hazard_start_frame} to {hazard_end_frame}")
                    print(f"[HPT_DL]  - Hazard seconds: {start_sec} to {end_sec}")

                    frames = []
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)

                    window_duration_sec = 5
                    stride_sec = 2
                    window_size = int(window_duration_sec * fps)
                    stride_size = int(stride_sec * fps)
                    for start_frame in range(0, total_frames - window_size + 1, stride_size):
                        end_frame = start_frame + window_size
                        window_frames = frames[start_frame:end_frame]

                        # Determine if this window contains a hazard
                        # hazard end frame has to be to the right of start frame
                        # and hazard start frame has to be to the left of end frame
                        if (start_frame <= (hazard_end_frame - self.min_hazard_frames)) and (end_frame >= (hazard_start_frame + self.min_hazard_frames)):
                            label = True
                        else:
                            label = False

                        if self.separate_videos:
                            temp_video_path = self.out_video_folder / f"{(label and 'Anom') or 'Norm'}_{video_path.stem}_{start_frame}_{end_frame}.mp4"
                        else:
                            temp_video_path = self.out_video_folder / "temp_video_out.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (window_frames[0].shape[1], window_frames[0].shape[0]))
                        for frame in window_frames:
                            out.write(frame)
                        out.release()

                        print(f"[HPT_DL] Window {start_frame}-{end_frame}: Label={label}")

                        if start_frame == 0:
                            new_images = window_frames
                        else:
                            new_images = window_frames[-stride_size:]

                        yield temp_video_path, label, new_images
                else:
                    print(f"No labels found for {video_file}")

                cap.release()
