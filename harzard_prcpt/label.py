import cv2
import csv
import os
import pandas as pd
import re

# -----------------------------
# CONFIGURATION
# -----------------------------
VIDEO_FOLDER = "videos/"         # folder containing all videos
CSV_OUTPUT = "labels.csv"        # combined output CSV

# Only 'a' is anomaly, everything else is normal
ANOMALY_KEY = "a"

FONT = cv2.FONT_HERSHEY_SIMPLEX


# -----------------------------
# UTIL FUNCTIONS
# -----------------------------
def write_csv_row(csv_path, row):
    try:
        df = pd.DataFrame([row])
        df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
        print(f"  ✓ Saved to CSV: {row}")
    except Exception as e:
        print(f"  ✗ ERROR saving to CSV: {e}")


def draw_text(img, text, y):
    cv2.putText(img, text, (20, y), FONT, 0.7, (0, 255, 0), 2)


# -----------------------------
# LABEL ONE VIDEO
# -----------------------------
def label_video(video_path, csv_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_id = os.path.basename(video_path).split(".")[0]

    paused = False
    labeling = False
    start_time = 0
    current_label = None
    frame = None
    frame_no = 0
    time_sec = 0.0

    print(f"\nNow labeling: {video_id}")
    print("Controls:")
    print("  SPACE → pause/resume")
    print("  A → start/end anomaly (only anomalies need to be labeled)")
    print("  Q → quit/skip this video")
    print()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
            time_sec = frame_no / fps

        if frame is None:
            break

        display = frame.copy()

        # Timeline bar
        progress = frame_no / cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cv2.rectangle(display, (0, display.shape[0]-5), (int(progress * display.shape[1]), display.shape[0]), (0,255,0), -1)

        # On-screen texts
        draw_text(display, f"Video: {video_id}", 30)
        draw_text(display, f"Time: {time_sec:.2f}s", 60)
        draw_text(display, f"Paused: {paused}", 90)

        if labeling:
            draw_text(display, f"Labeling: {current_label}", 120)
            cv2.rectangle(display, (0,0), (display.shape[1], display.shape[0]), (0,0,255), 4)

        cv2.imshow("Video Labeler", display)

        key = cv2.waitKey(1) & 0xFF

        # Quit video
        if key == ord("q"):
            break

        # Pause/Resume
        if key == ord(" "):
            paused = not paused

        # Label events - only 'a' key is used for labeling anomalies
        if key != 0 and chr(key).lower() == ANOMALY_KEY:
            # START LABELING ANOMALY
            if not labeling:
                labeling = True
                start_time = time_sec
                current_label = "anomaly"
                print(f"START anomaly at {start_time:.2f}s")

            # END LABELING ANOMALY → save CSV row
            else:
                end_time = time_sec
                labeling = False

                row = {
                    "video_id": video_id,
                    "start_seconds": round(start_time, 3),
                    "end_seconds": round(end_time, 3),
                    "label": "anomaly"
                }

                write_csv_row(csv_path, row)
                print(f"END anomaly at {end_time:.2f}s → SAVED")

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# MAIN LOOP FOR ALL VIDEOS
# -----------------------------
def natural_sort_key(text):
    """Convert text to a list of strings and numbers for natural sorting."""
    def convert(text_part):
        return int(text_part) if text_part.isdigit() else text_part.lower()
    return [convert(c) for c in re.split(r'(\d+)', text)]


def run_labeler():
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith((".mp4", ".avi", ".mov", ".webm"))]

    if len(video_files) == 0:
        print("No videos found in folder!")
        return

    # Sort videos in natural/sequential order (handles numbers correctly: 1, 2, 10 not 1, 10, 2)
    video_files.sort(key=natural_sort_key)
    total_videos = len(video_files)
    
    print("\n" + "="*50)
    print(f"FOUND {total_videos} VIDEO(S) TO LABEL")
    print("="*50)
    for i, video in enumerate(video_files, 1):
        print(f"  {i}. {video}")
    print("="*50 + "\n")

    for idx, video in enumerate(video_files, 1):
        print(f"\n{'='*50}")
        print(f"VIDEO {idx}/{total_videos}: {video}")
        print(f"{'='*50}")
        label_video(os.path.join(VIDEO_FOLDER, video), CSV_OUTPUT)
        print(f"\n✓ Completed video {idx}/{total_videos}: {video}")

    print("\n" + "="*50)
    print("LABELING COMPLETED!")
    print(f"Processed {total_videos} video(s)")
    print(f"Saved CSV → {CSV_OUTPUT}")
    print("="*50)


if __name__ == "__main__":
    run_labeler()
