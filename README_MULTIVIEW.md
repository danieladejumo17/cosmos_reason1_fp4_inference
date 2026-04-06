# Multiview FP16 Inference — Cosmos-Reason1-7B

Anomaly detection on synchronized multi-camera autonomous driving video using NVIDIA Cosmos-Reason1-7B. The model receives four camera views (front, left, right, rear) as a single conversation and classifies the scene as **Anomaly** or **Normal**.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on A100, RTX 4090)
- `ffmpeg` installed (`sudo apt install ffmpeg`)
- [vast.ai](https://vast.ai/) instance (recommended) or any CUDA-capable machine

## 1. Environment Setup

On a fresh vast.ai instance (or any machine), install all Python dependencies:

```bash
bash setup_fp16.sh
```

Or create an isolated virtual environment first:

```bash
bash setup_fp16.sh --venv
```

Verify the installation:

```bash
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available())"
python -c "import transformers; print('Transformers', transformers.__version__)"
```

## 2. Download the CARLA Dataset

Download and extract the CARLA camera frames into `carla_ds/`:

```bash
cd carla_ds/

# Download from Box. I have this folder public.
wget -L -O camera.zip "https://nyu.box.com/shared/static/66khwx1o4owrzdt77jeb8cljjd2n8fxl"

# Extract
unzip camera.zip -d camera_clean_individualframe

cd ..
```

After extraction, the folder structure should be:

```
carla_ds/
  camera_clean_individualframe/
    front/   000000.png, 000001.png, ...
    left/    000000.png, 000001.png, ...
    right/   000000.png, 000001.png, ...
    rear/    000000.png, 000001.png, ...
```

Synchronized frames share the same filename across all four view folders. Frame filenames may optionally have an `Anom_` or `Norm_` prefix (e.g., `Anom_000050.png`) to indicate ground-truth anomaly labels.

## 3. Generate the Video Dataset

Convert the image frames into sliding-window mp4 clips:

```bash
python carla_ds/generate_carla_vids.py
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--input_dir` | `carla_ds/camera_clean_individualframe` | Root dir with `front/`, `left/`, `right/`, `rear/` frame folders |
| `--output_dir` | `carla_ds/carla_vid_ds` | Output dir for generated video clips |
| `--fps` | `20` | Capture framerate of the original images |
| `--window_sec` | `5` | Duration of each output video in seconds |
| `--step_sec` | `2` | Sliding window step size in seconds |

### Example with custom parameters

```bash
python carla_ds/generate_carla_vids.py --fps 20 --window_sec 5 --step_sec 2
```

This produces the following output structure, compatible with the inference script:

```
carla_ds/
  carla_vid_ds/
    front_view/  Norm_0001.mp4, Norm_0002.mp4, ...
    left_view/   Norm_0001.mp4, Norm_0002.mp4, ...
    right_view/  Norm_0001.mp4, Norm_0002.mp4, ...
    back_view/   Norm_0001.mp4, Norm_0002.mp4, ...
```

- Videos are encoded with H.264 (libx264) via ffmpeg.
- If **any** frame in a window has the `Anom_` prefix, the clip is labeled `Anom_`. Otherwise it is labeled `Norm_`.
- Clip names are indexed sequentially: `Norm_0001.mp4`, `Norm_0002.mp4`, etc.

## 4. Run Multiview Inference

```bash
python multiview_nobatch_fp16_inference.py --video_dir carla_ds/carla_vid_ds/
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--video_dir` | *(required)* | Root dir containing `front_view/`, `left_view/`, `right_view/`, `back_view/` |
| `--model` | `nvidia/Cosmos-Reason1-7B` | HuggingFace model name or local path |
| `--fps` | `4` | Target FPS for sampling frames from each video |
| `--max_tokens` | `3` | Maximum new tokens for model generation |
| `--target_resolution` | `250x250` | Target resolution for video frame processing |
| `--output_json` | `<video_dir>/fp16_multiview_results.json` | Path to save JSON results |

### Example with custom parameters

```bash
python multiview_nobatch_fp16_inference.py \
    --video_dir carla_ds/carla_vid_ds/ \
    --model nvidia/Cosmos-Reason1-7B \
    --fps 4 \
    --target_resolution 250x250 \
    --output_json results/carla_multiview.json
```

### How it works

For each synchronized clip set (e.g., `Norm_0001.mp4`):

1. Loads the same clip from all four view directories.
2. Builds a single conversation with interleaved view labels and videos:
   ```
   "Front view:" [front video] "Left view:" [left video] "Right view:" [right video] "Rear view:" [rear video] [prompt]
   ```
3. Runs a single forward pass through the model.
4. Parses the output as `Anomaly`, `Normal`, or `Unknown`.

### Output

Results are saved to a JSON file with:

- **config**: model, resolution, FPS, and GPU info
- **summary**: counts of anomalies, normals, unknowns, errors, and timing
- **metrics**: accuracy, precision, recall, F1-score (when ground-truth labels are available)
- **results**: per-clip predictions with raw model output and timing

Example console output:

```
Found 16 synchronized clip sets (4 views each) — running FP16 multiview inference
==================================================
[1/16] Norm_0001.mp4: Normal (Load: 1.23s, Inference: 2.45s) (raw: 'Classification: Normal')
[2/16] Norm_0002.mp4: Normal (Load: 1.18s, Inference: 2.31s) (raw: 'Classification: Normal')
...
```

## Quick Start (all steps)

```bash
# 1. Install dependencies
bash setup_fp16.sh

# 2. Download dataset
cd carla_ds/
wget -L -O camera.zip "https://nyu.box.com/shared/static/66khwx1o4owrzdt77jeb8cljjd2n8fxl"
unzip camera.zip -d camera_clean_individualframe
cd ..

# 3. Generate video clips
python carla_ds/generate_carla_vids.py

# 4. Run inference
python multiview_nobatch_fp16_inference.py --video_dir carla_ds/carla_vid_ds/
```
