# STU Dataset - Video Processing for Anomaly Detection

This module provides utilities for processing the STU (Semantic Traffic Understanding) dataset, which contains LiDAR point clouds and camera images for autonomous driving anomaly detection.

## Dataset

Download the dataset from HuggingFace:
- Download `my_val.tar.gz` from https://huggingface.co/datasets/danieladejumo/stu_dataset/tree/main
- Extract the tar file into `stu_dataset/data/` directory:
  ```bash
  mkdir -p data
  tar -xzf my_val.tar.gz -C data/
  ```

## Environment Setup

### Option 1: Use the included virtual environment

```bash
# Activate the virtual environment
source .stu_venv/bin/activate

# Verify installation
python3 -c "import torch, cv2, numpy, open3d, PIL, scipy; print('All imports OK')"
```

### Option 2: Install in your own environment

```bash
pip install -r requirements.txt
```

## Usage

### STUDataset Class

Load and process individual frames from the dataset:

```python
from stu_video_dataset import STUDataset

# Load a single scene
ds = STUDataset("/path/to/scene_folder", single_scene=True)

# Get a sample (image, labels, file_path)
image, labels, image_file = ds[0]
```

### Video Dataloader

Generate videos from the dataset with sliding window:

```python
from stu_video_dataset import stu_video_dataloader

# Iterate over videos with labels
for video_path, has_anomaly, images in stu_video_dataloader(
    dataset_root="/path/to/data",
    window_size=50,
    step_size=20,
    fps=10,
    output_dir="./output_videos/"
):
    print(f"Video: {video_path}, Anomaly: {has_anomaly}")
```

### Metrics Tracking

Track prediction performance:

```python
from stu_video_dataset import Metrics

metrics = Metrics()
metrics.update(predictions=[1, 0, 1], targets=[1, 0, 0], inference_times=[0.1, 0.1, 0.1])
results = metrics.compute()
print(results)  # {'TP': 1, 'TN': 1, 'FP': 1, 'FN': 0, 'Accuracy': 0.67, ...}
```

## Data Format

- **Point clouds**: `velodyne/*.bin` - Nx4 float32 (x, y, z, intensity)
- **Labels**: `labels/*.label` - uint32 (semantic_label & 0xFFFF, instance_label >> 16)
- **Images**: `port_a_cam_0/*.png` - 1920x1208 RGB images

### Label Classes
- `0`: Ignore
- `1`: Inlier (normal)
- `2`: Anomaly

## Dependencies

- torch >= 2.0.0
- opencv-python-headless >= 4.8.0
- numpy >= 1.24.0
- open3d >= 0.18.0
- pillow >= 10.0.0
- scipy >= 1.10.0
