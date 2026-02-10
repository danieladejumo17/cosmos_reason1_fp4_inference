# STU DATASET


from pathlib import Path
import torch

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


# Normalize intensities
def normalize_intensities(intensities):
    return np.clip(intensities / 255, 0, 1)


# Filter points that are inside a given polygon
def filter_points_in_polygon(image_points, polygon, corresponding_3d_points):
    path = Path(polygon)
    inside = path.contains_points(image_points)
    return image_points[inside], corresponding_3d_points[inside], inside


# Save colored point cloud as a PCD file
def save_colored_pcd(points, colors, output_pcd_file):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_pcd_file, pcd)
    print(f"Saved colored PCD at: {output_pcd_file}")


def load_point_cloud(scan_path):
    scan = np.fromfile(scan_path, dtype=np.float32)
    scan = scan.reshape(
        (-1, 4)
    )  # The point cloud data is stored in a Nx4 format (x, y, z, intensity)
    points = scan[:, :3]  # Extracting the (x, y, z) coordinates
    intensities = scan[:, 3:]  # Extracting the (x, y, z) coordinates
    return (points, intensities)


def load_labels(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32).astype(np.int32)
    semantic_label = labels & 0xFFFF
    instance_label = labels >> 16
    return semantic_label, instance_label


def project_points_pinhole(points, camera_matrix, dist_coeffs):
    if points.size == 0:
        return (np.array([]), np.array([]))
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))
    image_points, _ = cv2.projectPoints(
        points.reshape(-1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    image_points = image_points.reshape(-1, 2)
    return image_points


def transform_points(points, T):
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed = (T @ points_hom.T).T[:, :3]
    return points_transformed


def get_image_labels(
    base_path,
    idx,
    image_width,
    image_height,
    camera_matrix,
    dist_coeffs,
    translation,
    yaw,
    pitch,
    roll,
):
    label_file = base_path / f"labels/{idx:06d}.label"
    point_file = base_path / f"velodyne/{idx:06d}.bin"
    # 
    label_file = Path(label_file)
    labels, _ = load_labels(label_file)
    points, _ = load_point_cloud(point_file)
    # 
    labels[(labels != 2) & (labels != 0)] = 1
    # 
    # Construct the transformation matrix
    r = R.from_euler("ZYX", [yaw, pitch, roll])
    rotation_matrix = r.as_matrix()
    transformation_inv = np.eye(4)
    transformation_inv[:3, :3] = rotation_matrix.T
    transformation_inv[:3, 3] = -np.dot(rotation_matrix.T, translation)
    # 
    # Transform points into the camera frame
    points_transformed = transform_points(points, transformation_inv)
    # 
    # Use only points in front of the camera (z > 0)
    valid_indices = points_transformed[:, 2] > 0
    points_camera_valid = points_transformed[valid_indices]
    valid_labels = labels[valid_indices]
    # 
    # Project to image plane
    image_points = project_points_pinhole(
        points_camera_valid, camera_matrix, dist_coeffs
    )
    # 
    # Convert projected points to integer coordinates
    pts_int = image_points.astype(int)
    # 
    # Filter points that lie within image bounds
    inside_mask = (
        (pts_int[:, 0] >= 0)
        & (pts_int[:, 0] < image_width)
        & (pts_int[:, 1] >= 0)
        & (pts_int[:, 1] < image_height)
    )
    pts_in = pts_int[inside_mask]
    valid_labels = valid_labels[inside_mask]
    # 
    # return pts_in, valid_labels, image
    return np.hstack((pts_in, valid_labels[:, None]))


# class STUDataset(Dataset):
class STUDataset():
    def __init__(self, base_path, offset=0, transform=None, single_scene=False):
        self.base_path = Path(base_path)
        self.transform = transform
        self.offset = offset
        if single_scene: # base_path/velodyne/*.bin
            self.data = sorted(list(self.base_path.glob("velodyne/*.bin")))
        else: # base_path/101/velodyne/*.bin
            self.data = sorted(list(self.base_path.glob("*/velodyne/*.bin")))
        # 
        self.image_width = 1920
        self.image_height = 1208
        self.camera_matrix = np.array(
            [
                [1827.48989, 0.0, 925.91346],
                [0.0, 1835.88358, 642.07154],
                [0.0, 0.0, 1.0],
            ]
        )
        self.dist_coeffs = np.array([-0.260735, 0.046071, 0.001173, -0.000154, 0.0])
        self.translation = np.array([0.7658, 0.0124, -0.3925])
        self.yaw = -1.5599
        self.pitch = 0.0188
        self.roll = -1.5563
    # 
    def __len__(self):
        return len(self.data)
    # 
    def __getitem__(self, idx):
        # 2 - anomaly
        # 0 - ignore
        # 1 - inlier
        base_path = self.data[idx].parent.parent
        idx = int(self.data[idx].stem)
        image_file = str(base_path / "port_a_cam_0" / (f"{idx:06d}" + ".png"))
        image = Image.open(image_file)  # BGR format by default
        # 
        # Apply optional transformation
        if self.transform:
            image = self.transform(image)
        # 
        label = get_image_labels(
            base_path,
            idx,
            image.size[0],
            image.size[1],
            self.camera_matrix,
            self.dist_coeffs,
            self.translation,
            self.yaw,
            self.pitch,
            self.roll,
        )
        # 
        return image, label, image_file

    def get_predictions_targets(self, uncertainty, target):
        coords = target.squeeze(0)

        # Separate indices
        x_indices = coords[:, 0]  # Width (columns)
        y_indices = coords[:, 1]  # Height (rows)
        labels_1 = coords[
            :, 2
        ]  # Label (0 = negative, 1 = positive, 2 = ignored)

        x_indices = torch.clamp(x_indices, 0, self.image_width - 1)
        y_indices = torch.clamp(y_indices, 0, self.image_height - 1)

        # Sample pixel values
        sampled_values = uncertainty[y_indices, x_indices]

        # Separate into categories
        uncertainty = sampled_values[labels_1 != 0]  # Pixels with label 0
        labels = labels_1[labels_1 != 0] - 1
        return uncertainty, labels
    




# PROCESS CHUNKS

def process_ds_chunk(chunk, anomaly_label=2):
    # chunk: [(image, label, image_file_path), ...]
    # return ([images], single_label)
    # extract images and labels    
    images = [item[0] for item in chunk]
    labels = [item[1] for item in chunk]
    # 
    # process labels
    # labels: [[[x1, y1, label1], [x2, y2, label2], ...], ...]
    # extract a single lable for each [x1, y1, label1], [x2, y2, label2], ...] in the labels list
    # flaten labels on second axis
    # 
    processed_labels = [np.any(label[:, 2] == anomaly_label) for label in labels]
    # processed_labels = [
    # for label in labels:
        # processed_labels.append(np.any(label[:, 2] == anomaly_label))
    # processed_labels = np.any(labels[:, :, 2] == anomaly_label, axis=1)
    # 
    return images, processed_labels # images: [PIL Image], processed_labels: [np.boolean]

def video_from_images(image_list, output_path, fps=10):
    if len(image_list) == 0:
        print("No images to create video.")
        return
    # 
    # Get dimensions from the first image
    first_image = image_list[0]
    width, height = first_image.size
    # 
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # 
    for img in image_list:
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video_writer.write(img_cv)
    # 
    video_writer.release()
    # print(f"Video saved at {output_path}")

def iter_ds(ds, window_size=50, step_size=20, video_output_path="output_video.mp4", fps=10, separate_videos=False):
    if window_size > len(ds):
        images, labels = process_ds_chunk(ds)
        # create a video from images
        video_from_images(images, video_output_path, fps=fps)
        yield video_output_path, np.any(labels), images
        return
    # 
    ret, ret_labels = process_ds_chunk([ds[i] for i in range(window_size - step_size)]) # get the first window_size - step_size frames
    first_sample = True
    for i in range(window_size - step_size, len(ds), step_size):
        # get the next step_size frames and labels
        ext = [ds[i + x] for x in range(step_size) if (i + x) < len(ds)]
        images, labels = process_ds_chunk(ext)
        #   
        # update ret and ret_labels
        ret.extend(images)
        ret_labels.extend(labels)
        # 
        # create a video from images
        if separate_videos:
            out_path = Path(video_output_path)
            file_stem = f"{out_path.stem}_{i}"
            file_suffix = out_path.suffix
            filename = file_stem + file_suffix
            out_path_video = out_path.parent / filename
            print("Writing video to", out_path_video)
        else:
            out_path_video = video_output_path
    

        video_from_images(ret, f"{out_path_video}", fps=fps)
        if first_sample:
            first_sample = False
            yield out_path_video, np.any(ret_labels), ret # generated video path, label, new images
        else:    
            yield out_path_video, np.any(ret_labels), images # generated video path, label, new images
        # 
        # shift the window
        ret = ret[step_size:]  # slide the window by step_size
        ret_labels = ret_labels[step_size:]

        # if first yield return ret, otherwise return ext



def stu_video_dataloader(dataset_root, window_size=50, step_size=20, fps=10, separate_videos=False, output_dir="./output_videos/"):
    dataset_root = Path(dataset_root)
    
    # create output dir if not exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Print full path of output dir
    print(f"Output videos will be saved to: {output_dir.resolve()}")

    # iterate over each folder in dataset_root
    for folder in sorted(dataset_root.iterdir()):
        if folder.is_dir():
            print(f"Processing folder: {folder.name}")
            ds = STUDataset(folder, single_scene=True)
            ds_iter = iter_ds(ds, window_size=window_size, step_size=step_size, video_output_path=Path(output_dir)/("scene_"+folder.name + ".mp4"), fps=fps, separate_videos=separate_videos)
            for video_path, label, images in ds_iter:
                yield video_path, label, images # generated video path, label, new images


# Define a metrics object for keeping track of the predictions and targets and computing TP, TN, FP, FN, accuracy, precision, recall, F1-score
class Metrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.total_time = 0
        self.count = 0


    def update(self, predictions, targets, inference_times):
        for p, t, inf_time in zip(predictions, targets, inference_times):
            if p == 1 and t == 1:
                self.tp += 1
            elif p == 0 and t == 0:
                self.tn += 1
            elif p == 1 and t == 0:
                self.fp += 1
            elif p == 0 and t == 1:
                self.fn += 1

            self.total_time += inf_time
            self.count += 1

    def compute(self):
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn) if (self.tp + self.tn + self.fp + self.fn) > 0 else 0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        avg_inference_time = self.total_time / self.count if self.count > 0 else 0
        return {
            "TP": self.tp,
            "TN": self.tn,
            "FP": self.fp,
            "FN": self.fn,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score,
            "Avg Inference Time": avg_inference_time
        }