import argparse
import json
import os
import numpy as np
import torch
from src import KittiDataset, LidarDetector
from tqdm import tqdm


def project_3d_to_2d_bbox(location, dimensions, rotation_y, calib):
    """Projects a 3D bounding box (Camera Frame) onto the 2D image plane [xmin, ymin, xmax, ymax]."""
    h, w, l = dimensions
    x, y, z = location
    ry = rotation_y

    # In Camera Frame, X axis is width (w) and Z axis is length (l)
    x_corners = [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]  # y is height (h) in Camera Frame
    z_corners = [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2]

    R = np.array(
        [
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)],
        ],
        dtype=np.float32,
    )

    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d = np.dot(R, corners_3d)
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z

    # Security filter for objects behind or too close to the camera
    if np.any(corners_3d[2, :] <= 0.1):
        return [10.0, 10.0, 100.0, 100.0]

    try:
        pts_2d = calib.rect2img(corners_3d.T)
        xmin = max(0.0, float(np.min(pts_2d[:, 0])))
        ymin = max(0.0, float(np.min(pts_2d[:, 1])))
        xmax = float(np.max(pts_2d[:, 0]))
        ymax = float(np.max(pts_2d[:, 1]))

        if (xmax - xmin) < 5 or (ymax - ymin) < 5:
            return [10.0, 10.0, 100.0, 100.0]
        return [xmin, ymin, xmax, ymax]
    except Exception:
        return [10.0, 10.0, 100.0, 100.0]


def format_kitti_line(det, bbox_2d):
    """Format prediction in standard KITTI 15-value row format:
    type truncated occluded alpha bbox_2d(4) dimensions(3) location(3) rotation_y score"""
    loc_x, loc_y, loc_z = det.location_3d
    h, w, l = det.dimensions_3d
    ry = det.rotation_y

    # Alpha (observational angle): alpha = ry - arctan2(x, z)
    alpha = ry - np.arctan2(loc_x, loc_z)
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

    xmin, ymin, xmax, ymax = bbox_2d

    return (
        f"{det.type} 0.00 0 {alpha:.2f} "
        f"{xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} "
        f"{h:.2f} {w:.2f} {l:.2f} "
        f"{loc_x:.2f} {loc_y:.2f} {loc_z:.2f} "
        f"{ry:.2f} {det.score:.4f}\n"
    )


def format_gt_line(obj):
    """Format Ground Truth object in standard KITTI format."""
    loc_x, loc_y, loc_z = obj["location_3d"]
    h, w, l = obj["dimensions_3d"]
    xmin, ymin, xmax, ymax = obj["bbox_2d"]

    return (
        f"{obj['type']} {obj['truncation']:.2f} {int(obj['occlusion'])} {obj['alpha']:.2f} "
        f"{xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} "
        f"{h:.2f} {w:.2f} {l:.2f} "
        f"{loc_x:.2f} {loc_y:.2f} {loc_z:.2f} "
        f"{obj['rotation_y']:.2f}\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="LiDAR Inference and Evaluation on KITTI Dataset with PointPillars"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/content/drive/MyDrive/3D_Perception/data/kitti_validation",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/content/drive/MyDrive/3D_Perception/experiments",
    )
    parser.add_argument(
        "--subsample_mode",
        type=str,
        default="none",
        choices=["none", "random", "beam"],
    )
    parser.add_argument("--subsample_ratio", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--conf_thresh", type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()

    # Experiment tag based on active subsampling strategy
    if args.subsample_mode == "random":
        tag = f"random_{int(args.subsample_ratio * 100)}perc"
    elif args.subsample_mode == "beam":
        tag = f"beam_{args.num_beams}rings"
    else:
        tag = "baseline_100perc"

    exp_dir = os.path.join(args.save_dir, tag)
    pred_dir = os.path.join(exp_dir, "pred_labels")
    gt_dir = os.path.join(exp_dir, "gt_labels")

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    dataset = KittiDataset(
        data_root=args.data_path,
        subsample_mode=args.subsample_mode,
        subsample_ratio=args.subsample_ratio,
        num_beams=args.num_beams,
    )
    detector = LidarDetector(conf_threshold=args.conf_thresh)

    total_samples = (
        len(dataset)
        if args.max_samples <= 0
        else min(args.max_samples, len(dataset))
    )
    print(
        f"\n Starting PointPillars Inference [{tag.upper()}] | Samples: {total_samples}/{len(dataset)}"
    )
    print(f" Destination for .txt files on Drive: {exp_dir}")

    for i in tqdm(range(total_samples), desc="Processing Frames"):
        sample = dataset[i]
        calib = sample["calib"]

        # Retrieve frame ID with 6-digit formatting
        if hasattr(dataset, "sample_ids") and i < len(dataset.sample_ids):
            sample_id = str(dataset.sample_ids[i]).zfill(6)
        else:
            sample_id = f"{i:06d}"

        # 1. Save Ground Truth .txt
        gt_objs = sample["gt_boxes"]
        gt_txt_path = os.path.join(gt_dir, f"{sample_id}.txt")
        with open(gt_txt_path, "w") as f_gt:
            for obj in gt_objs:
                f_gt.write(format_gt_line(obj))

        # 2. PointPillars Inference and Export Predictions .txt
        detections = detector.detect(sample)
        pred_txt_path = os.path.join(pred_dir, f"{sample_id}.txt")

        with open(pred_txt_path, "w") as f_pred:
            for det in detections:
                bbox_2d = project_3d_to_2d_bbox(
                    det.location_3d, det.dimensions_3d, det.rotation_y, calib
                )
                f_pred.write(format_kitti_line(det, bbox_2d))

    print(f"\n Inference and saving completed successfully!")
    print(f"  • Predictions saved in: {pred_dir}")
    print(f"  • Ground Truth saved in: {gt_dir}")


if __name__ == "__main__":
    main()