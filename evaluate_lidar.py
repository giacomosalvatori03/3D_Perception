import os
import argparse
import json
import numpy as np
from tqdm import tqdm
import torch

from mmdet3d.evaluation.functional.kitti_utils import kitti_eval
from src import KittiDataset, LidarDetector


def project_3d_to_2d_bbox(location, dimensions, rotation_y, calib):
    """Proietta un box 3D (Camera Frame) sul piano immagine 2D [xmin, ymin, xmax, ymax]."""
    h, w, l = dimensions
    x, y, z = location
    ry = rotation_y

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ], dtype=np.float32)

    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d = np.dot(R, corners_3d)
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z

    try:
        pts_2d = calib.rect2img(corners_3d.T)
        xmin = float(np.min(pts_2d[:, 0]))
        ymin = float(np.min(pts_2d[:, 1]))
        xmax = float(np.max(pts_2d[:, 0]))
        ymax = float(np.max(pts_2d[:, 1]))

        if (xmax - xmin) < 5 or (ymax - ymin) < 5:
            return np.array([10.0, 10.0, 100.0, 100.0], dtype=np.float32)
        return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
    except Exception:
        return np.array([10.0, 10.0, 100.0, 100.0], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Valutazione KITTI Nativa CUDA mmdet3d")
    parser.add_argument("--data_path", type=str, default="/content/drive/MyDrive/3D_Perception/data/kitti_validation")
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/3D_Perception/experiments")
    parser.add_argument("--subsample_mode", type=str, default="none", choices=["none", "random", "beam", "distance"])
    parser.add_argument("--subsample_ratio", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=32)
    parser.add_argument("--max_distance", type=float, default=35.0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--conf_thresh", type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = KittiDataset(
        data_root=args.data_path,
        pts_dir='velodyne_reduced',
        subsample_mode=args.subsample_mode,
        subsample_ratio=args.subsample_ratio,
        num_beams=args.num_beams,
        max_distance=args.max_distance
    )
    detector = LidarDetector(conf_threshold=args.conf_thresh)

    if args.subsample_mode == 'random':
        tag = f"random_{int(args.subsample_ratio * 100)}perc"
    elif args.subsample_mode == 'beam':
        tag = f"beam_{args.num_beams}rings"
    elif args.subsample_mode == 'distance':
        tag = f"dist_{int(args.max_distance)}m"
    else:
        tag = "baseline_100perc"

    total_samples = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))
    print(f"\n📊 Avvio Valutazione Nativa CUDA [{tag.upper()}] | Campioni: {total_samples}/{len(dataset)}")

    gt_annotations = []
    pred_annotations = []

    for i in tqdm(range(total_samples), desc="Valutazione Frame"):
        sample = dataset[i]
        calib = sample['calib']

        # Ground Truth (con formattazione rigida float32 / int32 per C++)
        gt_objs = sample['gt_boxes']
        gt_ann = {
            'name': np.array([obj['type'] for obj in gt_objs]),
            'truncated': np.ascontiguousarray([obj['truncation'] for obj in gt_objs], dtype=np.float32) if gt_objs else np.zeros(0, dtype=np.float32),
            'occluded': np.ascontiguousarray([obj['occlusion'] for obj in gt_objs], dtype=np.int32) if gt_objs else np.zeros(0, dtype=np.int32),
            'alpha': np.ascontiguousarray([obj['alpha'] for obj in gt_objs], dtype=np.float32) if gt_objs else np.zeros(0, dtype=np.float32),
            'bbox': np.ascontiguousarray([obj['bbox_2d'] for obj in gt_objs], dtype=np.float32) if gt_objs else np.zeros((0, 4), dtype=np.float32),
            'dimensions': np.ascontiguousarray([obj['dimensions_3d'] for obj in gt_objs], dtype=np.float32) if gt_objs else np.zeros((0, 3), dtype=np.float32),
            'location': np.ascontiguousarray([obj['location_3d'] for obj in gt_objs], dtype=np.float32) if gt_objs else np.zeros((0, 3), dtype=np.float32),
            'rotation_y': np.ascontiguousarray([obj['rotation_y'] for obj in gt_objs], dtype=np.float32) if gt_objs else np.zeros(0, dtype=np.float32)
        }
        gt_annotations.append(gt_ann)

        # Predizioni (Inferenza GPU)
        detections = detector.detect(sample)

        pred_bboxes_2d = []
        for det in detections:
            bbox_2d = project_3d_to_2d_bbox(det.location_3d, det.dimensions_3d, det.rotation_y, calib)
            pred_bboxes_2d.append(bbox_2d)

        pred_ann = {
            'name': np.array([det.type for det in detections]),
            'truncated': np.zeros(len(detections), dtype=np.float32),
            'occluded': np.zeros(len(detections), dtype=np.int32),
            'alpha': np.zeros(len(detections), dtype=np.float32),
            'bbox': np.ascontiguousarray(pred_bboxes_2d, dtype=np.float32) if detections else np.zeros((0, 4), dtype=np.float32),
            'dimensions': np.ascontiguousarray([det.dimensions_3d for det in detections], dtype=np.float32) if detections else np.zeros((0, 3), dtype=np.float32),
            'location': np.ascontiguousarray([det.location_3d for det in detections], dtype=np.float32) if detections else np.zeros((0, 3), dtype=np.float32),
            'rotation_y': np.ascontiguousarray([det.rotation_y for det in detections], dtype=np.float32) if detections else np.zeros(0, dtype=np.float32),
            'score': np.ascontiguousarray([det.score for det in detections], dtype=np.float32) if detections else np.zeros(0, dtype=np.float32)
        }
        pred_annotations.append(pred_ann)

    classes = ['Car', 'Pedestrian', 'Cyclist']
    result_str, ret_dict = kitti_eval(gt_annotations, pred_annotations, classes)

    print("\n" + result_str)

    txt_path = os.path.join(args.save_dir, f"report_{tag}.txt")
    json_path = os.path.join(args.save_dir, f"metrics_{tag}.json")

    with open(txt_path, "w") as f:
        f.write(result_str)

    with open(json_path, "w") as f:
        clean_ret_dict = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in ret_dict.items()}
        json.dump(clean_ret_dict, f, indent=4)

    print(f"✅ Risultati salvati su Drive:\n  • {txt_path}\n  • {json_path}")


if __name__ == "__main__":
    main()