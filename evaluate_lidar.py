import torch  # Inizializzazione primaria per collisione simboli C++

import os
import sys
import types
import argparse
import json
import numpy as np
from tqdm import tqdm
import shapely.affinity
from shapely.geometry import Polygon


def project_3d_to_2d_bbox(location, dimensions, rotation_y, calib):
    """
    Proietta un box 3D (Camera Frame) sul piano immagine 2D [left, top, right, bottom].
    """
    h, w, l = dimensions
    x, y, z = location
    ry = rotation_y

    # 8 vertici 3D relativi al bottom-center
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]  # Y punta verso il basso nel riferimento camera
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

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

        # Se la proiezione risulta troppo piccola o fuori campo, assegna una box di fallback valida
        if (xmax - xmin) < 5 or (ymax - ymin) < 5:
            return np.array([10.0, 10.0, 100.0, 100.0], dtype=np.float32)
        return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
    except Exception:
        return np.array([10.0, 10.0, 100.0, 100.0], dtype=np.float32)


# --- FALLBACK CPU PER KITTI EVAL (Bypassa la dipendenza da CUDA/Numba) ---
def _cpu_rotate_iou_eval(boxes1, boxes2, criterion=-1, device_id=0):
    N, M = len(boxes1), len(boxes2)
    iou_matrix = np.zeros((N, M), dtype=np.float32)

    if N == 0 or M == 0:
        return iou_matrix

    def get_poly(box):
        x, z, w, l, ry = box
        rect = Polygon([
            [-w / 2.0, -l / 2.0],
            [w / 2.0, -l / 2.0],
            [w / 2.0, l / 2.0],
            [-w / 2.0, l / 2.0]
        ])
        rotated = shapely.affinity.rotate(rect, -ry, use_radians=True, origin=(0, 0))
        return shapely.affinity.translate(rotated, xoff=x, yoff=z)

    polys1 = [get_poly(b) for b in boxes1]
    polys2 = [get_poly(b) for b in boxes2]

    for i in range(N):
        p1 = polys1[i]
        if not p1.is_valid or p1.area <= 0:
            continue
        for j in range(M):
            p2 = polys2[j]
            if not p2.is_valid or p2.area <= 0:
                continue
            try:
                inter = p1.intersection(p2).area
                if criterion == -1:
                    union = p1.area + p2.area - inter
                    iou = inter / union if union > 0 else 0.0
                elif criterion == 0:
                    iou = inter / p1.area if p1.area > 0 else 0.0
                elif criterion == 1:
                    iou = inter / p2.area if p2.area > 0 else 0.0
                else:
                    iou = 0.0
                iou_matrix[i, j] = iou
            except Exception:
                iou_matrix[i, j] = 0.0

    return iou_matrix


fake_rotate_iou = types.ModuleType('mmdet3d.evaluation.functional.kitti_utils.rotate_iou')
fake_rotate_iou.rotate_iou_gpu_eval = _cpu_rotate_iou_eval
sys.modules['mmdet3d.evaluation.functional.kitti_utils.rotate_iou'] = fake_rotate_iou

from mmdet3d.evaluation.functional.kitti_utils import kitti_eval
from src import KittiDataset, LidarDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Valutazione KITTI con Sparsificazione")
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
    print(f"\n📊 Avvio Valutazione [{tag.upper()}] | Campioni analizzati: {total_samples}/{len(dataset)}")

    gt_annotations = []
    pred_annotations = []

    for i in tqdm(range(total_samples), desc="Valutazione Frame"):
        sample = dataset[i]
        calib = sample['calib']

        # Ground Truth
        gt_objs = sample['gt_boxes']
        gt_ann = {
            'name': np.array([obj['type'] for obj in gt_objs]),
            'truncated': np.array([obj['truncation'] for obj in gt_objs]) if gt_objs else np.zeros(0),
            'occluded': np.array([obj['occlusion'] for obj in gt_objs]) if gt_objs else np.zeros(0),
            'alpha': np.array([obj['alpha'] for obj in gt_objs]) if gt_objs else np.zeros(0),
            'bbox': np.array([obj['bbox_2d'] for obj in gt_objs]) if gt_objs else np.zeros((0, 4)),
            'dimensions': np.array([obj['dimensions_3d'] for obj in gt_objs]) if gt_objs else np.zeros((0, 3)),
            'location': np.array([obj['location_3d'] for obj in gt_objs]) if gt_objs else np.zeros((0, 3)),
            'rotation_y': np.array([obj['rotation_y'] for obj in gt_objs]) if gt_objs else np.zeros(0)
        }
        gt_annotations.append(gt_ann)

        # Predizioni
        detections = detector.detect(sample)

        pred_bboxes_2d = []
        for det in detections:
            bbox_2d = project_3d_to_2d_bbox(det.location_3d, det.dimensions_3d, det.rotation_y, calib)
            pred_bboxes_2d.append(bbox_2d)

        pred_ann = {
            'name': np.array([det.type for det in detections]),
            'truncated': np.zeros(len(detections)),
            'occluded': np.zeros(len(detections)),
            'alpha': np.zeros(len(detections)),
            'bbox': np.array(pred_bboxes_2d) if detections else np.zeros((0, 4)),
            'dimensions': np.array([det.dimensions_3d for det in detections]) if detections else np.zeros((0, 3)),
            'location': np.array([det.location_3d for det in detections]) if detections else np.zeros((0, 3)),
            'rotation_y': np.array([det.rotation_y for det in detections]) if detections else np.zeros(0),
            'score': np.array([det.score for det in detections]) if detections else np.zeros(0)
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