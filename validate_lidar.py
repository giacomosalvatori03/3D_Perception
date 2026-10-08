import argparse
import json
import os
import numpy as np
import torch
from src import KittiDataset, LidarDetector
from tqdm import tqdm


def project_3d_to_2d_bbox(location, dimensions, rotation_y, calib):
    """Proietta un box 3D (Camera Frame) sul piano immagine 2D [xmin, ymin, xmax, ymax]."""
    h, w, l = dimensions
    x, y, z = location
    ry = rotation_y

    # CORRETTO: In Camera Frame l'asse X corrisponde alla larghezza (w) e Z alla lunghezza (l)
    x_corners = [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]  # y è la base inferiore
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

    # Filtro di sicurezza per oggetti dietro o troppo vicini alla fotocamera
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
    """Formatta la predizione nel formato riga 15-valori KITTI standard:

    type truncated occluded alpha bbox_2d(4) dimensions(3) location(3)
    rotation_y score
    """
    loc_x, loc_y, loc_z = det.location_3d
    h, w, l = det.dimensions_3d
    ry = det.rotation_y

    # Alpha (angolo di osservazione): alpha = ry - arctan2(x, z)
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
    """Formatta un oggetto Ground Truth nel formato KITTI standard."""
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
        description="Inferenza LiDAR e Salvataggio Predizioni KITTI"
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
        choices=["none", "random", "beam", "distance"],
    )
    parser.add_argument("--subsample_ratio", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=32)
    parser.add_argument("--max_distance", type=float, default=35.0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--conf_thresh", type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()

    # Tag dell'esperimento
    if args.subsample_mode == "random":
        tag = f"random_{int(args.subsample_ratio * 100)}perc"
    elif args.subsample_mode == "beam":
        tag = f"beam_{args.num_beams}rings"
    elif args.subsample_mode == "distance":
        tag = f"dist_{int(args.max_distance)}m"
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
        max_distance=args.max_distance,
    )
    detector = LidarDetector(conf_threshold=args.conf_thresh)

    total_samples = (
        len(dataset)
        if args.max_samples <= 0
        else min(args.max_samples, len(dataset))
    )
    print(
        f"\n📊 Avvio Inferenza PointPillars [{tag.upper()}] | Campioni: {total_samples}/{len(dataset)}"
    )
    print(f"💾 Destinazione file .txt su Drive: {exp_dir}")

    for i in tqdm(range(total_samples), desc="Elaborazione Frame"):
        sample = dataset[i]
        calib = sample["calib"]

        # Recupero ID del frame con formattazione a 6 cifre
        if hasattr(dataset, "sample_ids") and i < len(dataset.sample_ids):
            sample_id = str(dataset.sample_ids[i]).zfill(6)
        else:
            sample_id = f"{i:06d}"

        # 1. Salva Ground Truth .txt
        gt_objs = sample["gt_boxes"]
        gt_txt_path = os.path.join(gt_dir, f"{sample_id}.txt")
        with open(gt_txt_path, "w") as f_gt:
            for obj in gt_objs:
                f_gt.write(format_gt_line(obj))

        # 2. Inferenza PointPillars ed esportazione Predizioni .txt
        detections = detector.detect(sample)
        pred_txt_path = os.path.join(pred_dir, f"{sample_id}.txt")

        with open(pred_txt_path, "w") as f_pred:
            for det in detections:
                bbox_2d = project_3d_to_2d_bbox(
                    det.location_3d, det.dimensions_3d, det.rotation_y, calib
                )
                f_pred.write(format_kitti_line(det, bbox_2d))

    print(f"\n✅ Inferenza e salvataggio completati con successo!")
    print(f"  • Predizioni salvate in: {pred_dir}")
    print(f"  • Ground Truth salvate in: {gt_dir}")


if __name__ == "__main__":
    main()