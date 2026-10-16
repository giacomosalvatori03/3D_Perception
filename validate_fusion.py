import glob
import os
import shutil
import numpy as np
from src.calibration import Calibration
from src.traditional_fusion import TraditionalFusion
from tqdm import tqdm


class FusionValidator:
    """
    Validator that loads pre-computed 3D LiDAR and 2D Camera prediction
    files, executes Fusion module, and exports fused KITTI-formatted label
    files.
    """

    def __init__(
        self,
        data_path: str = "/content/drive/MyDrive/3D_Perception/data/kitti_validation",
        iou_thresh_2d: float = 0.25,
        alpha_boost: float = 0.25,
        final_conf_thresh: float = 0.25,
    ):
        self.data_path = data_path
        self.fusion_module = TraditionalFusion(
            iou_thresh_2d=iou_thresh_2d,
            alpha_boost=alpha_boost,
            final_conf_thresh=final_conf_thresh,
        )

    @staticmethod
    def _parse_3d_txt(file_path: str) -> list:
        preds = []
        if not os.path.exists(file_path):
            return preds
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] == "DontCare":
                    continue
                preds.append({
                    "type": parts[0],
                    "location_3d": [
                        float(parts[11]),
                        float(parts[12]),
                        float(parts[13]),
                    ],
                    "dimensions_3d": [
                        float(parts[8]),
                        float(parts[9]),
                        float(parts[10]),
                    ],
                    "rotation_y": float(parts[14]),
                    "score": float(parts[15]) if len(parts) >= 16 else 1.0,
                })
        return preds

    @staticmethod
    def _parse_2d_txt(file_path: str) -> list:
        preds = []
        if not os.path.exists(file_path):
            return preds
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] == "DontCare":
                    continue
                score = (
                    float(parts[15])
                    if len(parts) >= 16
                    else (float(parts[14]) if len(parts) == 15 else 1.0)
                )
                preds.append({
                    "type": parts[0],
                    "bbox_2d": [
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                        float(parts[7]),
                    ],
                    "score": score,
                })
        return preds

    @staticmethod
    def _format_fused_line(det: dict) -> str:
        loc_x, loc_y, loc_z = det["location_3d"]
        h, w, l = det["dimensions_3d"]
        ry = det["rotation_y"]
        score = det["score"]
        cls_type = det["type"]
        xmin, ymin, xmax, ymax = det.get("bbox_2d", [10.0, 10.0, 100.0, 100.0])

        alpha = ry - np.arctan2(loc_x, loc_z)
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

        return (
            f"{cls_type} 0.00 0 {alpha:.2f} "
            f"{xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} "
            f"{h:.2f} {w:.2f} {l:.2f} "
            f"{loc_x:.2f} {loc_y:.2f} {loc_z:.2f} "
            f"{ry:.2f} {score:.4f}\n"
        )

    def run_fusion(
        self,
        lidar_exp_dir: str,
        camera_exp_dir: str,
        output_exp_dir: str,
    ):
        """
        Loads 3D LiDAR and 2D Camera prediction files, runs
        Fusion module, and exports the resulting 3D predictions along
        with Ground Truth labels for evaluation.
        """
        lidar_pred_dir = os.path.join(lidar_exp_dir, "pred_labels")
        camera_pred_dir = os.path.join(camera_exp_dir, "pred_labels")
        gt_source_dir = os.path.join(lidar_exp_dir, "gt_labels")

        out_pred_dir = os.path.join(output_exp_dir, "pred_labels")
        out_gt_dir = os.path.join(output_exp_dir, "gt_labels")

        os.makedirs(out_pred_dir, exist_ok=True)
        os.makedirs(out_gt_dir, exist_ok=True)

        lidar_files = sorted(glob.glob(os.path.join(lidar_pred_dir, "*.txt")))
        print(f"\n Executing the fusion on {len(lidar_files)} frames...")

        for l_path in tqdm(lidar_files, desc="Fusing Detections"):
            sid = os.path.basename(l_path).replace(".txt", "")

            # Copy Ground Truth files to the output directory for direct evaluation
            gt_src = os.path.join(gt_source_dir, f"{sid}.txt")
            if os.path.exists(gt_src):
                shutil.copy(gt_src, os.path.join(out_gt_dir, f"{sid}.txt"))

            # Load 3D LiDAR predictions, 2D Camera predictions, and calibration data
            preds_3d = self._parse_3d_txt(l_path)
            preds_2d = self._parse_2d_txt(
                os.path.join(camera_pred_dir, f"{sid}.txt")
            )
            calib = Calibration(
                os.path.join(self.data_path, "calib", f"{sid}.txt")
            )

            # Call the fusion module to combine 3D and 2D predictions
            fused_preds = self.fusion_module.fuse(
                predictions_3d=preds_3d,
                detections_2d=preds_2d,
                calib=calib,
                filter_final=True,
            )

            # Save the fused predictions in KITTI format to the output directory
            out_txt_path = os.path.join(out_pred_dir, f"{sid}.txt")
            with open(out_txt_path, "w") as f_out:
                for det in fused_preds:
                    f_out.write(self._format_fused_line(det))

        print(
            f" Fusion completed! Output labels stored in: {output_exp_dir}"
        )