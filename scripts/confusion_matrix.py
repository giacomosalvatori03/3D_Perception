import glob
import os
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


class ConfusionMatrix3D:
    """Calculates and visualizes 3D Detection Confusion Matrices for KITTI predictions."""

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        iou_thresholds: Optional[dict] = None,
        conf_thresh: float = 0.25,
    ):
        """
        Args:
            class_names: List of target classes. Defaults to ['Car', 'Pedestrian', 'Cyclist'].
            iou_thresholds: Dict of IoU thresholds per class. Defaults to Car: 0.5, Pedestrian: 0.25, Cyclist: 0.25.
            conf_thresh: Minimum prediction confidence score to evaluate.
        """
        self.class_names = class_names or ["Car", "Pedestrian", "Cyclist"]
        self.matrix_classes = self.class_names + ["Background"]
        self.num_classes = len(self.class_names)
        
        self.iou_thresholds = iou_thresholds or {
            "Car": 0.50,
            "Pedestrian": 0.25,
            "Cyclist": 0.25,
            "default": 0.25,
        }
        self.conf_thresh = conf_thresh

    @staticmethod
    def _parse_kitti_txt(file_path: str, is_gt: bool = False, conf_thresh: float = 0.25) -> List[dict]:
        """Parses KITTI format prediction or ground truth text files."""
        boxes = []
        if not os.path.exists(file_path):
            return boxes

        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] == "DontCare":
                    continue

                cls_type = parts[0]
                h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
                x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
                ry = float(parts[14])
                score = float(parts[15]) if len(parts) >= 16 else 1.0

                if not is_gt and score < conf_thresh:
                    continue

                boxes.append({
                    "type": cls_type,
                    "location_3d": [x, y, z],
                    "dimensions_3d": [h, w, l],
                    "rotation_y": ry,
                    "score": score,
                })
        return boxes

    @staticmethod
    def _bev_box_to_polygon(box: dict) -> Polygon:
        """Converts 3D box center/dimensions/yaw to a 2D BEV Shapely Polygon."""
        x, _, z = box["location_3d"]
        _, w, l = box["dimensions_3d"]
        ry = box["rotation_y"]

        # Corners relative to center
        corners = np.array([
            [l / 2, w / 2],
            [l / 2, -w / 2],
            [-l / 2, -w / 2],
            [-l / 2, w / 2],
        ])

        # Rotation matrix around Y axis
        R = np.array([
            [np.cos(ry), np.sin(ry)],
            [-np.sin(ry), np.cos(ry)]
        ])

        rotated_corners = np.dot(corners, R)
        rotated_corners[:, 0] += x
        rotated_corners[:, 1] += z

        return Polygon(rotated_corners)

    def _compute_bev_iou(self, boxA: dict, boxB: dict) -> float:
        """Computes Bird's Eye View (BEV) 2D IoU between two 3D boxes."""
        try:
            polyA = self._bev_box_to_polygon(boxA)
            polyB = self._bev_box_to_polygon(boxB)

            if not polyA.is_valid or not polyB.is_valid:
                return 0.0

            inter = polyA.intersection(polyB).area
            union = polyA.area + polyB.area - inter
            return float(inter / union) if union > 0 else 0.0
        except Exception:
            return 0.0

    def compute_matrix(
        self,
        pred_dir: str,
        gt_dir: str,
    ) -> np.ndarray:
        """
        Calculates confusion matrix for an experiment directory.
        
        Returns:
            matrix: (N+1) x (N+1) numpy array where index N represents Background.
                    Rows = Ground Truth, Columns = Predictions.
        """
        matrix = np.zeros((self.num_classes + 1, self.num_classes + 1), dtype=np.int32)
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))

        for gt_path in gt_files:
            sid = os.path.basename(gt_path)
            pred_path = os.path.join(pred_dir, sid)

            gt_boxes = [b for b in self._parse_kitti_txt(gt_path, is_gt=True) if b["type"] in self.class_names]
            pred_boxes = [b for b in self._parse_kitti_txt(pred_path, is_gt=False, conf_thresh=self.conf_thresh) if b["type"] in self.class_names]

            matched_gt = set()
            matched_pred = set()

            # Compute IoU matrix
            iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
            for i, gt in enumerate(gt_boxes):
                for j, pred in enumerate(pred_boxes):
                    iou_matrix[i, j] = self._compute_bev_iou(gt, pred)

            # Greedy IoU matching
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                for _ in range(min(len(gt_boxes), len(pred_boxes))):
                    max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                    gt_idx, pred_idx = max_idx
                    max_iou = iou_matrix[gt_idx, pred_idx]

                    if max_iou <= 0.0:
                        break

                    gt_cls = gt_boxes[gt_idx]["type"]
                    pred_cls = pred_boxes[pred_idx]["type"]
                    req_iou = self.iou_thresholds.get(gt_cls, self.iou_thresholds["default"])

                    if max_iou >= req_iou:
                        gt_c_idx = self.class_names.index(gt_cls)
                        pred_c_idx = self.class_names.index(pred_cls)
                        matrix[gt_c_idx, pred_c_idx] += 1

                        matched_gt.add(gt_idx)
                        matched_pred.add(pred_idx)

                    # Zero out assigned row and column
                    iou_matrix[gt_idx, :] = -1.0
                    iou_matrix[:, pred_idx] = -1.0

            # Unmatched Ground Truth -> False Negatives (Background column)
            for i, gt in enumerate(gt_boxes):
                if i not in matched_gt:
                    gt_c_idx = self.class_names.index(gt["type"])
                    matrix[gt_c_idx, self.num_classes] += 1  # FN

            # Unmatched Predictions -> False Positives (Background row)
            for j, pred in enumerate(pred_boxes):
                if j not in matched_pred:
                    pred_c_idx = self.class_names.index(pred["type"])
                    matrix[self.num_classes, pred_c_idx] += 1  # FP

        return matrix

    def plot_matrix(
        self,
        matrix: np.ndarray,
        title: str = "3D Detection Confusion Matrix",
        normalize: bool = True,
        save_path: Optional[str] = None,
    ):
        """Plots and optionally saves the confusion matrix heatmap."""
        plt.figure(figsize=(8, 6.5))

        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            plot_matrix = (matrix / row_sums) * 100.0
            fmt = ".1f"
            cbar_label = "Percentage (%)"
        else:
            plot_matrix = matrix
            fmt = "d"
            cbar_label = "Count"

        im = plt.imshow(plot_matrix, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, label=cbar_label)

        ticks = np.arange(len(self.matrix_classes))
        plt.xticks(ticks, self.matrix_classes, rotation=45, ha="right", fontsize=11)
        plt.yticks(ticks, self.matrix_classes, fontsize=11)

        plt.title(title, fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Predicted Class", fontsize=12, labelpad=8)
        plt.ylabel("Ground Truth Class", fontsize=12, labelpad=8)

        # Annotate cells
        thresh = plot_matrix.max() / 2.0
        for i in range(plot_matrix.shape[0]):
            for j in range(plot_matrix.shape[1]):
                val = plot_matrix[i, j]
                text_str = f"{val:{fmt}}" + ("%" if normalize else "")
                plt.text(
                    j, i, text_str,
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=10, fontweight="bold"
                )

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f" Confusion Matrix saved to: {save_path}")

        plt.show()