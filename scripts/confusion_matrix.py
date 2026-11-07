import glob
import os
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon


class ConfusionMatrix3D:
    """Calculates and visualizes 3D Detection Confusion Matrices for KITTI predictions."""

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        iou_thresholds: Optional[dict] = None,
        conf_thresh: float = 0.25,
    ):
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
    def _parse_kitti_txt(file_path: str, is_gt: bool = False, conf_thresh: float = 0.25) -> Tuple[List[dict], List[dict]]:
        """Parses KITTI format files returning target boxes and DontCare regions."""
        boxes = []
        dontcare_boxes = []
        if not os.path.exists(file_path):
            return boxes, dontcare_boxes

        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                cls_type = parts[0]
                h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
                x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
                ry = float(parts[14])
                score = float(parts[15]) if len(parts) >= 16 else 1.0

                box_dict = {
                    "type": cls_type,
                    "location_3d": [x, y, z],
                    "dimensions_3d": [h, w, l],
                    "rotation_y": ry,
                    "score": score,
                }

                if cls_type == "DontCare":
                    dontcare_boxes.append(box_dict)
                else:
                    if not is_gt and score < conf_thresh:
                        continue
                    boxes.append(box_dict)
                    
        return boxes, dontcare_boxes

    @staticmethod
    def _bev_box_to_polygon(box: dict) -> Polygon:
        """Converts 3D box center/dimensions/yaw to a 2D BEV Shapely Polygon."""
        x, _, z = box["location_3d"]
        _, w, l = box["dimensions_3d"]
        ry = box["rotation_y"]

        # Vertex in the X-Z plane: width along X, length along Z
        corners = np.array([
            [w / 2, l / 2],
            [w / 2, -l / 2],
            [-w / 2, -l / 2],
            [-w / 2, l / 2],
        ])

        # Rotation matrix for yaw (ry) around Y-axis in BEV (X-Z plane)
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

    def compute_matrix(self, pred_dir: str, gt_dir: str) -> np.ndarray:
        matrix = np.zeros((self.num_classes + 1, self.num_classes + 1), dtype=np.int32)
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))

        for gt_path in gt_files:
            sid = os.path.basename(gt_path)
            pred_path = os.path.join(pred_dir, sid)

            gt_boxes, dontcares = self._parse_kitti_txt(gt_path, is_gt=True)
            gt_boxes = [b for b in gt_boxes if b["type"] in self.class_names]
            
            pred_boxes, _ = self._parse_kitti_txt(pred_path, is_gt=False, conf_thresh=self.conf_thresh)
            pred_boxes = [b for b in pred_boxes if b["type"] in self.class_names]

            # Filtering predictions that fall within DontCare regions
            filtered_preds = []
            for pred in pred_boxes:
                in_dontcare = False
                for dc in dontcares:
                    if self._compute_bev_iou(pred, dc) > 0.10:
                        in_dontcare = True
                        break
                if not in_dontcare:
                    filtered_preds.append(pred)
            pred_boxes = filtered_preds

            if not gt_boxes and not pred_boxes:
                continue

            matched_gt = set()
            matched_pred = set()

            # 1. Populates the IoU matrix, keeping only valid matches between the same class or valid ones
            iou_matrix = np.full((len(gt_boxes), len(pred_boxes)), -1.0, dtype=np.float32)
            for i, gt in enumerate(gt_boxes):
                req_iou = self.iou_thresholds.get(gt["type"], self.iou_thresholds["default"])
                for j, pred in enumerate(pred_boxes):
                    iou_val = self._compute_bev_iou(gt, pred)
                    # Allow matching only if it exceeds the GT class threshold
                    if iou_val >= req_iou:
                        iou_matrix[i, j] = iou_val

            # 2. Greedy matching algorithm: iteratively find the highest IoU and match them, then remove them from consideration
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                while True:
                    max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                    gt_idx, pred_idx = max_idx
                    max_iou = iou_matrix[gt_idx, pred_idx]

                    # No valid matches left
                    if max_iou <= 0.0:
                        break

                    gt_cls = gt_boxes[gt_idx]["type"]
                    pred_cls = pred_boxes[pred_idx]["type"]

                    gt_c_idx = self.class_names.index(gt_cls)
                    pred_c_idx = self.class_names.index(pred_cls)
                    matrix[gt_c_idx, pred_c_idx] += 1

                    matched_gt.add(gt_idx)
                    matched_pred.add(pred_idx)

                    # Zeroing out the matched row and column to avoid re-matching
                    iou_matrix[gt_idx, :] = -1.0
                    iou_matrix[:, pred_idx] = -1.0

            # GT non matched are considered False Negatives, incrementing the Background column for their class
            for i, gt in enumerate(gt_boxes):
                if i not in matched_gt:
                    gt_c_idx = self.class_names.index(gt["type"])
                    matrix[gt_c_idx, self.num_classes] += 1

            # Incrementing the Background row for their predicted class
            for j, pred in enumerate(pred_boxes):
                if j not in matched_pred:
                    pred_c_idx = self.class_names.index(pred["type"])
                    matrix[self.num_classes, pred_c_idx] += 1

        return matrix

    def plot_matrix(
        self,
        matrix: np.ndarray,
        title: str = "3D Detection Confusion Matrix",
        normalize: bool = True,
        save_path: Optional[str] = None,
    ):
        plt.figure(figsize=(8, 6.5))

        if normalize:
            plot_matrix = matrix.astype(np.float32).copy()
            # Normalize the first 3 rows (GT) with respect to the total GT per row
            for i in range(self.num_classes):
                total_gt = matrix[i, :].sum()
                if total_gt > 0:
                    plot_matrix[i, :] = (matrix[i, :] / total_gt) * 100.0
                else:
                    plot_matrix[i, :] = 0.0

            # Normalize the last row (Background) with respect to the total predictions per column
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

        thresh = plot_matrix.max() / 2.0
        for i in range(plot_matrix.shape[0]):
            for j in range(plot_matrix.shape[1]):
                val = plot_matrix[i, j]
                text_str = f"{val:{fmt}}" + ("%" if normalize and i < self.num_classes else "")
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
            print(f"Confusion Matrix saved to: {save_path}")

        plt.show()