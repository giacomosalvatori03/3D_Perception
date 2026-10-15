from typing import Any, Dict, List, Union
import numpy as np


class TraditionalFusion:
    """
    Geometric Late Fusion module for 3D perception.
    Projects 3D bounding boxes to 2D image plane, matches them with 2D YOLO
    detections, and updates 3D confidence scores based on multi-modal
    alignment.
    """

    def __init__(
        self,
        iou_thresh_2d: float = 0.30,
        alpha_boost: float = 0.25,
        final_conf_thresh: float = 0.25,
    ):
        """
        Args:
            iou_thresh_2d: Minimum 2D IoU required to match a projected 3D box with
            a 2D camera detection. alpha_boost: Scaling factor for confidence score
            boost: S_fused = S_3d + alpha * S_2d * IoU_2d. final_conf_thresh:
            Minimum confidence score required after fusion to retain a prediction.
        """
        self.iou_thresh_2d = iou_thresh_2d
        self.alpha_boost = alpha_boost
        self.final_conf_thresh = final_conf_thresh

    @staticmethod
    def project_3d_to_2d_bbox(
        location_3d: Union[List[float], np.ndarray],
        dimensions_3d: Union[List[float], np.ndarray],
        rotation_y: float,
        calib: Any,
        img_shape: tuple = (375, 1242),
    ) -> List[float]:
        """
        Projects a 3D bounding box (Camera Rectified Frame) onto the 2D image plane.
        Returns 2D bounding box [xmin, ymin, xmax, ymax].
        """
        h, w, l = dimensions_3d
        x, y, z = location_3d
        ry = rotation_y

        # 8 corners in Camera Rectified reference frame (Y is down, X is right, Z is front)
        x_corners = [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
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

        # Security filter: ignore objects behind or too close to camera
        if np.any(corners_3d[2, :] <= 0.1):
            return [0.0, 0.0, 0.0, 0.0]

        try:
            pts_2d = calib.rect2img(corners_3d.T)
            xmin = max(0.0, float(np.min(pts_2d[:, 0])))
            ymin = max(0.0, float(np.min(pts_2d[:, 1])))
            xmax = min(float(img_shape[1]), float(np.max(pts_2d[:, 0])))
            ymax = min(float(img_shape[0]), float(np.max(pts_2d[:, 1])))

            if (xmax - xmin) < 3 or (ymax - ymin) < 3:
                return [0.0, 0.0, 0.0, 0.0]

            return [xmin, ymin, xmax, ymax]
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def compute_iou_2d(boxA: List[float], boxB: List[float]) -> float:
        """Computes 2D Intersection over Union (IoU) between two boxes [xmin, ymin, xmax, ymax].
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        if interArea <= 0.0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        unionArea = boxAArea + boxBArea - interArea
        return float(interArea / unionArea) if unionArea > 0 else 0.0

    def fuse(
        self,
        predictions_3d: List[Union[Dict, Any]],
        detections_2d: List[Dict],
        calib: Any,
        filter_final: bool = True,
    ) -> List[Dict]:
        """Performs Late Fusion between 3D LiDAR predictions and 2D Camera detections.

        Args:
            predictions_3d: List of 3D detection dicts from LidarDetector.
            detections_2d: List of 2D detection dicts from CameraDetector.
            calib: Calibration object for the frame.
            filter_final: If True, filters out fused predictions with score <
              self.final_conf_thresh.

        Returns:
            fused_predictions: List of updated 3D detection dictionaries.
        """
        fused_predictions = []

        for pred in predictions_3d:
            # Standardize attribute access for dicts or objects
            if isinstance(pred, dict):
                loc = pred["location_3d"]
                dim = pred["dimensions_3d"]
                ry = pred["rotation_y"]
                score_3d = float(pred["score"])
                cls_type = pred["type"]
            else:
                loc = pred.location_3d
                dim = pred.dimensions_3d
                ry = pred.rotation_y
                score_3d = float(pred.score)
                cls_type = pred.type

            # Step 1: Project 3D box onto 2D image plane
            bbox_2d_proj = self.project_3d_to_2d_bbox(loc, dim, ry, calib)

            best_iou = 0.0
            best_det_2d = None

            # Step 2: Match with 2D Camera detections of the same class
            if bbox_2d_proj != [0.0, 0.0, 0.0, 0.0] and detections_2d:
                for det_2d in detections_2d:
                    if det_2d["type"] == cls_type:
                        iou = self.compute_iou_2d(
                            bbox_2d_proj, det_2d["bbox_2d"]
                        )
                        if iou > best_iou:
                            best_iou = iou
                            best_det_2d = det_2d

            # Step 3: Rescoring / Confidence boosting
            if best_iou >= self.iou_thresh_2d and best_det_2d is not None:
                score_2d = float(best_det_2d["score"])
                score_boost = self.alpha_boost * score_2d * best_iou
                fused_score = min(1.0, score_3d + score_boost)
                matched_2d = True
            else:
                fused_score = score_3d
                matched_2d = False

            fused_pred = {
                "type": cls_type,
                "location_3d": loc,
                "dimensions_3d": dim,
                "rotation_y": ry,
                "score": fused_score,
                "bbox_2d": bbox_2d_proj,
                "score_3d_raw": score_3d,
                "matched_2d": matched_2d,
                "iou_2d": best_iou,
            }

            if not filter_final or fused_score >= self.final_conf_thresh:
                fused_predictions.append(fused_pred)

        # Sort fused predictions by score descending
        fused_predictions.sort(key=lambda x: x["score"], reverse=True)
        return fused_predictions