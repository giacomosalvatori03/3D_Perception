from typing import Any, Dict, List, Union
import numpy as np


class TraditionalFusion:
    """Class-Adaptive Geometric Late Fusion module for 3D perception."""

    def __init__(
        self,
        iou_thresh_2d: Union[float, Dict[str, float]] = 0.30,
        alpha_boost: Union[float, Dict[str, float]] = 0.25,
        final_conf_thresh: Union[float, Dict[str, float]] = 0.25,
    ):
        """Args:
        iou_thresh_2d: Float or dict mapping class names (e.g. {'Car': 0.50}) to
        minimum 2D IoU. alpha_boost: Float or dict mapping class names to
        rescoring factor alpha. final_conf_thresh: Float or dict mapping class
        names to final score cutoff.
        """
        self.iou_thresh_2d = iou_thresh_2d
        self.alpha_boost = alpha_boost
        self.final_conf_thresh = final_conf_thresh

    def _get_param(
        self,
        param: Union[float, Dict[str, float]],
        cls_type: str,
        default_val: float,
    ) -> float:
        """Helper to resolve class-specific parameter values."""
        if isinstance(param, dict):
            return float(param.get(cls_type, param.get("default", default_val)))
        return float(param)

    @staticmethod
    def project_3d_to_2d_bbox(
        location_3d: Union[List[float], np.ndarray],
        dimensions_3d: Union[List[float], np.ndarray],
        rotation_y: float,
        calib: Any,
        img_shape: tuple = (375, 1242),
    ) -> List[float]:
        """Projects a 3D bounding box onto the 2D image plane using P2 matrix."""
        h, w, l = dimensions_3d
        x, y, z = location_3d
        ry = rotation_y

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

        if np.any(corners_3d[2, :] <= 0.1):
            return [0.0, 0.0, 0.0, 0.0]

        corners_3d_hom = np.vstack(
            [corners_3d, np.ones((1, 8), dtype=np.float32)]
        )
        pts_2d_hom = np.dot(calib.P2, corners_3d_hom)
        pts_2d = (pts_2d_hom[:2, :] / pts_2d_hom[2, :]).T

        xmin = max(0.0, float(np.min(pts_2d[:, 0])))
        ymin = max(0.0, float(np.min(pts_2d[:, 1])))
        xmax = min(float(img_shape[1]), float(np.max(pts_2d[:, 0])))
        ymax = min(float(img_shape[0]), float(np.max(pts_2d[:, 1])))

        if (xmax - xmin) < 3 or (ymax - ymin) < 3:
            return [0.0, 0.0, 0.0, 0.0]

        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def compute_iou_2d(boxA: List[float], boxB: List[float]) -> float:
        """Computes 2D Intersection over Union (IoU) between two boxes."""
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
        """Executes Late Fusion using class-specific matching and rescoring parameters."""
        fused_predictions = []

        for pred in predictions_3d:
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

            # Class specific parameters extraction
            iou_thresh = self._get_param(self.iou_thresh_2d, cls_type, 0.30)
            alpha = self._get_param(self.alpha_boost, cls_type, 0.25)
            conf_thresh = self._get_param(self.final_conf_thresh, cls_type, 0.25)

            # Step 1: 3D->2D projection
            bbox_2d_proj = self.project_3d_to_2d_bbox(loc, dim, ry, calib)

            best_iou = 0.0
            best_det_2d = None

            # Step 2: Matching 2D
            if bbox_2d_proj != [0.0, 0.0, 0.0, 0.0] and detections_2d:
                for det_2d in detections_2d:
                    if det_2d["type"] == cls_type:
                        iou = self.compute_iou_2d(bbox_2d_proj, det_2d["bbox_2d"])
                        if iou > best_iou:
                            best_iou = iou
                            best_det_2d = det_2d

            # Step 3: Rescoring
            if best_iou >= iou_thresh and best_det_2d is not None:
                score_2d = float(best_det_2d["score"])
                score_boost = alpha * score_2d * best_iou
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

            if not filter_final or fused_score >= conf_thresh:
                fused_predictions.append(fused_pred)

        fused_predictions.sort(key=lambda x: x["score"], reverse=True)
        return fused_predictions