import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from shapely.geometry import Polygon


class FastCLOCsDataset(Dataset):
    """
    PyTorch Dataset for extracting 2D-3D feature vectors using P2 projection 
    and BEV IoU target labeling for Fast-CLOCs score alignment.
    """
    def __init__(self, pred_3d_dir, pred_2d_dir, gt_3d_dir, frame_ids=None, calib_dir=None, iou_match_thresh=None):
        self.pred_3d_dir = pred_3d_dir
        self.pred_2d_dir = pred_2d_dir
        self.gt_3d_dir = gt_3d_dir
        self.calib_dir = calib_dir
        
        # IoU thresholds for BEV matching to assign positive labels (y=1)
        self.iou_match_thresh = iou_match_thresh or {
            "Car": 0.50,
            "Pedestrian": 0.25,
            "Cyclist": 0.25,
            "default": 0.25
        }
        
        self.class_map = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
        
        self.default_P2 = np.array([
            [721.5377, 0.0, 609.5593, 44.85728],
            [0.0, 721.5377, 172.8540, 0.2163761],
            [0.0, 0.0, 1.0, 0.002745884]
        ])

        if frame_ids is not None:
            self.frame_ids = frame_ids
        else:
            self.frame_ids = [
                os.path.splitext(os.path.basename(f))[0]
                for f in sorted(glob.glob(os.path.join(pred_3d_dir, "*.txt")))
            ]
            
        self.features, self.labels = self._build_candidate_pairs()

    def _load_p2_matrix(self, frame_id):
        if self.calib_dir:
            calib_filepath = os.path.join(self.calib_dir, f"{frame_id}.txt")
            if os.path.exists(calib_filepath):
                with open(calib_filepath, 'r') as f:
                    for line in f:
                        if line.startswith("P2:"):
                            return np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
        return self.default_P2

    def _project_3d_to_2d(self, loc, dim, ry, P2):
        h, w, l = dim
        x, y, z = loc
        
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        
        R = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        corners_3d = np.vstack([x_corners, y_corners, z_corners])
        corners_3d_cam = np.dot(R, corners_3d) + np.array([[x], [y], [z]])
        
        corners_3d_hom = np.vstack([corners_3d_cam, np.ones((1, 8))])
        pts_2d = np.dot(P2, corners_3d_hom)
        pts_2d[0, :] /= pts_2d[2, :]
        pts_2d[1, :] /= pts_2d[2, :]
        
        return [
            float(np.min(pts_2d[0, :])),
            float(np.min(pts_2d[1, :])),
            float(np.max(pts_2d[0, :])),
            float(np.max(pts_2d[1, :]))
        ]

    def _compute_2d_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def _compute_bev_iou(self, boxA, boxB):
        """Computes the real BEV IoU between two 3D boxes for correct GT label assignment."""
        try:
            x_a, _, z_a = boxA['loc']
            _, w_a, l_a = boxA['dim']
            ry_a = boxA['ry']
            corners_a = np.array([[w_a/2, l_a/2], [w_a/2, -l_a/2], [-w_a/2, -l_a/2], [-w_a/2, l_a/2]])
            R_a = np.array([[np.cos(ry_a), np.sin(ry_a)], [-np.sin(ry_a), np.cos(ry_a)]])
            polyA = Polygon(np.dot(corners_a, R_a) + np.array([x_a, z_a]))

            x_b, _, z_b = boxB['loc']
            _, w_b, l_b = boxB['dim']
            ry_b = boxB['ry']
            corners_b = np.array([[w_b/2, l_b/2], [w_b/2, -l_b/2], [-w_b/2, -l_b/2], [-w_b/2, l_b/2]])
            R_b = np.array([[np.cos(ry_b), np.sin(ry_b)], [-np.sin(ry_b), np.cos(ry_b)]])
            polyB = Polygon(np.dot(corners_b, R_b) + np.array([x_b, z_b]))

            if not polyA.is_valid or not polyB.is_valid:
                return 0.0

            inter = polyA.intersection(polyB).area
            union = polyA.area + polyB.area - inter
            return float(inter / max(union, 1e-6))
        except Exception:
            return 0.0

    def _parse_3d_file(self, filepath):
        boxes = []
        if not os.path.exists(filepath):
            return boxes
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] not in self.class_map:
                    continue
                
                cls_name = parts[0]
                alpha = float(parts[3]) if len(parts) > 3 else 0.0
                bbox_2d = [float(x) for x in parts[4:8]] if len(parts) >= 8 else [0.0, 0.0, 0.0, 0.0]
                dim = [float(x) for x in parts[8:11]]
                loc = [float(x) for x in parts[11:14]]
                ry = float(parts[14]) if len(parts) > 14 else 0.0
                score = float(parts[15]) if len(parts) > 15 else float(parts[-1])

                boxes.append({
                    'class': cls_name,
                    'class_idx': self.class_map[cls_name],
                    'alpha': alpha,
                    'bbox_2d': bbox_2d,
                    'dim': dim,
                    'loc': loc,
                    'ry': ry,
                    'score': score,
                    'depth': loc[2]
                })
        return boxes

    def _parse_2d_file(self, filepath):
        boxes = []
        if not os.path.exists(filepath):
            return boxes
            
        with open(filepath, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cls_name = parts[0]
                if cls_name in self.class_map:
                    bbox_2d = [float(p) for p in parts[4:8]]
                    score = float(parts[14]) if len(parts) > 14 else float(parts[1])
                    boxes.append({
                        'class': cls_name,
                        'bbox_2d': bbox_2d,
                        'score': score
                    })
        return boxes

    def _build_candidate_pairs(self):
        all_features = []
        all_labels = []

        for frame_id in self.frame_ids:
            preds_3d = self._parse_3d_file(os.path.join(self.pred_3d_dir, f"{frame_id}.txt"))
            preds_2d = self._parse_2d_file(os.path.join(self.pred_2d_dir, f"{frame_id}.txt"))
            gts_3d = self._parse_3d_file(os.path.join(self.gt_3d_dir, f"{frame_id}.txt"))

            if not preds_3d:
                continue

            P2 = self._load_p2_matrix(frame_id)

            for b3d in preds_3d:
                # 1. Determining target label y based on real 3D/BEV IoU with Ground Truth
                y_label = 0.0
                req_thresh = self.iou_match_thresh.get(b3d['class'], self.iou_match_thresh["default"])
                for gt in gts_3d:
                    if gt['class'] == b3d['class']:
                        bev_iou = self._compute_bev_iou(b3d, gt)
                        if bev_iou >= req_thresh:
                            y_label = 1.0
                            break

                # 2. Projecting 3D box to 2D camera plane using P2 matrix
                proj_2d_bbox = self._project_3d_to_2d(b3d['loc'], b3d['dim'], b3d['ry'], P2)

                # 3. Find the BEST 2D match (aligned with inference)
                best_2d_score = 0.0
                best_iou_2d = 0.0

                for b2d in preds_2d:
                    if b2d['class'] == b3d['class']:
                        iou_2d = self._compute_2d_iou(proj_2d_bbox, b2d['bbox_2d'])
                        if iou_2d > best_iou_2d:
                            best_iou_2d = iou_2d
                            best_2d_score = b2d['score']

                vol = b3d['dim'][0] * b3d['dim'][1] * b3d['dim'][2]
                one_hot = [0.0, 0.0, 0.0]
                one_hot[b3d['class_idx']] = 1.0

                feat = [
                    b3d['score'],
                    best_2d_score,
                    best_iou_2d,
                    b3d['depth'],
                    vol
                ] + one_hot

                all_features.append(feat)
                all_labels.append([y_label])

        if not all_features:
            return torch.empty((0, 8), dtype=torch.float32), torch.empty((0, 1), dtype=torch.float32)

        return torch.tensor(all_features, dtype=torch.float32), torch.tensor(all_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]