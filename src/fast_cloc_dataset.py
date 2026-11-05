import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class FastCLOCsDataset(Dataset):
    """PyTorch Dataset for extracting 2D-3D feature vectors using the P2 projection for correct 2D-3D geometric association."""
    def __init__(self, pred_3d_dir, pred_2d_dir, gt_3d_dir, frame_ids=None, calib_dir=None, iou_match_thresh=0.5):
        self.pred_3d_dir = pred_3d_dir
        self.pred_2d_dir = pred_2d_dir
        self.gt_3d_dir = gt_3d_dir
        self.calib_dir = calib_dir
        self.iou_match_thresh = iou_match_thresh
        self.class_map = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
        
        # Standard KITTI P2 matrix (fallback if calib_dir is not provided)
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
        """Load the P2 matrix from the calibration file for the given frame."""
        if self.calib_dir:
            calib_filepath = os.path.join(self.calib_dir, f"{frame_id}.txt")
            if os.path.exists(calib_filepath):
                with open(calib_filepath, 'r') as f:
                    for line in f:
                        if line.startswith("P2:"):
                            return np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
        return self.default_P2

    def _project_3d_to_2d(self, loc, dim, ry, P2):
        """Project a 3D box into the 2D image plane."""
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
                # 1. Compute GT label (1.0 if the 3D box is within 1.5m of a GT of the same class)
                y_label = 0.0
                for gt in gts_3d:
                    if gt['class'] == b3d['class']:
                        dist = np.linalg.norm(np.array(b3d['loc']) - np.array(gt['loc']))
                        if dist < 1.5:
                            y_label = 1.0
                            break

                # 2. Project 3D -> 2D in the camera plane
                proj_2d_bbox = self._project_3d_to_2d(b3d['loc'], b3d['dim'], b3d['ry'], P2)

                # 3. Match with YOLO 2D predictions using proj_2d_bbox
                matched_2d = False
                for b2d in preds_2d:
                    if b2d['class'] == b3d['class']:
                        iou_2d = self._compute_2d_iou(proj_2d_bbox, b2d['bbox_2d'])
                        
                        if iou_2d > 0.01:
                            vol = b3d['dim'][0] * b3d['dim'][1] * b3d['dim'][2]
                            one_hot = [0.0, 0.0, 0.0]
                            one_hot[b3d['class_idx']] = 1.0

                            feat = [
                                b3d['score'],
                                b2d['score'],
                                iou_2d,
                                b3d['depth'],
                                vol
                            ] + one_hot

                            all_features.append(feat)
                            all_labels.append([y_label])
                            matched_2d = True

                if not matched_2d:
                    vol = b3d['dim'][0] * b3d['dim'][1] * b3d['dim'][2]
                    one_hot = [0.0, 0.0, 0.0]
                    one_hot[b3d['class_idx']] = 1.0
                    feat = [b3d['score'], 0.0, 0.0, b3d['depth'], vol] + one_hot
                    all_features.append(feat)
                    all_labels.append([y_label])

        if not all_features:
            return torch.empty((0, 8), dtype=torch.float32), torch.empty((0, 1), dtype=torch.float32)

        return torch.tensor(all_features, dtype=torch.float32), torch.tensor(all_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]