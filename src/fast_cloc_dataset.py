import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class FastCLOCsDataset(Dataset):
    """
    PyTorch Dataset for extracting 2D-3D candidate feature vectors 
    and Ground Truth labels to train the Fast-CLOCs score aligner network.
    """
    def __init__(self, pred_3d_dir, pred_2d_dir, gt_3d_dir, frame_ids=None, calib_dir=None, iou_match_thresh=0.5):
        """
        Args:
            pred_3d_dir (str): Directory containing 3D candidate predictions (.txt).
            pred_2d_dir (str): Directory containing 2D candidate predictions (.txt).
            gt_3d_dir (str): Directory containing 3D Ground Truth annotations (.txt).
            frame_ids (list, optional): List of frame IDs to include in the dataset.
            calib_dir (str, optional): Directory containing KITTI calibration matrices.
            iou_match_thresh (float): 3D IoU threshold to consider a candidate as Positive (y=1).
        """
        self.pred_3d_dir = pred_3d_dir
        self.pred_2d_dir = pred_2d_dir
        self.gt_3d_dir = gt_3d_dir
        self.calib_dir = calib_dir
        self.iou_match_thresh = iou_match_thresh
        self.class_map = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
        
        # If frame_ids is provided, use only that subset; otherwise load all frames
        if frame_ids is not None:
            self.frame_ids = frame_ids
        else:
            self.frame_ids = [
                os.path.splitext(os.path.basename(f))[0]
                for f in sorted(glob.glob(os.path.join(pred_3d_dir, "*.txt")))
            ]
            
        self.features, self.labels = self._build_candidate_pairs()

    def _compute_2d_iou(self, boxA, boxB):
        """Calculates Intersection over Union (IoU) in 2D image plane coordinates."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def _parse_3d_file(self, filepath):
        """Parses KITTI 3D prediction or Ground Truth text files."""
        boxes = []
        if not os.path.exists(filepath):
            return boxes
        
        with open(filepath, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if not parts or parts[0] == 'DontCare':
                    continue
                
                cls_name = parts[0]
                if cls_name not in self.class_map:
                    continue
                
                # KITTI Format: class, truncated, occluded, alpha, 2d_bbox (4), 3d_dim (3), 3d_loc (3), rotation_y, score
                bbox_2d = [float(p) for p in parts[4:8]]
                h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
                x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
                score = float(parts[14]) if len(parts) > 14 else 1.0
                
                boxes.append({
                    'class': cls_name,
                    'class_idx': self.class_map[cls_name],
                    'bbox_2d': bbox_2d,
                    'dim': [h, w, l],
                    'loc': [x, y, z],
                    'depth': z,
                    'score': score
                })
        return boxes

    def _parse_2d_file(self, filepath):
        """Parses YOLOv8m 2D candidate predictions."""
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
        """Generates feature vectors x_i and target labels y_i for all 2D-3D overlaps."""
        all_features = []
        all_labels = []

        for frame_id in self.frame_ids:
            preds_3d = self._parse_3d_file(os.path.join(self.pred_3d_dir, f"{frame_id}.txt"))
            preds_2d = self._parse_2d_file(os.path.join(self.pred_2d_dir, f"{frame_id}.txt"))
            gts_3d = self._parse_3d_file(os.path.join(self.gt_3d_dir, f"{frame_id}.txt"))

            if not preds_3d:
                continue

            for b3d in preds_3d:
                # 1. Determine Target Label y (1.0 if 3D box matches GT within spatial distance, 0.0 otherwise)
                y_label = 0.0
                for gt in gts_3d:
                    if gt['class'] == b3d['class']:
                        dist = np.linalg.norm(np.array(b3d['loc']) - np.array(gt['loc']))
                        if dist < 1.5:  # Spatial distance threshold for positive match
                            y_label = 1.0
                            break

                # 2. Pair with 2D detections in the same image frame
                matched_2d = False
                for b2d in preds_2d:
                    if b2d['class'] == b3d['class']:
                        iou_2d = self._compute_2d_iou(b3d['bbox_2d'], b2d['bbox_2d'])
                        
                        if iou_2d > 0.01:  # Consider pairs with non-zero 2D overlap
                            vol = b3d['dim'][0] * b3d['dim'][1] * b3d['dim'][2]
                            one_hot = [0.0, 0.0, 0.0]
                            one_hot[b3d['class_idx']] = 1.0

                            # Feature Vector: [S_3D, S_2D, IoU_2D, Depth, Vol, OneHot_Class]
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

                # If no 2D match exists, add isolated 3D candidate with S_2D=0 and IoU_2D=0
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


if __name__ == "__main__":
    # Sanity check execution
    dataset = FastCLOCsDataset(
        pred_3d_dir="experiments/baseline_64beams",
        pred_2d_dir="experiments_camera/yolov8m_baseline_2d",
        gt_3d_dir="data/kitti_validation/label_2"
    )
    print("FastCLOCsDataset initialized successfully.")
    print(f"Total candidate pairs extracted: {len(dataset)}")
    if len(dataset) > 0:
        x_sample, y_sample = dataset[0]
        print(f"Input feature vector shape:  {x_sample.shape}")
        print(f"Target label tensor shape:   {y_sample.shape}")
        print(f"Sample feature values:       {x_sample}")
        print(f"Sample target label:         {y_sample.item()}")