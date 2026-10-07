import os
import numpy as np
from torch.utils.data import Dataset
from .calibration import Calibration
from .detection import Detection3D
from scripts.sparsifier import LidarSparsifier

class KittiDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        subsample_mode: str = 'none',      # 'none', 'random', 'beam', 'distance'
        subsample_ratio: float = 1.0,      # Usato per mode='random' (es. 0.5, 0.25, 0.1)
        num_beams: int = 32,               # Usato per mode='beam' (es. 32, 16)
        max_distance: float = 35.0         # Usato per mode='distance' (in metri)
    ):
        self.data_root = data_root
        self.subsample_mode = subsample_mode
        self.subsample_ratio = subsample_ratio
        self.num_beams = num_beams
        self.max_distance = max_distance

        self.pts_path = os.path.join(data_root, 'velodyne_reduced')
        self.calib_path = os.path.join(data_root, 'calib')
        self.label_path = os.path.join(data_root, 'label_2')
        self.image_path = os.path.join(data_root, 'image_2')

        self.sample_ids = sorted([
            os.path.splitext(f)[0] for f in os.listdir(self.pts_path) if f.endswith('.bin')
        ])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.sample_ids[idx]
        
        # 1. Carica punti LiDAR (.bin)
        bin_file = os.path.join(self.pts_path, f"{sample_id}.bin")
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)

        # 2. Applica sparsificazione se abilitata
        if self.subsample_mode != 'none':
            points = LidarSparsifier.sparsify(
                points,
                mode=self.subsample_mode,
                ratio=self.subsample_ratio,
                num_beams=self.num_beams,
                max_distance=self.max_distance
            )

        # 3. Carica Calibration e Ground Truth
        calib = Calibration(os.path.join(self.calib_path, f"{sample_id}.txt"))
        gt_boxes = self._load_labels(os.path.join(self.label_path, f"{sample_id}.txt"))
        
        img_path = os.path.join(self.image_path, f"{sample_id}.png")

        return {
            'sample_id': sample_id,
            'points': points,
            'calib': calib,
            'gt_boxes': gt_boxes,
            'image_path': img_path if os.path.exists(img_path) else None
        }

    def _parse_label(self, label_path):
        """
        Parsa il file txt di annotazione Ground Truth in formato KITTI.
        """
        objects = []
        if not os.path.exists(label_path):
            return objects

        with open(label_path, 'r') as f:
            for line in f.readlines():
                data = line.strip().split()
                if not data or data[0] == 'DontCare':
                    continue

                obj = {
                    'type': data[0],                               # Class (Car, Pedestrian, Cyclist, ecc.)
                    'truncation': float(data[1]),                  # Troncamento [0..1]
                    'occlusion': int(data[2]),                     # Occlusione (0, 1, 2, 3)
                    'alpha': float(data[3]),                       # Angolo osservazione [-pi..pi]
                    'bbox_2d': np.array([float(x) for x in data[4:8]], dtype=np.float32),   # [left, top, right, bottom]
                    'dimensions_3d': np.array([float(x) for x in data[8:11]], dtype=np.float32),  # [h, w, l]
                    'location_3d': np.array([float(x) for x in data[11:14]], dtype=np.float32),  # [x, y, z] camera frame
                    'rotation_y': float(data[14])                  # Rotazione Y [-pi..pi]
                }
                objects.append(obj)
        return objects