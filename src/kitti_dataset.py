import os
import cv2
import numpy as np
from src.calibration import Calibration

class KittiDataset():
    """
    DataLoader modulare per KITTI 3D Object Detection.
    Eredita da torch.utils.data.Dataset per essere direttamente compatibile
    con PyTorch DataLoader e training loop nei progetti futuri.
    """
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform

        self.image_dir = os.path.join(data_root, 'image_2')
        self.velo_dir = os.path.join(data_root, 'velodyne_reduced')
        self.calib_dir = os.path.join(data_root, 'calib')
        self.label_dir = os.path.join(data_root, 'label_2')

        # Carica la lista ordinata di tutti gli ID dei frame
        self.sample_ids = sorted([
            os.path.splitext(f)[0] 
            for f in os.listdir(self.image_dir) 
            if f.endswith('.png')
        ])
        print(f"📦 KittiDataset caricato da '{data_root}' ({self.velo_dir}): {len(self.sample_ids)} campioni trovati.")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        # 1. Carica Immagine RGB
        img_path = os.path.join(self.image_dir, f"{sample_id}.png")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Carica Point Cloud LiDAR (x, y, z, intensity)
        velo_path = os.path.join(self.velo_dir, f"{sample_id}.bin")
        points = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)

        # 3. Carica Matrici di Calibrazione
        calib_path = os.path.join(self.calib_dir, f"{sample_id}.txt")
        calib = Calibration(calib_path)

        # 4. Carica Ground Truth Labels (2D e 3D)
        label_path = os.path.join(self.label_dir, f"{sample_id}.txt")
        objects = self._parse_label(label_path)

        sample = {
            'id': sample_id,
            'image': image,           # np.ndarray (H, W, 3) uint8
            'points': points,         # np.ndarray (N, 4) float32 [x, y, z, intensity]
            'calib': calib,           # Istanza di Calibration
            'objects': objects        # Lista di dizionari Ground Truth
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

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