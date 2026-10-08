import os
import numpy as np
from torch.utils.data import Dataset
from src.calibration import Calibration
from scripts.sparsifier import LidarSparsifier


class KittiDataset(Dataset):
    """KITTI Dataset loader with built-in LiDAR sparsification support."""

    def __init__(
        self,
        data_root: str,
        subsample_mode: str = "none",
        subsample_ratio: float = 1.0,
        num_beams: int = 64,
    ):
        self.data_root = data_root
        self.subsample_mode = subsample_mode
        self.subsample_ratio = subsample_ratio
        self.num_beams = num_beams

        self.velo_dir = os.path.join(data_root, "velodyne_reduced")
        self.calib_dir = os.path.join(data_root, "calib")
        self.label_dir = os.path.join(data_root, "label_2")

        if os.path.exists(self.velo_dir):
            self.sample_ids = sorted([
                os.path.splitext(f)[0]
                for f in os.listdir(self.velo_dir)
                if f.endswith(".bin")
            ])
        else:
            self.sample_ids = []

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.sample_ids[idx]

        # Load LiDAR point cloud (.bin)
        velo_path = os.path.join(self.velo_dir, f"{sample_id}.bin")
        points = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)

        # Apply LiDAR subsampling (defaults to returning original cloud)
        points = LidarSparsifier.sparsify(
            points,
            mode=self.subsample_mode,
            target_beams=self.num_beams,
            keep_ratio=self.subsample_ratio,
        )

        # Load Calibration
        calib_path = os.path.join(self.calib_dir, f"{sample_id}.txt")
        calib = Calibration(calib_path)

        # Load Ground Truth boxes if present
        gt_boxes = []
        label_path = os.path.join(self.label_dir, f"{sample_id}.txt")
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts or parts[0] == "DontCare":
                        continue
                    obj_type = parts[0]
                    trunc = float(parts[1])
                    occ = int(parts[2])
                    alpha = float(parts[3])
                    bbox_2d = [float(x) for x in parts[4:8]]
                    dims = [float(x) for x in parts[8:11]]  # h, w, l
                    loc = [float(x) for x in parts[11:14]]  # x, y, z
                    ry = float(parts[14])

                    gt_boxes.append({
                        "type": obj_type,
                        "truncation": trunc,
                        "occlusion": occ,
                        "alpha": alpha,
                        "bbox_2d": bbox_2d,
                        "dimensions_3d": dims,
                        "location_3d": loc,
                        "rotation_y": ry,
                    })

        return {
            "sample_id": sample_id,
            "points": points,
            "calib": calib,
            "gt_boxes": gt_boxes,
        }