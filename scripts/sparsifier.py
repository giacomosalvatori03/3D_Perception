import numpy as np

class LidarSparsifier:
    """
    Module for synthetic sparsification of LiDAR point clouds.
    Supports:
      - 'random': uniform random subsampling.
      - 'beam': reduction of the number of beams/rings.
      - 'distance': filtering based on maximum distance radius.
    """
    @staticmethod
    def sparsify(
        points: np.ndarray,
        mode: str = 'none',
        ratio: float = 1.0,
        num_beams: int = 32,
        max_distance: float = 35.0
    ) -> np.ndarray:
        if mode == 'none' or ratio >= 1.0:
            return points

        if mode == 'random':
            num_pts = int(len(points) * ratio)
            indices = np.random.choice(len(points), size=num_pts, replace=False)
            return points[indices]

        elif mode == 'beam':
            # Calculate the elevation angle (pitch) for each point
            r = np.linalg.norm(points[:, :3], axis=1)
            pitch = np.arcsin(np.clip(points[:, 2] / (r + 1e-6), -1.0, 1.0))
            
            # KITTI Velodyne HDL-32E has 64 vertical rings
            total_rings = 64
            bins = np.linspace(pitch.min(), pitch.max(), total_rings)
            ring_ids = np.digitize(pitch, bins)
            
            # Calculate the stride to keep only 'num_beams' beams
            stride = max(1, total_rings // num_beams)
            allowed_rings = set(range(0, total_rings, stride))
            
            mask = np.isin(ring_ids, list(allowed_rings))
            return points[mask]

        elif mode == 'distance':
            # Keep points within the specified maximum distance from the origin
            distances = np.linalg.norm(points[:, :3], axis=1)
            return points[distances <= max_distance]

        return points