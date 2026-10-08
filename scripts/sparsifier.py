import numpy as np


class LidarSparsifier:
    """Utility class to apply LiDAR point cloud sparsification.

    Supports vertical beam reduction and uniform random point drop.
    """

    def __init__(self):
        pass

    @staticmethod
    def reduce_beams(
        points: np.ndarray,
        original_beams: int = 64,
        target_beams: int = 64,
    ) -> np.ndarray:
        """Simulates vertical beam reduction (e.g., 64-ring to 32 or 16 rings)

        by computing point elevation angles and sampling uniform ring intervals.
        """
        if target_beams >= original_beams or target_beams <= 0:
            return points

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        depth = np.sqrt(x**2 + y**2 + z**2)
        elevation = np.arcsin(np.clip(z / (depth + 1e-6), -1.0, 1.0))

        min_el, max_el = elevation.min(), elevation.max()
        if max_el - min_el < 1e-6:
            return points

        step = original_beams // target_beams
        ring_ids = np.floor(
            (elevation - min_el) / (max_el - min_el + 1e-6) * original_beams
        ).astype(int)

        mask = (ring_ids % step) == 0
        return points[mask]

    @staticmethod
    def random_drop(
        points: np.ndarray, keep_ratio: float = 1.0
    ) -> np.ndarray:
        """Uniformly drops points at random to simulate lower point cloud density."""
        if keep_ratio >= 1.0 or keep_ratio <= 0.0:
            return points

        num_points = points.shape[0]
        num_keep = int(num_points * keep_ratio)
        indices = np.random.choice(num_points, size=num_keep, replace=False)
        return points[indices]

    @staticmethod
    def sparsify(
        points: np.ndarray,
        mode: str = "none",
        target_beams: int = 64,
        keep_ratio: float = 1.0,
    ) -> np.ndarray:
        """Applies the selected subsampling strategy to the input point cloud.

        Defaults to mode='none', target_beams=64, keep_ratio=1.0 (no subsampling).
        """
        if mode in ["beam", "beam_reduction"]:
            return LidarSparsifier.reduce_beams(
                points, original_beams=64, target_beams=target_beams
            )
        elif mode in ["random", "random_drop"]:
            return LidarSparsifier.random_drop(points, keep_ratio=keep_ratio)
        elif mode in ["none", None]:
            return points
        else:
            raise ValueError(f"Unknown subsampling mode: {mode}")