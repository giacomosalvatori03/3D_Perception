import numpy as np

class LidarSparsifier:
    """
    Modulo per la sparsificazione sintetica delle nuvole di punti LiDAR.
    Supporta:
      - 'random': sottocampionamento casuale uniforme.
      - 'beam': riduzione del numero di fasci/ring verticali.
      - 'distance': filtraggio basato sul raggio di distanza massimo.
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
            # Calcolo angolo di elevazione (pitch) per ogni punto
            r = np.linalg.norm(points[:, :3], axis=1)
            pitch = np.arcsin(np.clip(points[:, 2] / (r + 1e-6), -1.0, 1.0))
            
            # KITTI Velodyne HDL-64E ha 64 ring verticali
            total_rings = 64
            bins = np.linspace(pitch.min(), pitch.max(), total_rings)
            ring_ids = np.digitize(pitch, bins)
            
            # Calcola lo stride per mantenere solo 'num_beams' fasci
            stride = max(1, total_rings // num_beams)
            allowed_rings = set(range(0, total_rings, stride))
            
            mask = np.isin(ring_ids, list(allowed_rings))
            return points[mask]

        elif mode == 'distance':
            # Mantiene solo i punti entro il raggio max_distance
            distances = np.linalg.norm(points[:, :3], axis=1)
            return points[distances <= max_distance]

        return points