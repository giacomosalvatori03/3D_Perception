import numpy as np
import torch

class Pillarizer:
    """
    Modulo di Pre-Processing che converte una nuvola di punti LiDAR (N, 4)
    in tensori di Pilastri (Pillar Features e Pillar Coordinates) per PointPillars.
    """
    def __init__(
        self,
        voxel_size=[0.16, 0.16, 4.0],
        point_cloud_range=[0.0, -39.68, -3.0, 69.12, 39.68, 1.0],
        max_num_points=32,
        max_pillars=12000
    ):
        self.vx, self.vy, self.vz = voxel_size
        self.pc_range = np.array(point_cloud_range, dtype=np.float32)
        self.max_points = max_num_points
        self.max_pillars = max_pillars

        # Calcola le dimensioni della griglia BEV
        self.grid_x = int(round((self.pc_range[3] - self.pc_range[0]) / self.vx)) # 432
        self.grid_y = int(round((self.pc_range[4] - self.pc_range[1]) / self.vy)) # 496

    def process(self, points: np.ndarray):
        """
        Input: points (N, 4) -> [x, y, z, intensity]
        Output:
          - pillar_features: Tensor (P, max_points, 9)
          - pillar_coords: Tensor (P, 3) -> [batch_idx, y_idx, x_idx]
        """
        # 1. Filtra i punti esterni al range spaziale definito
        mask = (
            (points[:, 0] >= self.pc_range[0]) & (points[:, 0] < self.pc_range[3]) &
            (points[:, 1] >= self.pc_range[1]) & (points[:, 1] < self.pc_range[4]) &
            (points[:, 2] >= self.pc_range[2]) & (points[:, 2] < self.pc_range[5])
        )
        pts = points[mask]

        if len(pts) == 0:
            return None, None

        # 2. Calcola le coordinate della griglia (x_idx, y_idx) per ciascun punto
        grid_x_coords = np.floor((pts[:, 0] - self.pc_range[0]) / self.vx).astype(np.int32)
        grid_y_coords = np.floor((pts[:, 1] - self.pc_range[1]) / self.vy).astype(np.int32)

        # 3. Raggruppa i punti per Pilastro
        pillar_dict = {}
        for i in range(len(pts)):
            coordKey = (grid_y_coords[i], grid_x_coords[i])
            if coordKey not in pillar_dict:
                if len(pillar_dict) >= self.max_pillars:
                    continue
                pillar_dict[coordKey] = []
            
            if len(pillar_dict[coordKey]) < self.max_points:
                pillar_dict[coordKey].append(pts[i])

        num_pillars = len(pillar_dict)
        if num_pillars == 0:
            return None, None

        # 4. Inizializza i tensor di output
        pillar_features = np.zeros((num_pillars, self.max_points, 9), dtype=np.float32)
        pillar_coords = np.zeros((num_pillars, 3), dtype=np.int32)

        for p_idx, (coordKey, p_points) in enumerate(pillar_dict.items()):
            p_pts = np.array(p_points, dtype=np.float32)
            n_pts = len(p_pts)

            # Coordinate griglia [batch_idx=0, y_idx, x_idx]
            pillar_coords[p_idx] = [0, coordKey[0], coordKey[1]]

            # Calcola centro di gravità aritmetico (xc, yc, zc) del pilastro
            mean_pt = np.mean(p_pts[:, :3], axis=0)

            # Calcola centro geometrico del pilastro sulla griglia (xp, yp)
            x_p = coordKey[1] * self.vx + self.pc_range[0] + self.vx / 2.0
            y_p = coordKey[0] * self.vy + self.pc_range[1] + self.vy / 2.0

            # Costruisci le 9 feature per punto:
            # [x, y, z, r, x - xc, y - yc, z - zc, x - xp, y - yp]
            f_pts = np.zeros((n_pts, 9), dtype=np.float32)
            f_pts[:, :4] = p_pts
            f_pts[:, 4:7] = p_pts[:, :3] - mean_pt
            f_pts[:, 7] = p_pts[:, 0] - x_p
            f_pts[:, 8] = p_pts[:, 1] - y_p

            pillar_features[p_idx, :n_pts, :] = f_pts

        return (
            torch.from_numpy(pillar_features),
            torch.from_numpy(pillar_coords)
        )