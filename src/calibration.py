import numpy as np

class Calibration:
    """
    Gestisce le matrici di calibrazione KITTI per trasformare punti
    tra i sistemi di riferimento LiDAR (Velodyne), Telecamera (Rectified) e Immagine 2D.
    """
    def __init__(self, calib_file):
        calibs = self._parse_calib_file(calib_file)
        
        # Matrice di proiezione 3x4 della fotocamera RGB principale (Camera 2)
        self.P2 = calibs['P2'].reshape(3, 4)
        
        # Matrice di rettificazione 3x3 per allineare le immagini
        self.R0_rect = calibs['R0_rect'].reshape(3, 3)
        
        # Matrice estrinseca 3x4: Trasformazione rigida da LiDAR a Camera
        self.Tr_velo_to_cam = calibs['Tr_velo_to_cam'].reshape(3, 4)

    def _parse_calib_file(self, filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                if ':' not in line:
                    continue
                key, val = line.split(':', 1)
                data[key.strip()] = np.array([float(x) for x in val.strip().split()])
        return data

    def velo2cam(self, pts_velo):
        """
        Trasforma punti 3D da coordinate LiDAR (x_front, y_left, z_up)
        a coordinate Camera Rectified (x_right, y_down, z_front).
        """
        pts_3d = pts_velo[:, :3]
        pts_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
        
        # 1. LiDAR -> Camera Frame
        pts_cam = np.dot(pts_hom, self.Tr_velo_to_cam.T)
        
        # 2. Camera Frame -> Camera Rectified Frame
        pts_rect = np.dot(pts_cam, self.R0_rect.T)
        return pts_rect

    def velo2img(self, pts_velo):
        """
        Proietta punti 3D LiDAR sui pixel 2D (u, v) dell'immagine RGB.
        Restituisce: (pts_2d, depth, valid_mask)
        """
        pts_rect = self.velo2cam(pts_velo)
        
        # Filtra i punti dietro la fotocamera (z <= 0)
        valid_mask = pts_rect[:, 2] > 0
        pts_rect_valid = pts_rect[valid_mask]
        
        # Proiezione prospettica sui pixel
        pts_hom_2d = np.dot(
            np.hstack((pts_rect_valid, np.ones((pts_rect_valid.shape[0], 1)))), 
            self.P2.T
        )
        
        # Normalizzazione in coordinate u, v
        pts_2d = pts_hom_2d[:, :2] / pts_hom_2d[:, 2:3]
        depth = pts_rect_valid[:, 2]
        
        return pts_2d, depth, valid_mask

    def get_3d_box_corners_cam(self, location, dimensions, rotation_y):
        """
        Calcola gli 8 vertici 3D di una Bounding Box nel sistema di riferimento della telecamera.
        - location: [x, y, z] (centro della base inferiore della box)
        - dimensions: [h, w, l] (altezza, larghezza, lunghezza)
        - rotation_y: angolo di rotazione attorno all'asse Y (radianti)
        """
        h, w, l = dimensions
        x, y, z = location

        # Vertici relativi al centro della box
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]  # Y punta verso il basso in camera frame
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

        # Matrice di rotazione Yaw
        R = np.array([
            [np.cos(rotation_y), 0, np.sin(rotation_y)],
            [0, 1, 0],
            [-np.sin(rotation_y), 0, np.cos(rotation_y)]
        ])

        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += x
        corners_3d[1, :] += y
        corners_3d[2, :] += z

        return corners_3d.T # Shape (8, 3)