import numpy as np

class Calibration:
    """Handle KITTI calibration matrices for transforming points between LiDAR (Velodyne),
      Camera (Rectified), and 2D Image reference frames.
      """
    def __init__(self, calib_file):
        calibs = self._parse_calib_file(calib_file)
        
        # Projection matrix for the main RGB camera (Camera 2)
        self.P2 = calibs['P2'].reshape(3, 4)
        
        # Rectification matrix for the camera (R0_rect) to align images
        self.R0_rect = calibs['R0_rect'].reshape(3, 3)
        
        # Extrinsic matrix 3x4: Rigid transformation from LiDAR to Camera
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
        Transform 3D points from LiDAR coordinates (x_front, y_left, z_up)
        to Rectified Camera coordinates (x_right, y_down, z_front).
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
        Project 3D LiDAR points onto 2D image pixels (u, v).
        Returns: (pts_2d, depth, valid_mask)
        """
        pts_rect = self.velo2cam(pts_velo)
        
        # Filter points behind the camera (z <= 0)
        valid_mask = pts_rect[:, 2] > 0
        pts_rect_valid = pts_rect[valid_mask]
        
        # Perspective projection onto pixels
        pts_hom_2d = np.dot(
            np.hstack((pts_rect_valid, np.ones((pts_rect_valid.shape[0], 1)))), 
            self.P2.T
        )
        
        # Normalization in u, v coordinates
        pts_2d = pts_hom_2d[:, :2] / pts_hom_2d[:, 2:3]
        depth = pts_rect_valid[:, 2]
        
        return pts_2d, depth, valid_mask

    def get_3d_box_corners_cam(self, location, dimensions, rotation_y):
        """Calculate the 8 corners of a 3D bounding box in camera coordinates.
        - location: [x, y, z] (center of the bottom face of the box)
        - dimensions: [h, w, l] (height, width, length)
        - rotation_y: rotation angle around the Y-axis (in radians)
        """
        h, w, l = dimensions
        x, y, z = location

        # Vertices relative to the box center
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]  # Y points down in camera frame
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

        # Rotation matrix for Yaw
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