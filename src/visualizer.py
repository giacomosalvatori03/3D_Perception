import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from src.detection import Detection3D

class Visualizer:
    """
    Visualization module to project 2D/3D Bounding Boxes onto RGB images
    and to generate oriented Bird's Eye View (BEV) on the X-Z plane.
    """
    BOX_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Inferior base
        (4, 5), (5, 6), (6, 7), (7, 4),  # Superior base
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical connections
    ]

    @staticmethod
    def project_rect_to_image(pts_rect: np.ndarray, P2: np.ndarray) -> np.ndarray:
        """Project 3D points in the rectified camera reference frame to 2D image pixels."""
        pts_hom = np.hstack((pts_rect, np.ones((pts_rect.shape[0], 1))))
        pts_2d_hom = np.dot(pts_hom, P2.T)
        pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]
        return pts_2d

    @classmethod
    def draw_box3d_on_image(
        cls,
        image: np.ndarray,
        location_3d: np.ndarray,
        dimensions_3d: np.ndarray,
        rotation_y: float,
        calib,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
        label: str = None
    ) -> np.ndarray:
        """Draw a 3D cuboid projected onto the RGB image."""
        img_out = image.copy()
        corners_3d = calib.get_3d_box_corners_cam(location_3d, dimensions_3d, rotation_y)
        
        if np.any(corners_3d[:, 2] <= 0.1):
            return img_out

        corners_2d = cls.project_rect_to_image(corners_3d, calib.P2).astype(int)

        for edge in cls.BOX_EDGES:
            pt1 = tuple(corners_2d[edge[0]])
            pt2 = tuple(corners_2d[edge[1]])
            cv2.line(img_out, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

        # Draw 'X' on the front face
        front_face_idx = [0, 1, 5, 4]
        pt_f0 = tuple(corners_2d[front_face_idx[0]])
        pt_f2 = tuple(corners_2d[front_face_idx[2]])
        pt_f1 = tuple(corners_2d[front_face_idx[1]])
        pt_f3 = tuple(corners_2d[front_face_idx[3]])
        cv2.line(img_out, pt_f0, pt_f2, color, max(1, thickness - 1), lineType=cv2.LINE_AA)
        cv2.line(img_out, pt_f1, pt_f3, color, max(1, thickness - 1), lineType=cv2.LINE_AA)

        if label:
            text_pos = (max(0, corners_2d[4, 0]), max(15, corners_2d[4, 1] - 5))
            cv2.putText(img_out, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return img_out

    @classmethod
    def draw_bev_box(
        cls,
        ax,
        location_3d,
        dimensions_3d,
        rotation_y,
        calib,
        color="g",
        linewidth=1.5,
        label=None,
    ):
        """Draws a 2D oriented bounding box in Bird's Eye View (BEV) with a directional line."""
        # Assicura che gli input siano array NumPy
        location_3d = np.asarray(location_3d)
        dimensions_3d = np.asarray(dimensions_3d)

        h, w, l = dimensions_3d
        x, y, z = location_3d

        # Calcolo dei 4 angoli nel piano BEV (XZ)
        x_corners = [w / 2, w / 2, -w / 2, -w / 2]
        z_corners = [l / 2, -l / 2, -l / 2, l / 2]

        R = np.array(
            [
                [np.cos(rotation_y), np.sin(rotation_y)],
                [-np.sin(rotation_y), np.cos(rotation_y)],
            ]
        )

        corners = np.vstack([x_corners, z_corners])
        corners = np.dot(R, corners)
        corners[0, :] += x
        corners[1, :] += z

        # Chiude il poligono collegando l'ultimo punto al primo
        corners_x = np.append(corners[0, :], corners[0, 0])
        corners_z = np.append(corners[1, :], corners[1, 0])

        ax.plot(corners_x, corners_z, color=color, linewidth=linewidth, label=label)

        # Direzione del fronte del box
        front_center = np.array(
            [
                x + (l / 2) * np.sin(rotation_y),
                z + (l / 2) * np.cos(rotation_y),
            ]
        )
        box_center = location_3d[[0, 2]]

        ax.plot(
            [box_center[0], front_center[0]],
            [box_center[1], front_center[1]],
            color=color,
            linewidth=linewidth + 0.5,
        )

    @classmethod
    def visualize_scene(
        cls,
        sample: dict,
        predictions: List[Detection3D] = None,
        draw_gt: bool = True,
        figsize: tuple = (16, 7)
    ):
        """Visualize side-by-side:
        1. RGB Image with projected 3D cuboids.
        2. BEV (Bird's Eye View) with Oriented 2D Rectangles for GT and Predictions.
        """
        image = sample['image'].copy()
        calib = sample['calib']
        points = sample['points']
        gt_objects = sample['labels']

        # 1. Projection onto RGB Image (GT = Green, Pred = Red)
        if draw_gt and gt_objects:
            for obj in gt_objects:
                image = cls.draw_box3d_on_image(
                    image=image,
                    location_3d=obj['location_3d'],
                    dimensions_3d=obj['dimensions_3d'],
                    rotation_y=obj['rotation_y'],
                    calib=calib,
                    color=(0, 255, 0),
                    thickness=2,
                    label=f"GT: {obj['type']}"
                )

        if predictions:
            for pred in predictions:
                image = cls.draw_box3d_on_image(
                    image=image,
                    location_3d=pred.location_3d,
                    dimensions_3d=pred.dimensions_3d,
                    rotation_y=pred.rotation_y,
                    calib=calib,
                    color=(255, 0, 0),
                    thickness=2,
                    label=f"Pred: {pred.type} {pred.score:.2f}"
                )

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Subplot 1: RGB Image with 3D Bounding Boxes
        axes[0].imshow(image)
        axes[0].set_title(f"3D Bounding Boxes projected on RGB Image - Frame {sample['sample_id']}")
        axes[0].axis('off')

        # Subplot 2: Bird's Eye View (BEV)
        pts_cam = calib.velo2cam(points)
        bev_mask = (pts_cam[:, 2] > 0) & (pts_cam[:, 2] < 60) & (np.abs(pts_cam[:, 0]) < 25)
        pts_bev = pts_cam[bev_mask]

        # LiDAR points in BEV (X-Z plane)
        axes[1].scatter(pts_bev[:, 0], pts_bev[:, 2], c=pts_bev[:, 2], cmap='viridis', s=0.5, alpha=0.5)

        # Draw Oriented GT Boxes in BEV (Green)
        if draw_gt and gt_objects:
            for i, obj in enumerate(gt_objects):
                label = 'Ground Truth' if i == 0 else None
                cls.draw_bev_box(
                    ax=axes[1],
                    location_3d=obj['location_3d'],
                    dimensions_3d=obj['dimensions_3d'],
                    rotation_y=obj['rotation_y'],
                    calib=calib,
                    color='g',
                    linewidth=1.8,
                    label=label
                )

        # Draw Oriented Prediction Boxes in BEV (Red)
        if predictions:
            for i, pred in enumerate(predictions):
                label = 'Prediction' if i == 0 else None
                cls.draw_bev_box(
                    ax=axes[1],
                    location_3d=pred.location_3d,
                    dimensions_3d=pred.dimensions_3d,
                    rotation_y=pred.rotation_y,
                    calib=calib,
                    color='r',
                    linewidth=2.0,
                    label=label
                )

        axes[1].set_xlim(-25, 25)
        axes[1].set_ylim(0, 60)
        axes[1].set_xlabel("X (Lateral, meters)")
        axes[1].set_ylabel("Z (Depth, meters)")
        axes[1].set_title("Bird's Eye View (BEV) - Oriented 2D Rectangles")
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend(loc='upper right')

        plt.tight_layout()
        plt.show()