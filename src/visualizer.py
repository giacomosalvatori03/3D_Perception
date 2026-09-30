import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from src.detection import Detection3D

class Visualizer:
    """
    Modulo di visualizzazione per proiettare Bounding Box 2D/3D sulle immagini RGB
    e per generare viste dall'alto (Bird's Eye View - BEV) orientate sul piano X-Z.
    """

    BOX_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Base inferiore
        (4, 5), (5, 6), (6, 7), (7, 4),  # Base superiore
        (0, 4), (1, 5), (2, 6), (3, 7)   # Inserzioni verticali
    ]

    @staticmethod
    def project_rect_to_image(pts_rect: np.ndarray, P2: np.ndarray) -> np.ndarray:
        """Proietta punti 3D nel sistema di riferimento fotocamera rettificato sui pixel 2D dell'immagine."""
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
        """Disegna un cuboide 3D proiettato sull'immagine RGB."""
        img_out = image.copy()
        corners_3d = calib.get_3d_box_corners_cam(location_3d, dimensions_3d, rotation_y)
        
        if np.any(corners_3d[:, 2] <= 0.1):
            return img_out

        corners_2d = cls.project_rect_to_image(corners_3d, calib.P2).astype(int)

        for edge in cls.BOX_EDGES:
            pt1 = tuple(corners_2d[edge[0]])
            pt2 = tuple(corners_2d[edge[1]])
            cv2.line(img_out, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

        # Disegna 'X' sulla faccia anteriore
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
        location_3d: np.ndarray,
        dimensions_3d: np.ndarray,
        rotation_y: float,
        calib,
        color: str = 'g',
        linewidth: float = 1.5,
        label: str = None
    ):
        """
        Disegna un rettangolo orientato (Oriented 2D Box) sul piano BEV (X-Z).
        Aggiunge una linea vettoriale dal centro alla parte anteriore per mostrare l'orientamento.
        """
        # Calcola gli 8 vertici 3D
        corners_3d = calib.get_3d_box_corners_cam(location_3d, dimensions_3d, rotation_y)
        
        # Prendi i 4 vertici della base inferiore proiettati su (X, Z)
        bev_corners = corners_3d[:4, [0, 2]] # Shape (4, 2) -> (x, z)
        
        # Chiudi il poligono collegando l'ultimo punto al primo
        bev_corners_closed = np.vstack([bev_corners, bev_corners[0]])
        
        # Disegna il perimetro del rettangolo orientato
        ax.plot(bev_corners_closed[:, 0], bev_corners_closed[:, 1], color=color, linewidth=linewidth, label=label)
        
        # Calcola e disegna la linea direzionale (dal centro della box verso il fronte del veicolo)
        front_center = (bev_corners[0] + bev_corners[1]) / 2.0
        box_center = location_3d[[0, 2]]
        ax.plot([box_center[0], front_center[0]], [box_center[1], front_center[1]], color=color, linewidth=linewidth + 0.5)

    @classmethod
    def visualize_scene(
        cls,
        sample: dict,
        predictions: List[Detection3D] = None,
        draw_gt: bool = True,
        figsize: tuple = (16, 7)
    ):
        """
        Visualizza affiancati:
        1. Immagine RGB con Cuboidi 3D proiettati.
        2. Vista BEV (Bird's Eye View) con Rettangoli 2D Orientati per GT e Predizioni.
        """
        image = sample['image'].copy()
        calib = sample['calib']
        points = sample['points']
        gt_objects = sample['objects']

        # 1. Proiezione su Immagine RGB (GT = Verde, Pred = Rosso)
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

        # Subplot 1: Immagine RGB
        axes[0].imshow(image)
        axes[0].set_title(f"3D Bounding Boxes proiettate su Immagine - Frame {sample['id']}")
        axes[0].axis('off')

        # Subplot 2: Bird's Eye View (BEV)
        pts_cam = calib.velo2cam(points)
        bev_mask = (pts_cam[:, 2] > 0) & (pts_cam[:, 2] < 60) & (np.abs(pts_cam[:, 0]) < 25)
        pts_bev = pts_cam[bev_mask]

        # Punti LiDAR
        axes[1].scatter(pts_bev[:, 0], pts_bev[:, 2], c=pts_bev[:, 2], cmap='viridis', s=0.5, alpha=0.5)

        # Disegna Bounding Box orientate GT in BEV (Verde)
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

        # Disegna Bounding Box orientate Predette in BEV (Rosso)
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
        axes[1].set_xlabel("X (Laterale, metri)")
        axes[1].set_ylabel("Z (Profondità, metri)")
        axes[1].set_title("Bird's Eye View (BEV) - Rettangoli Orientati 2D")
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend(loc='upper right')

        plt.tight_layout()
        plt.show()