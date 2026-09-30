import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from src.detection import Detection3D

class Visualizer:
    """
    Modulo di visualizzazione per proiettare Bounding Box 2D/3D sulle immagini RGB
    e per generare viste dall'alto (Bird's Eye View - BEV) della point cloud LiDAR.
    """

    # Connessioni degli 8 vertici per formare i 12 spigoli del cuboide 3D
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
        """
        Disegna un cuboide 3D proiettato sull'immagine RGB.
        - color: Tupla RGB/BGR (es. (0, 255, 0) per verde GT, (255, 0, 0) per rosso Pred)
        """
        img_out = image.copy()
        
        # 1. Calcola gli 8 vertici 3D nel frame fotocamera
        corners_3d = calib.get_3d_box_corners_cam(location_3d, dimensions_3d, rotation_y)
        
        # Ignora l'oggetto se è posizionato interamente dietro la fotocamera (z <= 0)
        if np.any(corners_3d[:, 2] <= 0.1):
            return img_out

        # 2. Proietta gli 8 vertici sui pixel 2D
        corners_2d = cls.project_rect_to_image(corners_3d, calib.P2).astype(int)

        # 3. Disegna i 12 spigoli del cuboide
        for edge in cls.BOX_EDGES:
            pt1 = tuple(corners_2d[edge[0]])
            pt2 = tuple(corners_2d[edge[1]])
            cv2.line(img_out, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

        # 4. Disegna una 'X' sulla faccia anteriore per evidenziare la direzione di marcia
        front_face_idx = [0, 1, 5, 4]
        pt_f0 = tuple(corners_2d[front_face_idx[0]])
        pt_f2 = tuple(corners_2d[front_face_idx[2]])
        pt_f1 = tuple(corners_2d[front_face_idx[1]])
        pt_f3 = tuple(corners_2d[front_face_idx[3]])
        cv2.line(img_out, pt_f0, pt_f2, color, max(1, thickness - 1), lineType=cv2.LINE_AA)
        cv2.line(img_out, pt_f1, pt_f3, color, max(1, thickness - 1), lineType=cv2.LINE_AA)

        # 5. Etichetta opzionale (es. Classe + Confidenza)
        if label:
            top_left = corners_2d[:, 1].min(), corners_2d[:, 1].min()
            text_pos = (max(0, corners_2d[4, 0]), max(15, corners_2d[4, 1] - 5))
            cv2.putText(img_out, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return img_out

    @classmethod
    def visualize_scene(
        cls,
        sample: dict,
        predictions: List[Detection3D] = None,
        draw_gt: bool = True,
        figsize: tuple = (14, 7)
    ):
        """
        Visualizza affiancati:
        1. Immagine RGB con Bounding Box 3D Ground Truth (Verde) e Predizioni (Rosso).
        2. Mappa vista dall'alto Bird's Eye View (BEV) dei punti LiDAR con le Bounding Box.
        """
        image = sample['image'].copy()
        calib = sample['calib']
        points = sample['points']
        gt_objects = sample['objects']

        # 1. Disegna le Bounding Box 3D Ground Truth (Verde: (0, 255, 0))
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

        # 2. Disegna le Predizioni 3D del Detector (Rosso: (255, 0, 0))
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

        # 3. Costruzione del Plot BEV (Vista dall'alto sul piano X-Z)
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Immagine RGB proiettata
        axes[0].imshow(image)
        axes[0].set_title(f"3D Bounding Boxes proiettate su Immagine - Frame {sample['id']}")
        axes[0].axis('off')

        # Plot 2: Bird's Eye View (BEV)
        # Converti punti LiDAR in coordinate fotocamera (x_right, y_down, z_forward)
        pts_cam = calib.velo2cam(points)
        
        # Filtra i punti nella regione di interesse BEV (es. X tra -25m e +25m, Z tra 0m e 60m)
        bev_mask = (pts_cam[:, 2] > 0) & (pts_cam[:, 2] < 60) & (np.abs(pts_cam[:, 0]) < 25)
        pts_bev = pts_cam[bev_mask]

        # Scatter plot dei punti LiDAR (vista X-Z dall'alto)
        axes[1].scatter(pts_bev[:, 0], pts_bev[:, 2], c=pts_bev[:, 2], cmap='viridis', s=0.5, alpha=0.6)
        
        # Disegna Ground Truth su BEV
        if draw_gt and gt_objects:
            for obj in gt_objects:
                x, _, z = obj['location_3d']
                axes[1].plot(x, z, 'go', markersize=6, label='GT Center')
                
        # Disegna Predizioni su BEV
        if predictions:
            for pred in predictions:
                x, _, z = pred.location_3d
                axes[1].plot(x, z, 'rx', markersize=8, label='Pred Center')

        axes[1].set_xlim(-25, 25)
        axes[1].set_ylim(0, 60)
        axes[1].set_xlabel("X (Laterale, metri)")
        axes[1].set_ylabel("Z (Profondità, metri)")
        axes[1].set_title("Bird's Eye View (BEV) - Piano X-Z")
        axes[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()