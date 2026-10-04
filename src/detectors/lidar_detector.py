import os
import torch
import numpy as np
from typing import List
from .base_detector import BaseDetector
from .pillarization import Pillarizer
from .pointpillars_net import PointPillarsNet
from src.detection import Detection3D
from scripts.pointpillar_weights import download_pointpillars_weights

CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']

# Dimensioni medie degli Anchor 3D per KITTI [h, w, l]
ANCHOR_SIZES = {
    'Car': [1.56, 1.60, 3.90],
    'Pedestrian': [1.73, 0.60, 0.80],
    'Cyclist': [1.73, 0.60, 1.76]
}

class LidarDetector(BaseDetector):
    """
    3D Object Detector basato unicamente su dati LiDAR.
    Integrazione per modello PointPillars con pre-filtraggio ROI Frustum.
    """
    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = 0.3,
        max_distance: float = 70.0,
        nms_iou_threshold: float = 0.2,
        device: str = None
    ):
        super().__init__(conf_threshold=conf_threshold)
        self.max_distance = max_distance
        self.nms_iou_threshold = nms_iou_threshold
        
        # 1. Selezione automatica del device (GPU CUDA / CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"⚙️ LidarDetector inizializzato su device: {self.device}")

        # 2. Inizializzazione della Pillarization (Pre-Processing)
        self.pillarizer = Pillarizer()

        # 3. Inizializzazione della rete PointPillars
        self.net = PointPillarsNet().to(self.device)

        # 4. Caricamento del Checkpoint/Modello
        self.checkpoint = self._load_model(model_path)
        self.model = self.checkpoint  # Assegna il modello caricato a self.model

    def _load_model(self, model_path: str):
        """
        Scarica (se necessario) e carica il checkpoint del modello PointPillars.
        """
        if model_path is None:
            model_path = download_pointpillars_weights()

        if model_path is None or not os.path.exists(model_path):
            print("⚠️ Impossibile reperire i pesi del modello. LidarDetector sara in modalita mock.")
            return None

        print(f"📦 Caricamento checkpoint PyTorch da: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            print("✅ Checkpoint PointPillars caricato con successo in memoria!")
            return checkpoint
        except Exception as e:
            print(f"❌ Errore durante il caricamento del checkpoint: {e}")
            return None

    # def filter_roi_frustum(self, points: np.ndarray, calib, img_shape: tuple) -> np.ndarray:
    #     """
    #     Filtra i punti LiDAR mantenendo unicamente quelli che ricadono nel campo visivo (FOV)
    #     della fotocamera e all'interno del range di profondità desiderato [0, max_distance].
    #     """
    #     if points.shape[0] == 0:
    #         return points

    #     # 1. Proietta i punti nel frame fotocamera e sui pixel 2D
    #     # pts_2d e depth contengono solo i punti con z > 0 (valid_mask == True)
    #     pts_2d, depth, valid_mask = calib.velo2img(points)
        
    #     h, w = img_shape[:2]
        
    #     # 2. Maschera per i punti proiettati che cadono nei limiti dell'immagine 2D e del range
    #     in_img_bounds = (
    #         (depth > 0.1) & (depth < self.max_distance) &
    #         (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) &
    #         (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    #     )
        
    #     # 3. Ricostruisci la maschera booleana della dimensione originale di 'points'
    #     final_mask = np.zeros(len(points), dtype=bool)
    #     final_mask[valid_mask] = in_img_bounds
        
    #     return points[final_mask]

    def _preprocess(self, points: np.ndarray) -> dict:
        """
        Converte la nuvola di punti LiDAR filtrata nei tensori di pilastri (Pillar Features & Coords).
        """
        features, coords = self.pillarizer.process(points)
        
        if features is None:
            return None

        return {
            'pillar_features': features.to(self.device),
            'pillar_coords': coords.to(self.device)
        }

    def _postprocess(self, cls_preds, box_preds, calib) -> List[Detection3D]:
        """
        Decodifica delle uscite della rete, filtraggio per confidenza e 3D NMS.
        """
        # Dimensione output griglia feature map: (1, Channels, H, W)
        B, C_cls, H, W = cls_preds.shape
        
        # Reshape predizioni classificazione e box
        cls_scores = torch.sigmoid(cls_preds.permute(0, 2, 3, 1).reshape(-1, 3)) # Score per le 3 classi
        box_deltas = box_preds.permute(0, 2, 3, 1).reshape(-1, 7)
        
        # Filtra i candidati sopra la soglia di confidenza
        max_scores, class_ids = torch.max(cls_scores, dim=1)
        valid_mask = max_scores > self.conf_threshold
        
        if not valid_mask.any():
            return []

        valid_scores = max_scores[valid_mask]
        valid_class_ids = class_ids[valid_mask]
        valid_deltas = box_deltas[valid_mask]
        
        # Decodifica approssimata delle posizioni 3D LiDAR (X, Y, Z, H, W, L, Yaw)
        # Conversione semplificata per griglia BEV
        boxes_lidar = torch.zeros_like(valid_deltas)
        boxes_lidar[:, 0] = valid_deltas[:, 0] * 0.32 + 34.56  # X
        boxes_lidar[:, 1] = valid_deltas[:, 1] * 0.32          # Y
        boxes_lidar[:, 2] = valid_deltas[:, 2] - 1.0           # Z
        boxes_lidar[:, 3:6] = torch.exp(valid_deltas[:, 3:6]) * 1.6 # Dimensioni h, w, l
        boxes_lidar[:, 6] = valid_deltas[:, 6]                 # Rotation Yaw

        # Applicazione Non-Maximum Suppression (NMS) in vista BEV 2D
        # Crea box 2D [x1, y1, x2, y2] per la funzione NMS di torchvision
        boxes_bev_2d = torch.zeros((len(boxes_lidar), 4), device=self.device)
        boxes_bev_2d[:, 0] = boxes_lidar[:, 0] - boxes_lidar[:, 4] / 2
        boxes_bev_2d[:, 1] = boxes_lidar[:, 1] - boxes_lidar[:, 3] / 2
        boxes_bev_2d[:, 2] = boxes_lidar[:, 0] + boxes_lidar[:, 4] / 2
        boxes_bev_2d[:, 3] = boxes_lidar[:, 1] + boxes_lidar[:, 3] / 2

        keep_indices = torchvision.ops.nms(boxes_bev_2d, valid_scores, self.nms_iou_threshold)
        
        # Costruzione degli oggetti Detection3D finali
        detections = []
        for idx in keep_indices:
            box_lid = boxes_lidar[idx].cpu().numpy()
            score = float(valid_scores[idx].cpu().numpy())
            cls_id = int(valid_class_ids[idx].cpu().numpy())
            cls_name = CLASS_NAMES[cls_id]

            # Trasforma le coordinate del centro 3D dal sistema LiDAR al sistema Fotocamera (Rectified)
            pt_lidar = box_lid[:3].reshape(1, 3)
            pt_cam = calib.velo2cam(pt_lidar)[0]

            # Formato Detection3D: box_3d = [x, y, z, h, w, l, r_y] nel frame fotocamera
            box_3d_cam = [
                pt_cam[0], pt_cam[1], pt_cam[2],
                box_lid[3], box_lid[4], box_lid[5],
                box_lid[6]
            ]

            det = Detection3D(
                label=cls_name,
                box_3d=box_3d_cam,
                score=score
            )
            detections.append(det)

        return detections

    def detect(self, sample: dict) -> List[Detection3D]:
        """
        Esegue la pipeline completa di detection 3D da LiDAR.
        """
        points = sample['points']
        calib = sample['calib']

        # 1. Pre-processing (Pillarization)
        inputs = self._preprocess(points)
        if inputs is None:
            return []

        # 2. Forward Pass
        with torch.no_grad():
            cls_preds, box_preds = self.net(inputs['pillar_features'], inputs['pillar_coords'])

        # 3. Post-processing & Costruzione Detection3D
        detections = self._postprocess(cls_preds, box_preds, calib)

        return detections