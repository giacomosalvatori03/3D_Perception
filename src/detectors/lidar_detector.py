import os
import torch
import numpy as np
from typing import List
from src.detectors.base_detector import BaseDetector
from src.detection import Detection3D
from scripts.pointpillar_weights import download_pointpillars_weights

class LidarDetector(BaseDetector):
    """
    3D Object Detector basato unicamente su dati LiDAR.
    Integrazione per modello PointPillars con pre-filtraggio ROI Frustum.
    """
    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = 0.3,
        apply_roi_filter: bool = True,
        max_distance: float = 70.0,
        device: str = None
    ):
        super().__init__(conf_threshold=conf_threshold)
        self.apply_roi_filter = apply_roi_filter
        self.max_distance = max_distance
        self.model_path = model_path
        
        # Selezione automatica del device (GPU CUDA se disponibile, altrimenti CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"⚙️ LidarDetector inizializzato su device: {self.device}")

        # Caricamento automatico dei pesi e inizializzazione modello
        self.checkpoint = self._load_model(model_path)

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

    def filter_roi_frustum(self, points: np.ndarray, calib, img_shape: tuple) -> np.ndarray:
        """
        Filtra i punti LiDAR mantenendo unicamente quelli che ricadono nel campo visivo (FOV)
        della fotocamera e all'interno del range di profondità desiderato [0, max_distance].
        """
        if points.shape[0] == 0:
            return points

        # 1. Proietta i punti nel frame fotocamera e sui pixel 2D
        # pts_2d e depth contengono solo i punti con z > 0 (valid_mask == True)
        pts_2d, depth, valid_mask = calib.velo2img(points)
        
        h, w = img_shape[:2]
        
        # 2. Maschera per i punti proiettati che cadono nei limiti dell'immagine 2D e del range
        in_img_bounds = (
            (depth > 0.1) & (depth < self.max_distance) &
            (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) &
            (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
        )
        
        # 3. Ricostruisci la maschera booleana della dimensione originale di 'points'
        final_mask = np.zeros(len(points), dtype=bool)
        final_mask[valid_mask] = in_img_bounds
        
        return points[final_mask]

    def _preprocess(self, points: np.ndarray) -> dict:
        """
        Prepara la point cloud nel formato atteso da PointPillars (es. Voxelization / Pillarization).
        """
        # Formato di input standard LiDAR KITTI: (N, 4) -> [x, y, z, intensity]
        return {"pts": points}

    def _postprocess(self, raw_outputs, calib) -> List[Detection3D]:
        """
        Converte i tensor di output grezzi della rete (boxes 3D, scores, class_ids)
        in oggetti Detection3D standardizzati.
        """
        detections = []
        # Esempio di parsing degli output grezzi
        # bbox_3d grezzi: [x, y, z, dx, dy, dz, heading]
        
        # Nel caso mock/test restituisce lista vuota se non ci sono output
        if raw_outputs is None:
            return detections
            
        return detections

    def detect(self, sample: dict) -> List[Detection3D]:
        """
        Esegue la pipeline completa di detection 3D da LiDAR.
        """
        points = sample['points']
        calib = sample['calib']
        img_shape = sample['image'].shape

        # 1. Pre-filtraggio ROI Frustum (se abilitato)
        if self.apply_roi_filter:
            points = self.filter_roi_frustum(points, calib, img_shape)

        # 2. Pre-processing
        inputs = self._preprocess(points)

        # 3. Forward Pass (Infeerensa con PointPillars)
        if self.model is not None:
            # raw_outputs = self.model(inputs)
            raw_outputs = None
        else:
            raw_outputs = None

        # 4. Post-processing & Costruzione Detection3D
        detections = self._postprocess(raw_outputs, calib)

        return detections