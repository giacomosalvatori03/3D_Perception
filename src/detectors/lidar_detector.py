import os
import glob
import torch
import numpy as np
from typing import List

from mmdet3d.apis import init_model, inference_detector
from mmdet3d.utils import register_all_modules
from mim.commands import download

from .base_detector import BaseDetector
from ..detection import Detection3D

class LidarDetector(BaseDetector):
    """
    Adapter per 3D Object Detection basato sulla libreria mmdet3d.
    Carica la configurazione e i pesi ufficiali OpenMMLab per PointPillars (KITTI 3-class).
    """
    def __init__(
        self,
        config_path: str = None,
        checkpoint_path: str = None,
        conf_threshold: float = 0.3,
        device: str = None
    ):
        super().__init__(conf_threshold=conf_threshold)
        
        # Inizializza i moduli interni di OpenMMLab
        register_all_modules(init_default_scope=True)
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"⚙️ LidarDetector (mmdet3d) inizializzato su: {self.device}")

        self.checkpoint_path = checkpoint_path or 'weights/pointpillar_kitti.pth'
        self.config_path = self._resolve_config_path(config_path)

        self._ensure_weights_and_config()
        
        print("📦 Caricamento modello mmdet3d...")
        self.model = init_model(self.config_path, self.checkpoint_path, device=self.device)
        print("✅ Modello mmdet3d caricato con successo!")

    def _resolve_config_path(self, custom_path: str) -> str:
        if custom_path and os.path.exists(custom_path):
            return custom_path

        candidates = [
            '/content/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py',
            'weights/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py',
            'mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return 'weights/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'

    def _ensure_weights_and_config(self):
        """Scarica e gestisce la presenza sia del file .py che dei pesi .pth."""
        os.makedirs('weights', exist_ok=True)
        
        need_download = not os.path.exists(self.checkpoint_path) or not os.path.exists(self.config_path)
        if need_download:
            print("⬇️ Download file di configurazione e pesi ufficiali con OpenMIM...")
            try:
                download(package='mmdet3d', configs=['pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class'], where='weights')
                
                pth_files = glob.glob('weights/*.pth')
                if pth_files and not os.path.exists(self.checkpoint_path):
                    os.rename(pth_files[0], self.checkpoint_path)

                py_files = glob.glob('weights/*.py')
                if py_files and not os.path.exists(self.config_path):
                    self.config_path = py_files[0]
            except Exception as e:
                print(f"⚠️ Download tramite MIM non riuscito: {e}. Uso fallback GitHub...")
                if not os.path.exists(self.checkpoint_path):
                    fallback_url = "https://github.com/giacomosalvatori03/3D_Perception/releases/download/v1.0.0/pointpillar_kitti.pth"
                    torch.hub.download_url_to_file(fallback_url, self.checkpoint_path)

    def detect(self, sample: dict) -> List[Detection3D]:
        """Esegue l'inferenza e converte le predizioni in oggetti Detection3D."""
        points = sample['points']
        calib = sample['calib']
        
        temp_bin_path = '/tmp/temp_points.bin'
        points.astype(np.float32).tofile(temp_bin_path)

        with torch.no_grad():
            result = inference_detector(self.model, temp_bin_path)
            if isinstance(result, tuple):
                result = result[0]

        if os.path.exists(temp_bin_path):
            os.remove(temp_bin_path)

        pred_instances = result.pred_instances_3d
        scores = pred_instances.scores_3d.cpu().numpy()
        labels = pred_instances.labels_3d.cpu().numpy()
        bboxes_3d = pred_instances.bboxes_3d.tensor.cpu().numpy()

        class_names = ['Car', 'Pedestrian', 'Cyclist']
        detections = []

        for i in range(len(scores)):
            score = float(scores[i])
            if score < self.conf_threshold:
                continue

            cls_id = int(labels[i])
            cls_name = class_names[cls_id] if cls_id < len(class_names) else 'Unknown'

            box_lidar = bboxes_3d[i]
            
            pt_lidar = box_lidar[:3].reshape(1, 3)
            pt_cam = calib.velo2cam(pt_lidar)[0]

            det = Detection3D(
                obj_type=cls_name,
                dimensions_3d=[box_lidar[5], box_lidar[4], box_lidar[3]],
                location_3d=pt_cam,
                rotation_y=float(box_lidar[6]),
                score=score
            )
            detections.append(det)

        return detections