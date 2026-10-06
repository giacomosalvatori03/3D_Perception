import glob
import os
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
        
        self.device = device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"⚙️ LidarDetector (mmdet3d) inizializzato su: {self.device}")

        # Config e Checkpoint predefiniti per PointPillars KITTI 3-Class
        self.config_path = config_path or 'mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
        self.checkpoint_path = checkpoint_path or 'weights/pointpillar_kitti.pth'

        self._download_weights_if_missing()
        
        print("📦 Caricamento modello mmdet3d...")
        self.model = init_model(self.config_path, self.checkpoint_path, device=self.device)
        print("✅ Modello mmdet3d caricato con successo!")

    # def _download_weights_if_missing(self):
    #     """Scarica i pesi ufficiali PointPillars tramite OpenMIM se non presenti."""
    #     if not os.path.exists(self.checkpoint_path):
    #         os.makedirs('weights', exist_ok=True)
    #         print("⬇️ Scaricamento pesi ufficiali PointPillars con OpenMIM...")
    #         try:
    #             download(package='mmdet3d', configs=['pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class'], dest='weights')
    #             # Rinomina il file scaricato in weights/pointpillar_kitti.pth
    #             downloaded_files = glob.glob('weights/*.pth')
    #             if downloaded_files:
    #                 os.rename(downloaded_files[0], self.checkpoint_path)
    #         except Exception as e:
    #             print(f"⚠️ Download tramite MIM fallito: {e}. Tento il download di fallback...")
    #             # fallback_url = "https://github.com/giacomosalvatori03/3D_Perception/releases/download/v1.0.0/pointpillar_kitti.pth"
    #             #torch.hub.download_url_to_file(fallback_url, self.checkpoint_path)

    def _download_weights_if_missing(self):
        """Scarica i pesi ufficiali se non presenti nella cartella weights/."""
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            url = "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"
            print(f"⬇️ Scaricamento pesi ufficiali PointPillars da OpenMMLab...")
            torch.hub.download_url_to_file(url, self.checkpoint_path)
            #https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth

    def detect(self, sample: dict) -> List[Detection3D]:
        """
        Esegue l'inferenza sui punti LiDAR del sample tramite mmdet3d 
        e converte i risultati in oggetti Detection3D.
        """
        points = sample['points']
        calib = sample['calib']
        
        # Salvataggio temporaneo del file .bin per mmdet3d
        temp_bin_path = '/tmp/temp_points.bin'
        points.astype(np.float32).tofile(temp_bin_path)

        # Inferenza mmdet3d
        with torch.no_grad():
            result, _ = inference_detector(self.model, temp_bin_path)

        if os.path.exists(temp_bin_path):
            os.remove(temp_bin_path)

        # Decodifica predizioni mmdet3d (DetDataSample)
        pred_instances = result.pred_instances_3d
        scores = pred_instances.scores_3d.cpu().numpy()
        labels = pred_instances.labels_3d.cpu().numpy()
        bboxes_3d = pred_instances.bboxes_3d.tensor.cpu().numpy() # [x, y, z, dx, dy, dz, yaw] in LiDAR frame

        class_names = ['Car', 'Pedestrian', 'Cyclist']
        detections = []

        for i in range(len(scores)):
            score = float(scores[i])
            if score < self.conf_threshold:
                continue

            cls_id = int(labels[i])
            cls_name = class_names[cls_id] if cls_id < len(class_names) else 'Unknown'

            box_lidar = bboxes_3d[i] # [x, y, z, dx, dy, dz, yaw]
            
            # Conversione coordinate centro 3D da LiDAR a Fotocamera (Rectified)
            pt_lidar = box_lidar[:3].reshape(1, 3)
            pt_cam = calib.velo2cam(pt_lidar)[0]

            det = Detection3D(
                obj_type=cls_name,
                dimensions_3d=[box_lidar[5], box_lidar[4], box_lidar[3]], # [h, w, l]
                location_3d=pt_cam,
                rotation_y=float(box_lidar[6]),
                score=score
            )
            detections.append(det)

        return detections