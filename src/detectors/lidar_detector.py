import os
import urllib.request
import torch
import numpy as np
import mmdet3d

from mmdet3d.structures import Box3DMode
from mmdet3d.apis import init_model, inference_detector
from .base_detector import BaseDetector, Detection3D

# Risoluzione dinamica della root del progetto (2 livelli sopra src/detectors)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# URL dei Pesi Ufficiali OpenMMLab (mmdet3d v1.x)
WEIGHTS_URL = "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"


def get_default_config_path():
    """Risolve il percorso del config PointPillars con albero _base_ integrato in mmdet3d."""
    mmdet3d_dir = os.path.dirname(mmdet3d.__file__)
    
    # Percorso standard MIM dentro site-packages di mmdet3d
    mim_config = os.path.join(
        mmdet3d_dir, '.mim', 'configs', 'pointpillars', 
        'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
    )
    if os.path.exists(mim_config):
        return mim_config
        
    # Percorso di fallback locale
    return os.path.join(PROJECT_ROOT, 'configs', 'pointpillars_kitti.py')


class LidarDetector(BaseDetector):
    def __init__(self, config_path=None, checkpoint_path=None, conf_threshold=0.3, device=None):
        super().__init__()
        
        # Gestione corretta dell'indice GPU/CPU
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda':
            self.device = 'cuda:0'
        else:
            self.device = device

        self.conf_threshold = conf_threshold

        # Percorsi
        self.config_path = config_path or get_default_config_path()
        self.checkpoint_path = checkpoint_path or os.path.join(PROJECT_ROOT, 'weights', 'pointpillar_kitti.pth')

        # Download automatico dei PESI se non presenti
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            print(f"📥 Pesi non trovati in '{self.checkpoint_path}'. Download in corso (~115 MB)...")
            urllib.request.urlretrieve(WEIGHTS_URL, self.checkpoint_path)
            print("✅ Pesi scaricati con successo!")

        print(f"🔧 Inizializzazione PointPillars su {self.device}")
        print(f"📄 Config: {self.config_path}")
        self.model = init_model(self.config_path, self.checkpoint_path, device=self.device)

    def detect(self, sample):
        pts_path = sample['pts_path']
        calib = sample['calib']

        # Inferenza
        result, _ = inference_detector(self.model, pts_path)
        pred_instances = result.pred_instances_3d

        scores = pred_instances.scores_3d.cpu().numpy()
        labels = pred_instances.labels_3d.cpu().numpy()
        bboxes_3d = pred_instances.bboxes_3d

        class_names = ['Car', 'Pedestrian', 'Cyclist']
        detections = []

        if len(scores) == 0:
            return detections

        # Conversione Nativa mmdet3d: LiDAR -> Camera Frame
        rect = calib.rect
        Trv2c = calib.Tr_v2c
        rt_mat = Trv2c @ rect.T

        bboxes_cam = bboxes_3d.convert_to(Box3DMode.CAM, rt_mat)
        cam_tensor = bboxes_cam.tensor.cpu().numpy()

        for i in range(len(scores)):
            score = float(scores[i])
            if score < self.conf_threshold:
                continue

            cls_id = int(labels[i])
            cls_name = class_names[cls_id] if cls_id < len(class_names) else 'Unknown'

            box_cam = cam_tensor[i]
            loc = [float(box_cam[0]), float(box_cam[1]), float(box_cam[2])]
            dims = [float(box_cam[4]), float(box_cam[5]), float(box_cam[3])]  # [h, w, l]
            ry = float(box_cam[6])

            det = Detection3D(
                obj_type=cls_name,
                dimensions_3d=dims,
                location_3d=loc,
                rotation_y=ry,
                score=score
            )
            detections.append(det)

        return detections