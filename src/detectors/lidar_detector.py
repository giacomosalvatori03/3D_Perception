import os
import torch
import numpy as np
from mmdet3d.structures import Box3DMode
from mmdet3d.apis import init_model, inference_detector
from .base_detector import BaseDetector, Detection3D

# Risoluzione dinamica del percorso radice del repository (2 livelli sopra src/detectors)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class LidarDetector(BaseDetector):
    def __init__(self, config_path=None, checkpoint_path=None, conf_threshold=0.3, device=None):
        super().__init__()
        
        # Gestione corretta dell'indice del dispositivo GPU/CPU
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda':
            self.device = 'cuda:0'
        else:
            self.device = device

        self.conf_threshold = conf_threshold

        # Percorsi assoluti dinamici per evitare errori di cartella
        self.config_path = config_path or os.path.join(PROJECT_ROOT, 'configs', 'pointpillars_kitti.py')
        self.checkpoint_path = checkpoint_path or os.path.join(PROJECT_ROOT, 'weights', 'pointpillar_kitti.pth')

        self.model = init_model(self.config_path, self.checkpoint_path, device=self.device)

    def detect(self, sample):
        pts_path = sample['pts_path']
        calib = sample['calib']

        # Inferenza PointPillars
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