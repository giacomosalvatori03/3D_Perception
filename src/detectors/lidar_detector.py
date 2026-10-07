import os
import urllib.request
import numpy as np
import torch
import mmdet3d

from mmdet3d.structures import Box3DMode
from mmdet3d.apis import init_model, inference_detector
from .base_detector import BaseDetector, Detection3D

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
WEIGHTS_URL = "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"


class LidarDetector(BaseDetector):
    def __init__(self, config_path=None, checkpoint_path=None, conf_threshold=0.3, device=None):
        super().__init__()

        # Dispositivo GPU / CPU
        if device is None or device == 'cuda':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.conf_threshold = conf_threshold

        # Risoluzione Config nativo mmdet3d
        if config_path and os.path.exists(config_path):
            self.config_path = config_path
        else:
            mmdet3d_dir = os.path.dirname(mmdet3d.__file__)
            self.config_path = os.path.join(
                mmdet3d_dir, '.mim', 'configs', 'pointpillars',
                'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
            )

        # Risoluzione Checkpoint con download automatico
        self.checkpoint_path = checkpoint_path or os.path.join(PROJECT_ROOT, 'weights', 'pointpillar_kitti.pth')
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            print(f"📥 Download pesi ufficiali PointPillars in corso...")
            urllib.request.urlretrieve(WEIGHTS_URL, self.checkpoint_path)
            print("✅ Download completato!")

        print(f"⚡ Caricamento PointPillars su {self.device}")
        self.model = init_model(self.config_path, self.checkpoint_path, device=self.device)

    def _get_rt_matrix(self, calib):
        """Estrae la matrice 4x4 LiDAR -> Camera Rectified in modo robusto."""
        r0 = getattr(calib, 'R0', getattr(calib, 'R0_rect', getattr(calib, 'r0', None)))
        v2c = getattr(calib, 'V2C', getattr(calib, 'Tr_v2c', getattr(calib, 'tr_velo_to_cam', None)))

        if isinstance(calib, dict):
            r0 = r0 if r0 is not None else (calib.get('R0') or calib.get('R0_rect'))
            v2c = v2c if v2c is not None else (calib.get('V2C') or calib.get('Tr_v2c') or calib.get('tr_velo_to_cam'))

        if r0 is not None and v2c is not None:
            r0_4x4 = np.eye(4, dtype=np.float32)
            r0_4x4[:3, :3] = r0[:3, :3]
            v2c_4x4 = np.eye(4, dtype=np.float32)
            v2c_4x4[:3, :4] = v2c[:3, :4]
            return (r0_4x4 @ v2c_4x4).astype(np.float32)

        # Fallback tramite metodo velo2cam se presente nell'oggetto Calibration
        if hasattr(calib, 'velo2cam'):
            pts = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)
            cam_pts = calib.velo2cam(pts)
            t = cam_pts[0]
            R = (cam_pts[1:] - t).T
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = R
            mat[:3, 3] = t
            return mat.astype(np.float32)

        return np.eye(4, dtype=np.float32)

    def detect(self, sample):
        # Estrazione flessibile del percorso nuvola di punti
        # pts_path = None
        # for k in ['pts_path', 'velodyne_path', 'points_path', 'bin_path']:
        #     if k in sample and sample[k] is not None:
        #         pts_path = sample[k]
        #         break

        # if pts_path is None:
        #     raise KeyError(f"Nessun percorso nuvola di punti trovato nel sample. Chiavi trovate: {list(sample.keys())}")

        points = sample['points']
        calib = sample['calib']

        # Inferenza con PointPillars
        result, _ = inference_detector(self.model, points)
        pred_instances = result.pred_instances_3d

        scores = pred_instances.scores_3d.cpu().numpy()
        labels = pred_instances.labels_3d.cpu().numpy()
        bboxes_3d = pred_instances.bboxes_3d

        class_names = ['Car', 'Pedestrian', 'Cyclist']
        detections = []

        if len(scores) == 0:
            return detections

        # Conversione LiDAR -> Camera Frame
        rt_mat = self._get_rt_matrix(calib)
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

            # CORREZIONE MAPPATURA DIMENSIONI:
            # box_cam[4] = Height (h)
            # box_cam[3] = Width (w)
            # box_cam[5] = Length (l)
            h = float(box_cam[4])
            w = float(box_cam[3])
            l = float(box_cam[5])
            dims = [h, w, l]

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