import os
import urllib.request
import mmdet3d
import numpy as np
import torch
from mmdet3d.apis import inference_detector, init_model
from mmdet3d.structures import Box3DMode

from .base_detector import BaseDetector, Detection3D

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
WEIGHTS_URL = "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"


class LidarDetector(BaseDetector):

    def __init__(
        self,
        config_path=None,
        checkpoint_path=None,
        conf_threshold=0.3,
        device=None,
    ):
        super().__init__()

        if device is None or device == "cuda":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.conf_threshold = conf_threshold

        if config_path and os.path.exists(config_path):
            self.config_path = config_path
        else:
            mmdet3d_dir = os.path.dirname(mmdet3d.__file__)
            self.config_path = os.path.join(
                mmdet3d_dir,
                ".mim",
                "configs",
                "pointpillars",
                "pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py",
            )

        self.checkpoint_path = checkpoint_path or os.path.join(
            PROJECT_ROOT, "weights", "pointpillar_kitti.pth"
        )
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            print("📥 Download pesi ufficiali PointPillars in corso...")
            urllib.request.urlretrieve(WEIGHTS_URL, self.checkpoint_path)
            print("✅ Download completato!")

        print(f"⚡ Caricamento PointPillars su {self.device}")
        self.model = init_model(
            self.config_path, self.checkpoint_path, device=self.device
        )

    def _get_rt_matrix(self, calib):
        """Estrae in modo rigoroso la matrice 4x4 di trasformazione LiDAR -> Camera Rectified (R0_rect @ Tr_velo_to_cam)."""
        r0, v2c = None, None

        # 1. Estrazione flessibile da Dizionario o Oggetto Calibration
        if isinstance(calib, dict):
            r0 = (
                calib.get("R0_rect")
                or calib.get("R0")
                or calib.get("r0_rect")
                or calib.get("r0")
            )
            v2c = (
                calib.get("Tr_velo_to_cam")
                or calib.get("tr_velo_to_cam")
                or calib.get("V2C")
                or calib.get("Tr_v2c")
            )
        else:
            for attr in ["R0_rect", "R0", "r0_rect", "r0"]:
                if hasattr(calib, attr):
                    r0 = getattr(calib, attr)
                    break
            for attr in [
                "Tr_velo_to_cam",
                "tr_velo_to_cam",
                "V2C",
                "Tr_v2c",
                "velo2cam",
            ]:
                if hasattr(calib, attr):
                    v2c = getattr(calib, attr)
                    break

        # 2. Costruzione e validazione della matrice 4x4
        if r0 is not None and v2c is not None and not callable(v2c):
            r0_arr = np.asarray(r0, dtype=np.float32)
            v2c_arr = np.asarray(v2c, dtype=np.float32)

            if r0_arr.size == 9:
                r0_arr = r0_arr.reshape(3, 3)
            if v2c_arr.size == 12:
                v2c_arr = v2c_arr.reshape(3, 4)

            r0_4x4 = np.eye(4, dtype=np.float32)
            r0_4x4[:3, :3] = r0_arr[:3, :3]

            v2c_4x4 = np.eye(4, dtype=np.float32)
            v2c_4x4[:3, :4] = v2c_arr[:3, :4]

            # R0_rect @ Tr_velo_to_cam
            return (r0_4x4 @ v2c_4x4).astype(np.float32)

        # 3. Fallback tramite metodo di proiezione se presente
        if hasattr(calib, "velo2cam") and callable(getattr(calib, "velo2cam")):
            pts = np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
            )
            cam_pts = calib.velo2cam(pts)
            t = cam_pts[0]
            R = (cam_pts[1:] - t).T
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = R
            mat[:3, 3] = t
            return mat.astype(np.float32)

        # Se non troviamo la calibrazione, solleviamo un errore esplicito anziché fallire in silenzio
        raise AttributeError(
            "❌ Errore Calibrazione: Impossibile trovare R0_rect e/o Tr_velo_to_cam nell'oggetto calib!"
        )

    def detect(self, sample):
        points = sample["points"]
        calib = sample["calib"]

        # Inferenza PointPillars
        result, _ = inference_detector(self.model, points)
        pred_instances = result.pred_instances_3d

        scores = pred_instances.scores_3d.cpu().numpy()
        labels = pred_instances.labels_3d.cpu().numpy()
        bboxes_3d = pred_instances.bboxes_3d

        class_names = ["Car", "Pedestrian", "Cyclist"]
        detections = []

        if len(scores) == 0:
            return detections

        # Conversione coordinata dal riferimento LiDAR al riferimento Camera Rettificato
        rt_mat = self._get_rt_matrix(calib)
        bboxes_cam = bboxes_3d.convert_to(Box3DMode.CAM, rt_mat)
        cam_tensor = bboxes_cam.tensor.cpu().numpy()

        for i in range(len(scores)):
            score = float(scores[i])
            if score < self.conf_threshold:
                continue

            cls_id = int(labels[i])
            cls_name = (
                class_names[cls_id]
                if cls_id < len(class_names)
                else "Unknown"
            )

            box_cam = cam_tensor[i]

            # In CameraInstance3DBoxes di MMDetection3D:
            # box_cam[0,1,2] = centro (x, y, z) - con origine (0.5, 1.0, 0.5) ovvero già la base inferiore!
            # box_cam[3] = Width (w), box_cam[4] = Height (h), box_cam[5] = Length (l)
            # box_cam[6] = Rotation Yaw (ry)
            loc = [float(box_cam[0]), float(box_cam[1]), float(box_cam[2])]
            h = float(box_cam[4])
            w = float(box_cam[3])
            l = float(box_cam[5])
            ry = float(box_cam[6])

            det = Detection3D(
                obj_type=cls_name,
                dimensions_3d=[h, w, l],
                location_3d=loc,
                rotation_y=ry,
                score=score,
            )
            detections.append(det)

        return detections