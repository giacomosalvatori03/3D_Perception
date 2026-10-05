import os
import torch
import torchvision
import numpy as np
from typing import List

from .base_detector import BaseDetector
from .pillarization import Pillarizer
from .pointpillars_net import PointPillarsNet
from ..detection import Detection3D
from scripts.pointpillar_weights import download_pointpillars_weights

CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']

class LidarDetector(BaseDetector):
    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = 0.3,
        nms_iou_threshold: float = 0.2,
        device: str = None
    ):
        super().__init__(conf_threshold=conf_threshold)
        self.nms_iou_threshold = nms_iou_threshold
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"⚙️ LidarDetector inizializzato su device: {self.device}")

        self.pillarizer = Pillarizer()
        self.net = PointPillarsNet().to(self.device)
        self.anchors = self._generate_anchors().to(self.device)

        self._load_model(model_path)
        self.net.eval()

    def _generate_anchors(self) -> torch.Tensor:
        """Genera la griglia fisica di Anchor 3D (248, 216, 6, 7) per KITTI."""
        W_feat, H_feat = 216, 248
        x_min, y_min, x_max, y_max = 0.0, -39.68, 69.12, 39.68

        x_centers = torch.linspace(x_min + 0.16, x_max - 0.16, W_feat)
        y_centers = torch.linspace(y_min + 0.16, y_max - 0.16, H_feat)
        y_grid, x_grid = torch.meshgrid(y_centers, x_centers, indexing='ij')

        anchor_configs = [
            {'size': [3.9, 1.6, 1.56], 'z': -1.78, 'rots': [0.0, 1.5707963]}, # Car
            {'size': [0.8, 0.6, 1.73], 'z': -0.60, 'rots': [0.0, 1.5707963]}, # Pedestrian
            {'size': [1.76, 0.6, 1.73], 'z': -0.60, 'rots': [0.0, 1.5707963]}  # Cyclist
        ]

        anchors_list = []
        for cfg in anchor_configs:
            dx, dy, dz = cfg['size']
            z = cfg['z']
            for r in cfg['rots']:
                a = torch.zeros((H_feat, W_feat, 7))
                a[..., 0] = x_grid
                a[..., 1] = y_grid
                a[..., 2] = z
                a[..., 3] = dx
                a[..., 4] = dy
                a[..., 5] = dz
                a[..., 6] = r
                anchors_list.append(a)

        return torch.stack(anchors_list, dim=2) # (248, 216, 6, 7)

    def _load_model(self, model_path: str):
        if model_path is None:
            model_path = download_pointpillars_weights()

        if model_path is None or not os.path.exists(model_path):
            print("⚠️ Impossibile reperire i pesi del modello.")
            return

        print(f"📦 Caricamento pesi PyTorch da: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('state_dict', checkpoint)

            cleaned_state_dict = {}
            for k, v in state_dict.items():
                key = k
                if 'pts_middle_encoder.pfn_layers.0.' in key:
                    key = key.replace('pts_middle_encoder.pfn_layers.0.', 'pfn.')
                elif 'pts_backbone.' in key:
                    key = key.replace('pts_backbone.', 'backbone.')
                elif 'pts_neck.' in key:
                    key = key.replace('pts_neck.', 'backbone.')
                elif 'pts_bbox_head.conv_cls.' in key:
                    key = key.replace('pts_bbox_head.conv_cls.', 'conv_cls.')
                elif 'pts_bbox_head.conv_reg.' in key:
                    key = key.replace('pts_bbox_head.conv_reg.', 'conv_box.')
                cleaned_state_dict[key] = v

            missing, _ = self.net.load_state_dict(cleaned_state_dict, strict=False)
            print(f"✅ Checkpoint caricato! Layer mancanti (deve essere 0): {len(missing)}")
        except Exception as e:
            print(f"❌ Errore durante il caricamento pesi: {e}")

    def _preprocess(self, points: np.ndarray) -> dict:
        features, coords = self.pillarizer.process(points)
        if features is None:
            return None
        return {
            'pillar_features': features.to(self.device),
            'pillar_coords': coords.to(self.device)
        }

    def _postprocess(self, cls_preds, box_preds, calib) -> List[Detection3D]:
        cls_scores = torch.sigmoid(cls_preds[0].permute(1, 2, 0).view(248, 216, 6, 3))
        box_deltas = box_preds[0].permute(1, 2, 0).view(248, 216, 6, 7)

        anchor_cls_map = torch.tensor([0, 0, 1, 1, 2, 2], device=self.device)
        scores_for_anchors = cls_scores.gather(-1, anchor_cls_map.view(1, 1, 6, 1).expand(248, 216, 6, 1)).squeeze(-1)

        valid_mask = scores_for_anchors > self.conf_threshold
        if not valid_mask.any():
            return []

        valid_scores = scores_for_anchors[valid_mask]
        valid_deltas = box_deltas[valid_mask]
        valid_anchors = self.anchors[valid_mask]
        valid_cls_ids = anchor_cls_map.view(1, 1, 6).expand(248, 216, 6)[valid_mask]

        # Decodifica residui -> coordinate metriche assolute
        d_diag = torch.sqrt(valid_anchors[:, 3]**2 + valid_anchors[:, 4]**2)
        x_pred = valid_deltas[:, 0] * d_diag + valid_anchors[:, 0]
        y_pred = valid_deltas[:, 1] * d_diag + valid_anchors[:, 1]
        z_pred = valid_deltas[:, 2] * valid_anchors[:, 5] + valid_anchors[:, 2]
        dx_pred = valid_anchors[:, 3] * torch.exp(valid_deltas[:, 3])
        dy_pred = valid_anchors[:, 4] * torch.exp(valid_deltas[:, 4])
        dz_pred = valid_anchors[:, 5] * torch.exp(valid_deltas[:, 5])
        r_pred = valid_anchors[:, 6] + valid_deltas[:, 6]

        boxes_lidar = torch.stack([x_pred, y_pred, z_pred, dx_pred, dy_pred, dz_pred, r_pred], dim=-1)

        boxes_bev_2d = torch.zeros((len(boxes_lidar), 4), device=self.device)
        boxes_bev_2d[:, 0] = boxes_lidar[:, 0] - boxes_lidar[:, 4] / 2
        boxes_bev_2d[:, 1] = boxes_lidar[:, 1] - boxes_lidar[:, 3] / 2
        boxes_bev_2d[:, 2] = boxes_lidar[:, 0] + boxes_lidar[:, 4] / 2
        boxes_bev_2d[:, 3] = boxes_lidar[:, 1] + boxes_lidar[:, 3] / 2

        if len(valid_scores) > 1000:
            valid_scores, topk = torch.topk(valid_scores, 1000)
            boxes_bev_2d = boxes_bev_2d[topk]
            boxes_lidar = boxes_lidar[topk]
            valid_cls_ids = valid_cls_ids[topk]

        keep = torchvision.ops.nms(boxes_bev_2d, valid_scores, self.nms_iou_threshold)

        detections = []
        for idx in keep:
            box_lid = boxes_lidar[idx].cpu().numpy()
            score = float(valid_scores[idx].cpu().numpy())
            cls_name = CLASS_NAMES[int(valid_cls_ids[idx].cpu().numpy())]

            pt_lidar = box_lid[:3].reshape(1, 3)
            pt_cam = calib.velo2cam(pt_lidar)[0]

            det = Detection3D(
                obj_type=cls_name,
                dimensions_3d=[box_lid[5], box_lid[4], box_lid[3]], # h, w, l
                location_3d=pt_cam,
                rotation_y=float(box_lid[6]),
                score=score
            )
            detections.append(det)

        return detections

    def detect(self, sample: dict) -> List[Detection3D]:
        points = sample['points']
        calib = sample['calib']

        inputs = self._preprocess(points)
        if inputs is None:
            return []

        with torch.no_grad():
            cls_preds, box_preds = self.net(inputs['pillar_features'], inputs['pillar_coords'])

        return self._postprocess(cls_preds, box_preds, calib)