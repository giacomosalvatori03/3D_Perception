import os
import glob
import shutil
import torch
import numpy as np

from src.fast_cloc_model import FastCLOCsAligner
from src.fast_cloc_dataset import FastCLOCsDataset


class FastCLOCsEvaluator:
    """
    Inference engine for Fast-CLOCs with accurate 3D BEV NMS post-processing.
    """
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = FastCLOCsAligner(input_dim=8).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.class_map = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
     
        self.default_P2 = np.array([
            [721.5377, 0.0, 609.5593, 44.85728],
            [0.0, 721.5377, 172.8540, 0.2163761],
            [0.0, 0.0, 1.0, 0.002745884]
        ])
        print(f"FastCLOCsEvaluator initialized with weights from: {model_path}")

    def _get_bev_box(self, loc, dim, ry):
        """
        Calculates the 4 vertices of the BEV footprint (X-Z plane).
        Width (w) is associated with the X-axis, Length (l) is associated with the Z-axis (KITTI Camera 2).
        """
        x, _, z = loc
        _, w, l = dim
        
        corners = np.array([
            [ w/2,  l/2],
            [ w/2, -l/2],
            [-w/2, -l/2],
            [-w/2,  l/2]
        ])
        
        R = np.array([
            [np.cos(ry), np.sin(ry)],
            [-np.sin(ry), np.cos(ry)]
        ])
        
        return np.dot(corners, R) + np.array([x, z])

    def _compute_bev_iou(self, boxA, boxB):
        """Computes the BEV IoU between two 3D bounding boxes."""
        try:
            from shapely.geometry import Polygon
            polyA = Polygon(self._get_bev_box(boxA['loc'], boxA['dim'], boxA['ry']))
            polyB = Polygon(self._get_bev_box(boxB['loc'], boxB['dim'], boxB['ry']))
            
            if not polyA.is_valid or not polyB.is_valid:
                return 0.0
                
            inter = polyA.intersection(polyB).area
            union = polyA.area + polyB.area - inter
            return float(inter / max(union, 1e-6))
        except Exception:
            _, w_a, l_a = boxA['dim']
            _, w_b, l_b = boxB['dim']
            
            x1_a, x2_a = boxA['loc'][0] - w_a / 2, boxA['loc'][0] + w_a / 2
            z1_a, z2_a = boxA['loc'][2] - l_a / 2, boxA['loc'][2] + l_a / 2
            
            x1_b, x2_b = boxB['loc'][0] - w_b / 2, boxB['loc'][0] + w_b / 2
            z1_b, z2_b = boxB['loc'][2] - l_b / 2, boxB['loc'][2] + l_b / 2
            
            inter_x = max(0.0, min(x2_a, x2_b) - max(x1_a, x1_b))
            inter_z = max(0.0, min(z2_a, z2_b) - max(z1_a, z1_b))
            inter = inter_x * inter_z
            
            areaA = l_a * w_a
            areaB = l_b * w_b
            return float(inter / max(areaA + areaB - inter, 1e-6))

    def _apply_3d_nms(self, candidates, nms_thresh=0.25):
        """Applies BEV NMS per class to remove 3D duplicates."""
        if not candidates:
            return []

        sorted_candidates = sorted(candidates, key=lambda c: c['learned_score'], reverse=True)
        keep = []

        while len(sorted_candidates) > 0:
            current = sorted_candidates.pop(0)
            keep.append(current)

            remaining = []
            for item in sorted_candidates:
                if item['class'] == current['class']:
                    iou_bev = self._compute_bev_iou(current, item)
                    if iou_bev < nms_thresh:
                        remaining.append(item)
                else:
                    remaining.append(item)
            sorted_candidates = remaining

        return keep

    def run_inference(
        self,
        pred_3d_dir: str,
        pred_2d_dir: str,
        output_dir: str,
        gt_3d_dir: str = None,
        calib_dir: str = None,
        val_frames_file: str = None,
        conf_thresh: float = 0.01,
        nms_thresh: float = 0.25  
    ):
        pred_out_dir = os.path.join(output_dir, "pred_labels")
        os.makedirs(pred_out_dir, exist_ok=True)

        if gt_3d_dir is not None:
            gt_out_dir = os.path.join(output_dir, "gt_labels")
            os.makedirs(gt_out_dir, exist_ok=True)

        allowed_frame_ids = None
        if val_frames_file is not None and os.path.exists(val_frames_file):
            with open(val_frames_file, "r") as f:
                allowed_frame_ids = set(line.strip() for line in f.readlines() if line.strip())
            print(f"Filtering evaluation to {len(allowed_frame_ids)} validation frames specified in: {val_frames_file}")

        all_frame_files = sorted(glob.glob(os.path.join(pred_3d_dir, "*.txt")))
        
        if allowed_frame_ids is not None:
            frame_files = [
                f for f in all_frame_files 
                if os.path.splitext(os.path.basename(f))[0] in allowed_frame_ids
            ]
        else:
            frame_files = all_frame_files

        # Light-weight dataset parser instantiation without loading pairs into memory
        dataset_parser = FastCLOCsDataset(
            pred_3d_dir=pred_3d_dir,
            pred_2d_dir=pred_2d_dir,
            gt_3d_dir=gt_3d_dir,
            calib_dir=calib_dir,
            frame_ids=[]
        )

        copied_gt_count = 0
        processed_count = 0
        
        for filepath in frame_files:
            filename = os.path.basename(filepath)
            frame_id = os.path.splitext(filename)[0]

            if gt_3d_dir is not None:
                src_gt = os.path.join(gt_3d_dir, filename)
                dst_gt = os.path.join(gt_out_dir, filename)
                if os.path.exists(src_gt):
                    shutil.copy(src_gt, dst_gt)
                    copied_gt_count += 1

            P2 = dataset_parser._load_p2_matrix(frame_id)
            preds_3d = dataset_parser._parse_3d_file(filepath)
            preds_2d = dataset_parser._parse_2d_file(os.path.join(pred_2d_dir, filename))
            
            out_filepath = os.path.join(pred_out_dir, filename)
            
            if not preds_3d:
                open(out_filepath, 'w').close()
                processed_count += 1
                continue

            frame_candidates = []

            for b3d in preds_3d:
                proj_2d_bbox = dataset_parser._project_3d_to_2d(b3d['loc'], b3d['dim'], b3d['ry'], P2)
                
                best_2d_score = 0.0
                best_iou_2d = 0.0
                best_2d_bbox = proj_2d_bbox

                for b2d in preds_2d:
                    if b2d['class'] == b3d['class']:
                        iou = dataset_parser._compute_2d_iou(proj_2d_bbox, b2d['bbox_2d'])
                        if iou > best_iou_2d:
                            best_iou_2d = iou
                            best_2d_score = b2d['score']
                            best_2d_bbox = b2d['bbox_2d']

                vol = b3d['dim'][0] * b3d['dim'][1] * b3d['dim'][2]
                one_hot = [0.0, 0.0, 0.0]
                one_hot[b3d['class_idx']] = 1.0

                feat_vector = [
                    b3d['score'],
                    best_2d_score,
                    best_iou_2d,
                    b3d['depth'],
                    vol
                ] + one_hot

                feat_tensor = torch.tensor([feat_vector], dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    learned_score = self.model(feat_tensor, return_logits=False).item()

                if learned_score >= conf_thresh:
                    candidate_dict = b3d.copy()
                    candidate_dict['learned_score'] = learned_score
                    candidate_dict['best_2d_bbox'] = best_2d_bbox
                    frame_candidates.append(candidate_dict)

            # 1. 3D NMS in BEV post-processing
            filtered_candidates = self._apply_3d_nms(frame_candidates, nms_thresh=nms_thresh)

            # 2. Writing filtered predictions
            updated_lines = []
            for item in filtered_candidates:
                b2d_str = " ".join([f"{p:.2f}" for p in item['best_2d_bbox']])
                dim_str = " ".join([f"{p:.2f}" for p in item['dim']])
                loc_str = " ".join([f"{p:.2f}" for p in item['loc']])
                
                alpha = item.get('alpha', 0.00)
                ry = item.get('ry', 0.00)

                line = (
                    f"{item['class']} 0.00 0 {alpha:.2f} {b2d_str} "
                    f"{dim_str} {loc_str} {ry:.2f} {item['learned_score']:.4f}\n"
                )
                updated_lines.append(line)

            with open(out_filepath, 'w') as f:
                f.writelines(updated_lines)
                
            processed_count += 1

        print(f"Inference complete. Processed {processed_count} frames | Copied {copied_gt_count} GT files -> Output saved to: {output_dir}")