import os
import glob
import shutil
import torch
import numpy as np

from src.fast_cloc_model import FastCLOCsAligner
from src.fast_cloc_dataset import FastCLOCsDataset


class FastCLOCsEvaluator:
    """
    Inference engine for Fast-CLOCs.
    Loads trained weights, projects 3D candidate bounding boxes into 2D image coordinates,
    computes accurate 2D/3D IoU with YOLO predictions, and updates KITTI prediction text files.
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
     
        # P2 standard KITTI used as a fallback if calib file is not specified
        self.default_P2 = np.array([
            [721.5377, 0.0, 609.5593, 44.85728],
            [0.0, 721.5377, 172.8540, 0.2163761],
            [0.0, 0.0, 1.0, 0.002745884]
        ])
        print(f"FastCLOCsEvaluator initialized with weights from: {model_path}")

    def _load_p2_matrix(self, calib_filepath: str):
        """Loads the P2 matrix from the KITTI calibration file if available."""
        if calib_filepath and os.path.exists(calib_filepath):
            with open(calib_filepath, 'r') as f:
                for line in f:
                    if line.startswith("P2:"):
                        return np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
        return self.default_P2

    def _project_3d_to_2d(self, loc, dim, ry, P2):
        """Projects a 3D bounding box (loc, dim, ry) into the 2D image plane [xmin, ymin, xmax, ymax]."""
        h, w, l = dim
        x, y, z = loc
        
        # Vertices of 3D bounding box in the object coordinate system
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        
        # Rotation matrix around the Y-axis
        R = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        corners_3d = np.vstack([x_corners, y_corners, z_corners])
        corners_3d_cam = np.dot(R, corners_3d) + np.array([[x], [y], [z]])
        
        # Homogeneous coordinates for projection
        corners_3d_hom = np.vstack([corners_3d_cam, np.ones((1, 8))])
        
        # 2D Projection using the P2 matrix
        pts_2d = np.dot(P2, corners_3d_hom)
        pts_2d[0, :] /= pts_2d[2, :]
        pts_2d[1, :] /= pts_2d[2, :]
        
        xmin = float(np.min(pts_2d[0, :]))
        xmax = float(np.max(pts_2d[0, :]))
        ymin = float(np.min(pts_2d[1, :]))
        ymax = float(np.max(pts_2d[1, :]))
        
        return [xmin, ymin, xmax, ymax]

    def run_inference(
        self,
        pred_3d_dir: str,
        pred_2d_dir: str,
        output_dir: str,
        gt_3d_dir: str = None,
        calib_dir: str = None,
        val_frames_file: str = None,
        conf_thresh: float = 0.01
    ):
        pred_out_dir = os.path.join(output_dir, "pred_labels")
        os.makedirs(pred_out_dir, exist_ok=True)

        if gt_3d_dir is not None:
            gt_out_dir = os.path.join(output_dir, "gt_labels")
            os.makedirs(gt_out_dir, exist_ok=True)

        allowed_frame_ids = None
        if val_frames_file is not None:
            if os.path.exists(val_frames_file):
                with open(val_frames_file, "r") as f:
                    allowed_frame_ids = set(line.strip() for line in f.readlines() if line.strip())
                print(f"Filtering evaluation to {len(allowed_frame_ids)} validation frames specified in: {val_frames_file}")
            else:
                print(f"Warning: Specified val_frames_file '{val_frames_file}' not found. Processing all available frames.")

        all_frame_files = sorted(glob.glob(os.path.join(pred_3d_dir, "*.txt")))
        
        if allowed_frame_ids is not None:
            frame_files = [
                f for f in all_frame_files 
                if os.path.splitext(os.path.basename(f))[0] in allowed_frame_ids
            ]
        else:
            frame_files = all_frame_files

        dataset_parser = FastCLOCsDataset(
            pred_3d_dir=pred_3d_dir,
            pred_2d_dir=pred_2d_dir,
            gt_3d_dir=pred_3d_dir
        )

        processed_count = 0
        copied_gt_count = 0

        for filepath in frame_files:
            filename = os.path.basename(filepath)
            frame_id = os.path.splitext(filename)[0]

            if gt_3d_dir is not None:
                src_gt = os.path.join(gt_3d_dir, filename)
                dst_gt = os.path.join(gt_out_dir, filename)
                if os.path.exists(src_gt):
                    shutil.copy(src_gt, dst_gt)
                    copied_gt_count += 1

            # Load P2 matrix for the current frame
            calib_filepath = os.path.join(calib_dir, f"{frame_id}.txt") if calib_dir else None
            P2 = self._load_p2_matrix(calib_filepath)

            preds_3d = dataset_parser._parse_3d_file(filepath)
            preds_2d = dataset_parser._parse_2d_file(os.path.join(pred_2d_dir, filename))
            
            out_filepath = os.path.join(pred_out_dir, filename)
            
            if not preds_3d:
                open(out_filepath, 'w').close()
                processed_count += 1
                continue

            updated_lines = []
            
            for b3d in preds_3d:
                # 1. Project the 3D box into the 2D image space
                proj_2d_bbox = self._project_3d_to_2d(b3d['loc'], b3d['dim'], b3d['ry'], P2)
                
                best_2d_score = 0.0
                best_iou_2d = 0.0
                best_2d_bbox = proj_2d_bbox  # Fallback

                # 2. Compute IoU between projected 3D box and YOLO 2D predictions
                for b2d in preds_2d:
                    if b2d['class'] == b3d['class']:
                        iou = dataset_parser._compute_2d_iou(proj_2d_bbox, b2d['bbox_2d'])
                        if iou > best_iou_2d:
                            best_iou_2d = iou
                            best_2d_score = b2d['score']
                            best_2d_bbox = b2d['bbox_2d']

                # 3. Construct feature vector for Fast-CLOCs
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
                    learned_score = self.model(feat_tensor).item()

                if learned_score >= conf_thresh:
                    b2d_str = " ".join([f"{p:.2f}" for p in best_2d_bbox])
                    dim_str = " ".join([f"{p:.2f}" for p in b3d['dim']])
                    loc_str = " ".join([f"{p:.2f}" for p in b3d['loc']])
                    
                    alpha = b3d.get('alpha', 0.00)
                    ry = b3d.get('ry', 0.00)

                    line = (
                        f"{b3d['class']} 0.00 0 {alpha:.2f} {b2d_str} "
                        f"{dim_str} {loc_str} {ry:.2f} {learned_score:.4f}\n"
                    )
                    updated_lines.append(line)

            with open(out_filepath, 'w') as f:
                f.writelines(updated_lines)
                
            processed_count += 1

        print(f"Inference complete. Processed {processed_count} frames | Copied {copied_gt_count} GT files -> Output saved to: {output_dir}")