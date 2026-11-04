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
    Loads trained weights, updates 3D candidate confidence scores,
    and exports updated KITTI prediction text files formatted for LidarEvaluator.
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
        print(f"FastCLOCsEvaluator initialized with weights from: {model_path}")

    def run_inference(
        self,
        pred_3d_dir: str,
        pred_2d_dir: str,
        output_dir: str,
        gt_3d_dir: str = None,
        val_frames_file: str = None,
        conf_thresh: float = 0.10
    ):
        """
        Executes score alignment, builds standard output hierarchy (pred_labels and gt_labels),
        and copies ground truth annotations if gt_3d_dir is provided.
        """
        # Compatible with KITTI evaluation scripts
        pred_out_dir = os.path.join(output_dir, "pred_labels")
        os.makedirs(pred_out_dir, exist_ok=True)

        if gt_3d_dir is not None:
            gt_out_dir = os.path.join(output_dir, "gt_labels")
            os.makedirs(gt_out_dir, exist_ok=True)

        # Validation frame filter: if provided, only process frames listed in this file
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
            gt_3d_dir=pred_3d_dir  # Dummy path for parsing functions
        )

        processed_count = 0
        copied_gt_count = 0

        for filepath in frame_files:
            filename = os.path.basename(filepath)
            frame_id = os.path.splitext(filename)[0]

            # Copy Ground Truth files to the experiment structure if gt_3d_dir is provided
            if gt_3d_dir is not None:
                src_gt = os.path.join(gt_3d_dir, filename)
                dst_gt = os.path.join(gt_out_dir, filename)
                if os.path.exists(src_gt):
                    shutil.copy(src_gt, dst_gt)
                    copied_gt_count += 1

            preds_3d = dataset_parser._parse_3d_file(filepath)
            preds_2d = dataset_parser._parse_2d_file(os.path.join(pred_2d_dir, filename))
            
            out_filepath = os.path.join(pred_out_dir, filename)
            
            if not preds_3d:
                open(out_filepath, 'w').close()
                processed_count += 1
                continue

            updated_lines = []
            
            for b3d in preds_3d:
                best_2d_score = 0.0
                best_iou_2d = 0.0
                best_2d_bbox = b3d['bbox_2d']  # Default fallback

                for b2d in preds_2d:
                    if b2d['class'] == b3d['class']:
                        iou = dataset_parser._compute_2d_iou(b3d['bbox_2d'], b2d['bbox_2d'])
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
                    learned_score = self.model(feat_tensor).item()

                if learned_score >= conf_thresh:
                    # Use the actual 2D BBox (best_2d_bbox) for KITTI line formatting
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