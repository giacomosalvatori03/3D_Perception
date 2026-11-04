import os
import glob
import torch
import numpy as np

from src.fast_cloc_model import FastCLOCsAligner
from src.fast_cloc_dataset import FastCLOCsDataset


class FastCLOCsEvaluator:
    """
    Inference engine for Fast-CLOCs.
    Loads trained weights, updates 3D candidate confidence scores,
    and exports updated KITTI prediction text files for mAP40 evaluation.
    Supports optional frame filtering (e.g., using val_frames.txt) for strict evaluation.
    """
    def __init__(self, model_path: str, device: str = None):
        """
        Args:
            model_path (str): Path to trained .pth model checkpoint.
            device (str, optional): Computation device ('cuda' or 'cpu').
        """
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
        val_frames_file: str = None,
        conf_thresh: float = 0.10
    ):
        """
        Executes score alignment across frame prediction files and saves updated predictions.

        Args:
            pred_3d_dir (str): Path to input 3D candidate predictions.
            pred_2d_dir (str): Path to input 2D candidate predictions.
            output_dir (str): Directory to store updated prediction text files.
            val_frames_file (str, optional): Path to text file containing target validation frame IDs (e.g., val_frames.txt).
                                            If provided, only these frames will be evaluated.
            conf_thresh (float): Final confidence cutoff threshold for output boxes.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load allowed frame IDs if validation file is specified
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
        for filepath in frame_files:
            filename = os.path.basename(filepath)
            frame_id = os.path.splitext(filename)[0]
            
            preds_3d = dataset_parser._parse_3d_file(filepath)
            preds_2d = dataset_parser._parse_2d_file(os.path.join(pred_2d_dir, filename))
            
            out_filepath = os.path.join(output_dir, filename)
            
            if not preds_3d:
                # Create empty file if no 3D predictions exist
                open(out_filepath, 'w').close()
                processed_count += 1
                continue

            updated_lines = []
            
            for b3d in preds_3d:
                # Find matching 2D candidates
                best_2d_score = 0.0
                best_iou_2d = 0.0
                
                for b2d in preds_2d:
                    if b2d['class'] == b3d['class']:
                        iou = dataset_parser._compute_2d_iou(b3d['bbox_2d'], b2d['bbox_2d'])
                        if iou > best_iou_2d:
                            best_iou_2d = iou
                            best_2d_score = b2d['score']

                # Build input feature tensor
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

                # Neural Network Inference
                with torch.no_grad():
                    learned_score = self.model(feat_tensor).item()

                # Filter predictions below final cutoff threshold
                if learned_score >= conf_thresh:
                    # Reconstruct KITTI format string with updated learned score
                    b2d_str = " ".join([f"{p:.2f}" for p in b3d['bbox_2d']])
                    dim_str = " ".join([f"{p:.2f}" for p in b3d['dim']])
                    loc_str = " ".join([f"{p:.2f}" for p in b3d['loc']])
                    
                    line = (
                        f"{b3d['class']} 0.00 0 0.00 {b2d_str} "
                        f"{dim_str} {loc_str} 0.00 {learned_score:.4f}\n"
                    )
                    updated_lines.append(line)

            # Write updated predictions to file
            with open(out_filepath, 'w') as f:
                f.writelines(updated_lines)
                
            processed_count += 1

        print(f"Inference complete. Processed {processed_count} frames -> Output saved to: {output_dir}")


if __name__ == "__main__":
    # Example inference run on 16-beam degraded regime using validation frame filter
    evaluator = FastCLOCsEvaluator(model_path="models_checkpoints/fast_cloc_best.pth")
    evaluator.run_inference(
        pred_3d_dir="experiments/beam_16rings",
        pred_2d_dir="experiments_camera/yolov8m_baseline_2d",
        output_dir="experiments_fastcloc/fusion_beam_16rings",
        val_frames_file="models_checkpoints/val_frames.txt",
        conf_thresh=0.15
    )