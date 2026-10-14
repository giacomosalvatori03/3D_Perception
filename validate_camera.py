import os
from src import KittiDataset
from src.detectors import CameraDetector
from tqdm import tqdm


class CameraValidator:
    """
    Runs 2D object detection validation on KITTI dataset and exports
    predictions and ground truths in standard KITTI format.
    """

    def __init__(
        self,
        data_path: str = "/content/drive/MyDrive/3D_Perception/data/kitti_validation",
        save_dir: str = "/content/drive/MyDrive/3D_Perception/experiments_camera",
        repo_id: str = "dronefreak/kitti-yolov8m",
        conf_thresh: float = 0.25,
        device: str = "cuda",
    ):
        self.data_path = data_path
        self.save_dir = save_dir
        self.conf_thresh = conf_thresh

        # Initialize dataset and 2D detector
        self.dataset = KittiDataset(data_root=data_path)
        self.detector = CameraDetector(
            repo_id=repo_id, conf_thresh=conf_thresh, device=device
        )

    @staticmethod
    def format_kitti_2d_pred_line(det: dict) -> str:
        """Formats 2D prediction in standard 15-value KITTI format with dummy 3D values."""
        xmin, ymin, xmax, ymax = det["bbox_2d"]
        return (
            f"{det['type']} 0.00 0 -10.00 "
            f"{xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} "
            f"-1.00 -1.00 -1.00 -1000.00 -1000.00 -1000.00 -10.00 "
            f"{det['score']:.4f}\n"
        )

    @staticmethod
    def format_gt_line(obj: dict) -> str:
        """Formats Ground Truth object into standard KITTI line."""
        loc_x, loc_y, loc_z = obj["location_3d"]
        h, w, l = obj["dimensions_3d"]
        xmin, ymin, xmax, ymax = obj["bbox_2d"]

        return (
            f"{obj['type']} {obj['truncation']:.2f} {int(obj['occlusion'])} {obj['alpha']:.2f} "
            f"{xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} "
            f"{h:.2f} {w:.2f} {l:.2f} "
            f"{loc_x:.2f} {loc_y:.2f} {loc_z:.2f} "
            f"{obj['rotation_y']:.2f}\n"
        )

    def validate(self, exp_name: str = "yolov8_camera_2d", max_samples: int = -1):
        """Runs validation inference over the dataset and exports label files."""
        exp_dir = os.path.join(self.save_dir, exp_name)
        pred_dir = os.path.join(exp_dir, "pred_labels")
        gt_dir = os.path.join(exp_dir, "gt_labels")

        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        total_samples = (
            len(self.dataset)
            if max_samples <= 0
            else min(max_samples, len(self.dataset))
        )

        print(
            f"\n Starting 2D Camera Validation [{exp_name.upper()}] | Samples: {total_samples}/{len(self.dataset)}"
        )
        print(f" Output destination: {exp_dir}")

        for i in tqdm(range(total_samples), desc="Processing 2D Frames"):
            sample = self.dataset[i]
            img_path = sample["sample_id"]

            # Retrieve frame ID formatted to 6 digits
            if hasattr(self.dataset, "sample_ids") and i < len(
                self.dataset.sample_ids
            ):
                sample_id = str(self.dataset.sample_ids[i]).zfill(6)
            else:
                sample_id = f"{i:06d}"

            # 1. Save Ground Truth label file
            gt_objs = sample["gt_boxes"]
            gt_txt_path = os.path.join(gt_dir, f"{sample_id}.txt")
            with open(gt_txt_path, "w") as f_gt:
                for obj in gt_objs:
                    f_gt.write(self.format_gt_line(obj))

            # 2. Run 2D Camera Inference and export predictions
            detections_2d = self.detector.detect(img_path)
            pred_txt_path = os.path.join(pred_dir, f"{sample_id}.txt")

            with open(pred_txt_path, "w") as f_pred:
                for det in detections_2d:
                    f_pred.write(self.format_kitti_2d_pred_line(det))

        print(f"\n 2D Validation completed successfully!")
        print(f"  • Predictions saved in: {pred_dir}")
        print(f"  • Ground Truths saved in: {gt_dir}")