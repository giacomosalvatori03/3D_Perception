import os
import glob
import cv2
import matplotlib.pyplot as plt


class CameraVisualizer:
    """Visualizes 2D bounding box predictions (YOLO) and GT annotations
    directly on KITTI RGB images in a clean grid layout.
    """

    def __init__(
        self,
        data_path: str = "/content/drive/MyDrive/3D_Perception/data/kitti_validation",
    ):
        self.data_path = data_path

    @staticmethod
    def load_2d_boxes(txt_path: str, is_pred: bool = False, conf_thresh: float = 0.25) -> list:
        """Parses KITTI format txt file and extracts 2D bounding boxes."""
        boxes = []
        if not os.path.exists(txt_path):
            return boxes

        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] == "DontCare":
                    continue

                cls_name = parts[0]
                xmin, ymin, xmax, ymax = [float(x) for x in parts[4:8]]

                if is_pred and len(parts) >= 16:
                    score = float(parts[-1])
                elif is_pred and len(parts) == 15:
                    score = float(parts[14])
                else:
                    score = 1.0

                if is_pred and score < conf_thresh:
                    continue

                boxes.append({
                    "type": cls_name,
                    "bbox_2d": [xmin, ymin, xmax, ymax],
                    "score": score,
                })
        return boxes

    def visualize_experiment_2d(
        self,
        exp_dir: str,
        sample_ids: list = None,
        num_samples: int = 3,
        conf_thresh: float = 0.25,
        save_plots: bool = True,
    ):
        """Draws 2D GT and predicted boxes on RGB images and displays/saves a comparison grid."""
        pred_dir = os.path.join(exp_dir, "pred_labels")
        gt_dir = os.path.join(exp_dir, "gt_labels")
        save_vis_dir = os.path.join(exp_dir, "visualizations")

        if save_plots:
            os.makedirs(save_vis_dir, exist_ok=True)

        # If no sample_ids are provided, take the first num_samples from the GT files
        if sample_ids is None:
            gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
            sample_ids = [os.path.basename(f).replace(".txt", "") for f in gt_files[:num_samples]]

        n_samples = len(sample_ids)
        fig, axes = plt.subplots(n_samples, 1, figsize=(14, 4.5 * n_samples))
        if n_samples == 1:
            axes = [axes]

        for i, sid in enumerate(sample_ids):
            img_path = os.path.join(self.data_path, "image_2", f"{sid}.png")
            if not os.path.exists(img_path):
                print(f" Image not found for frame #{sid}: {img_path}")
                continue

            image_bgr = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            gt_path = os.path.join(gt_dir, f"{sid}.txt")
            pred_path = os.path.join(pred_dir, f"{sid}.txt")

            gt_boxes = self.load_2d_boxes(gt_path, is_pred=False)
            pred_boxes = self.load_2d_boxes(pred_path, is_pred=True, conf_thresh=conf_thresh)

            # Draw GT boxes in green and label them with class names
            for box in gt_boxes:
                if box["type"] in ["Car", "Pedestrian", "Cyclist"]:
                    x1, y1, x2, y2 = [int(v) for v in box["bbox_2d"]]
                    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        image_rgb,
                        f"GT: {box['type']}",
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            # Draw YOLOv8 predictions in red and label them with class names and confidence scores
            for box in pred_boxes:
                x1, y1, x2, y2 = [int(v) for v in box["bbox_2d"]]
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    image_rgb,
                    f"{box['type']} {box['score']:.2f}",
                    (x1, min(y2 + 15, image_rgb.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

            axes[i].imshow(image_rgb)
            axes[i].set_title(
                f"Frame #{sid} - Detection 2D (Green: Ground Truth | Red: YOLOv8 Pred)",
                fontsize=11,
            )
            axes[i].axis("off")

        plt.tight_layout()

        if save_plots:
            out_path = os.path.join(save_vis_dir, "summary_2d_grid.png")
            plt.savefig(out_path, bbox_inches="tight", dpi=150)
            print(f" 2D Grid Visualization saved to Drive: {out_path}")

        plt.show()
        plt.close(fig)