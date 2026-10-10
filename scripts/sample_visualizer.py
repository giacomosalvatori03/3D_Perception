import glob
import os
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np

from scripts.lidar_evaluator import KittiEvaluator
from src.detection import Detection3D
from src.kitti_dataset import KittiDataset
from scripts.sparsifier import LidarSparsifier
from src.visualizer import Visualizer


class SampleVisualizer:
    """Utility class to identify complex KITTI frames and visualize GT vs Predictions

    in a compact grid format, correctly applying the experiment's LiDAR
    subsampling to the rendered point cloud.
    """

    def __init__(
        self,
        data_path: str = "/content/drive/MyDrive/3D_Perception/data/kitti_validation",
    ):
        self.data_path = data_path
        self.dataset = KittiDataset(data_root=data_path)

    @staticmethod
    def load_predictions_as_detections(
        pred_txt_path: str, conf_thresh: float = 0.3
    ) -> list:
        """Parses a KITTI prediction .txt file and converts annotations to a list of Detection3D objects."""
        detections = []
        if not os.path.exists(pred_txt_path):
            return detections

        with open(pred_txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 16:
                    continue
                obj_type = parts[0]
                score = float(parts[15])
                if score < conf_thresh:
                    continue

                h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
                x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
                ry = float(parts[14])

                det = Detection3D(
                    obj_type=obj_type,
                    dimensions_3d=[h, w, l],
                    location_3d=[x, y, z],
                    rotation_y=ry,
                    score=score,
                )
                detections.append(det)

        return detections

    @staticmethod
    def parse_subsampling_from_exp_dir(exp_dir: str):
        """Extracts subsampling mode and parameters from experiment directory name."""
        exp_name = os.path.basename(os.path.normpath(exp_dir)).lower()

        if "random" in exp_name:
            match = re.search(r"(\d+)", exp_name)
            if match:
                perc = float(match.group(1))
                return "random", perc / 100.0, 64
        elif "beam" in exp_name:
            match = re.search(r"(\d+)", exp_name)
            if match:
                beams = int(match.group(1))
                return "beam", 1.0, beams

        return "none", 1.0, 64

    def find_interesting_samples(
        self, gt_dir: str, num_samples: int = 3
    ) -> list:
        """Analyzes GT annotations and ranks frame IDs based on object density, class diversity, and distance."""
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
        sample_scores = []

        for gt_path in gt_files:
            sample_id = os.path.basename(gt_path).replace(".txt", "")
            anno = KittiEvaluator.load_kitti_txt(gt_path, is_pred=False)
            if anno is None or len(anno["name"]) == 0:
                continue

            names = list(anno["name"])
            locs = anno["location"]

            unique_classes = set(names) - {"DontCare", "Van", "Person_sitting"}
            num_classes = len(unique_classes)
            total_objects = len(
                [n for n in names if n in ["Car", "Pedestrian", "Cyclist"]]
            )
            far_objects = (
                sum(1 for loc in locs if loc[2] > 30.0) if len(locs) > 0 else 0
            )

            score = (num_classes * 10) + total_objects + (far_objects * 2)

            sample_scores.append({
                "sample_id": sample_id,
                "score": score,
                "num_classes": num_classes,
                "total_objects": total_objects,
                "far_objects": far_objects,
            })

        sample_scores.sort(key=lambda x: x["score"], reverse=True)
        return sample_scores[:num_samples]

    @staticmethod
    def render_frame_to_axes(
        ax_img,
        ax_bev,
        sample: dict,
        predictions: list,
        draw_gt: bool = True,
    ):
        """Renders 3D projected bounding boxes and BEV point cloud onto given Matplotlib subplots."""
        image = sample["image"].copy()
        calib = sample["calib"]
        points = sample["points"]
        gt_objects = sample["labels"]

        # 1. Render 3D boxes on RGB image
        if draw_gt and gt_objects:
            for obj in gt_objects:
                image = Visualizer.draw_box3d_on_image(
                    image=image,
                    location_3d=obj["location_3d"],
                    dimensions_3d=obj["dimensions_3d"],
                    rotation_y=obj["rotation_y"],
                    calib=calib,
                    color=(0, 255, 0),
                    thickness=2,
                    label=f"GT: {obj['type']}",
                )

        if predictions:
            for pred in predictions:
                image = Visualizer.draw_box3d_on_image(
                    image=image,
                    location_3d=pred.location_3d,
                    dimensions_3d=pred.dimensions_3d,
                    rotation_y=pred.rotation_y,
                    calib=calib,
                    color=(255, 0, 0),
                    thickness=2,
                    label=f"Pred: {pred.type} {pred.score:.2f}",
                )

        ax_img.imshow(image)
        ax_img.set_title(
            f"Frame #{sample['sample_id']} - 3D Camera Projection", fontsize=10
        )
        ax_img.axis("off")

        # 2. Render BEV point cloud (subsampled) and oriented 2D boxes
        pts_cam = calib.velo2cam(points)
        bev_mask = (
            (pts_cam[:, 2] > 0)
            & (pts_cam[:, 2] < 60)
            & (np.abs(pts_cam[:, 0]) < 25)
        )
        pts_bev = pts_cam[bev_mask]

        ax_bev.scatter(
            pts_bev[:, 0],
            pts_bev[:, 2],
            c=pts_bev[:, 2],
            cmap="viridis",
            s=0.3,
            alpha=0.4,
        )

        if draw_gt and gt_objects:
            for i, obj in enumerate(gt_objects):
                label = "Ground Truth" if i == 0 else None
                Visualizer.draw_bev_box(
                    ax=ax_bev,
                    location_3d=obj["location_3d"],
                    dimensions_3d=obj["dimensions_3d"],
                    rotation_y=obj["rotation_y"],
                    calib=calib,
                    color="g",
                    linewidth=1.5,
                    label=label,
                )

        if predictions:
            for i, pred in enumerate(predictions):
                label = "Prediction" if i == 0 else None
                Visualizer.draw_bev_box(
                    ax=ax_bev,
                    location_3d=pred.location_3d,
                    dimensions_3d=pred.dimensions_3d,
                    rotation_y=pred.rotation_y,
                    calib=calib,
                    color="r",
                    linewidth=1.8,
                    label=label,
                )

        ax_bev.set_xlim(-25, 25)
        ax_bev.set_ylim(0, 60)
        ax_bev.set_xlabel("X (m)", fontsize=9)
        ax_bev.set_ylabel("Z (m)", fontsize=9)
        ax_bev.set_title(
            f"Frame #{sample['sample_id']} - BEV View ({len(points)} pts)",
            fontsize=10,
        )
        ax_bev.grid(True, linestyle=":", alpha=0.5)
        ax_bev.legend(loc="upper right", fontsize=8)

    def visualize_experiment(
        self,
        exp_dir: str,
        num_samples: int = 3,
        conf_thresh: float = 0.3,
        save_plots: bool = True,
        grid_layout: bool = True,
    ):
        """Visualizes top complex frames in a compact grid format applying matching point cloud sparsification."""
        pred_dir = os.path.join(exp_dir, "pred_labels")
        gt_dir = os.path.join(exp_dir, "gt_labels")
        save_vis_dir = os.path.join(exp_dir, "visualizations")

        if save_plots:
            os.makedirs(save_vis_dir, exist_ok=True)

        top_samples = self.find_interesting_samples(
            gt_dir, num_samples=num_samples
        )
        if not top_samples:
            print("⚠️ No valid Ground Truth samples found for visualization.")
            return

        # Extract subsampling settings corresponding to this experiment
        sub_mode, sub_ratio, target_beams = self.parse_subsampling_from_exp_dir(
            exp_dir
        )

        n_samples = len(top_samples)

        if grid_layout:
            fig, axes = plt.subplots(
                n_samples, 2, figsize=(13, 3.8 * n_samples)
            )
            if n_samples == 1:
                axes = np.expand_dims(axes, axis=0)

            for i, info in enumerate(top_samples):
                sid = info["sample_id"]
                idx = self._resolve_sample_index(sid)

                sample = self.dataset[idx].copy()
                sample["sample_id"] = sid
                sample["labels"] = sample.get(
                    "gt_boxes", sample.get("labels", [])
                )
                self._ensure_image_key(sample, sid)

                # Apply experiment-specific LiDAR sparsification to rendering points
                sample["points"] = LidarSparsifier.sparsify(
                    sample["points"],
                    mode=sub_mode,
                    target_beams=target_beams,
                    keep_ratio=sub_ratio,
                )

                pred_txt_path = os.path.join(pred_dir, f"{sid}.txt")
                predictions = self.load_predictions_as_detections(
                    pred_txt_path, conf_thresh=conf_thresh
                )

                self.render_frame_to_axes(
                    axes[i, 0],
                    axes[i, 1],
                    sample,
                    predictions,
                    draw_gt=True,
                )

            plt.tight_layout()

            if save_plots:
                grid_out_path = os.path.join(
                    save_vis_dir, "summary_grid_bev.png"
                )
                plt.savefig(grid_out_path, bbox_inches="tight", dpi=150)
                print(f"✅ Saved compact grid visualization to Drive: {grid_out_path}")

            plt.show()
            plt.close(fig)

    def _resolve_sample_index(self, sid: str) -> int:
        """Resolves frame string ID to integer dataset index."""
        if hasattr(self.dataset, "sample_ids"):
            str_ids = [str(s).zfill(6) for s in self.dataset.sample_ids]
            if sid in str_ids:
                return str_ids.index(sid)
        return int(sid)

    def _ensure_image_key(self, sample: dict, sid: str):
        """Loads RGB image from disk if missing from dataset sample dictionary."""
        if "image" not in sample or sample["image"] is None:
            img_path = os.path.join(self.data_path, "image_2", f"{sid}.png")
            if os.path.exists(img_path):
                img_bgr = cv2.imread(img_path)
                sample["image"] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                sample["image"] = np.zeros((375, 1242, 3), dtype=np.uint8)