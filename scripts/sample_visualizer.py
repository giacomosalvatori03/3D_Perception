import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.detection import Detection3D
from src.evaluator import KittiEvaluator
from src.kitti_dataset import KittiDataset
from src.visualizer import Visualizer


class SampleVisualizer:
    """Classe per identificare i frame più significativi di un esperimento

    e visualizzare GT e Predizioni direttamente nei notebook.
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
        """Legge un file .txt di predizione e lo riconverte in lista di oggetti Detection3D."""
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

    def find_interesting_samples(
        self, gt_dir: str, num_samples: str = 3
    ) -> list:
        """Analizza i Ground Truth e seleziona gli ID dei frame con maggiore densità e varietà di oggetti."""
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
        sample_scores = []

        for gt_path in gt_files:
            sample_id = os.path.basename(gt_path).replace(".txt", "")
            anno = KittiEvaluator.load_kitti_txt(gt_path, is_pred=False)
            if anno is None or len(anno["name"]) == 0:
                continue

            names = list(anno["name"])
            locs = anno["location"]

            # Criteri di interesse
            unique_classes = set(names) - {"DontCare", "Van", "Person_sitting"}
            num_classes = len(unique_classes)
            total_objects = len(
                [n for n in names if n in ["Car", "Pedestrian", "Cyclist"]]
            )
            far_objects = (
                sum(1 for loc in locs if loc[2] > 30.0) if len(locs) > 0 else 0
            )

            # Punteggio di complessità della scena
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

    def visualize_experiment(
        self,
        exp_dir: str,
        num_samples: int = 3,
        conf_thresh: float = 0.3,
        save_plots: bool = True,
    ):
        """Visualizza direttamente nel notebook i campioni più complessi di un esperimento."""
        pred_dir = os.path.join(exp_dir, "pred_labels")
        gt_dir = os.path.join(exp_dir, "gt_labels")
        save_vis_dir = os.path.join(exp_dir, "visualizations")

        if save_plots:
            os.makedirs(save_vis_dir, exist_ok=True)

        top_samples = self.find_interesting_samples(
            gt_dir, num_samples=num_samples
        )

        for info in top_samples:
            sid = info["sample_id"]
            print(
                f"\n🖼️ Frame #{sid} | Classi uniche: {info['num_classes']} | "
                f"Oggetti Totali: {info['total_objects']} | Distanti (>30m): {info['far_objects']}"
            )

            # Trova l'indice del sample nel dataset
            idx = None
            if hasattr(self.dataset, "sample_ids"):
                str_ids = [str(s).zfill(6) for s in self.dataset.sample_ids]
                if sid in str_ids:
                    idx = str_ids.index(sid)

            if idx is None:
                idx = int(sid)

            sample = self.dataset[idx]
            sample["sample_id"] = sid
            sample["labels"] = sample.get("gt_boxes", sample.get("labels", []))

            # Caricamento immagine RGB
            if "image" not in sample or sample["image"] is None:
                img_path = os.path.join(self.data_path, "image_2", f"{sid}.png")
                if os.path.exists(img_path):
                    img_bgr = cv2.imread(img_path)
                    sample["image"] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                else:
                    sample["image"] = np.zeros((375, 1242, 3), dtype=np.uint8)

            # Carica predizioni .txt
            pred_txt_path = os.path.join(pred_dir, f"{sid}.txt")
            predictions = self.load_predictions_as_detections(
                pred_txt_path, conf_thresh=conf_thresh
            )

            # Rendering della scena
            Visualizer.visualize_scene(sample, predictions, draw_gt=True)

            if save_plots:
                out_path = os.path.join(save_vis_dir, f"sample_{sid}_bev.png")
                plt.savefig(out_path, bbox_inches="tight", dpi=150)

            plt.show()  # Rendering immediato nell'output della cella Jupyter/Colab