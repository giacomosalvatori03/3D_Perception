import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from IPython.display import display

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


class CameraEvaluator:
    """Evaluator for KITTI 2D Bounding Box Detection (mAP40)."""

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        iou_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.classes = classes or ["Car", "Pedestrian", "Cyclist"]
        self.diffs = ["Easy", "Moderate", "Hard"]
        # Standard KITTI 2D IoU thresholds
        self.iou_thresholds = iou_thresholds or {
            "Car": 0.70,
            "Pedestrian": 0.50,
            "Cyclist": 0.50,
        }

    @staticmethod
    def load_kitti_txt(file_path: str, is_pred: bool = False) -> Optional[dict]:
        """Loads a KITTI annotation file (.txt) and extracts 2D bounding boxes and scores."""
        if not os.path.exists(file_path):
            return None

        names, truncs, occs, bboxes, scores = [], [], [], [], []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or len(parts) < 8:
                    continue
                names.append(parts[0])
                truncs.append(float(parts[1]))
                occs.append(int(float(parts[2])))
                bboxes.append([float(x) for x in parts[4:8]])  # xmin, ymin, xmax, ymax

                if is_pred and len(parts) >= 15:
                    scores.append(float(parts[14]))

        return {
            "name": np.array(names, dtype=object),
            "truncated": np.array(truncs, dtype=np.float32),
            "occluded": np.array(occs, dtype=np.int32),
            "bbox": (
                np.array(bboxes, dtype=np.float32)
                if bboxes
                else np.zeros((0, 4), dtype=np.float32)
            ),
            "score": (
                np.array(scores, dtype=np.float32)
                if is_pred and scores
                else np.zeros(len(names), dtype=np.float32)
            ),
        }

    @staticmethod
    def compute_iou_2d(boxA: np.ndarray, boxB: np.ndarray) -> float:
        """Computes 2D Intersection over Union (IoU) between two bounding boxes [xmin, ymin, xmax, ymax]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        if interArea <= 0.0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        unionArea = boxAArea + boxBArea - interArea
        return float(interArea / unionArea) if unionArea > 0 else 0.0

    @staticmethod
    def is_gt_ignored(
        name: str,
        cls_name: str,
        h_2d: float,
        trunc: float,
        occ: int,
        diff_level: int,
    ) -> Tuple[bool, bool]:
        """Determines if a Ground Truth object belongs to the target class and whether it should be ignored based on difficulty."""
        ignored_classes = {
            "Car": ["Van", "DontCare"],
            "Pedestrian": ["Person_sitting", "DontCare"],
            "Cyclist": ["DontCare"],
        }

        if name in ignored_classes.get(cls_name, ["DontCare"]):
            return True, True

        if name != cls_name:
            return False, False

        is_easy = h_2d >= 40 and trunc <= 0.15 and occ == 0
        is_mod = h_2d >= 25 and trunc <= 0.30 and occ <= 1
        is_hard = h_2d >= 25 and trunc <= 0.50 and occ <= 2

        if diff_level == 0:
            ignored = not is_easy
        elif diff_level == 1:
            ignored = not is_mod
        else:
            ignored = not is_hard

        return True, ignored

    def eval_class_difficulty(
        self,
        gt_annos: List[dict],
        pred_annos: List[dict],
        cls_name: str,
        diff_level: int,
        iou_thresh: float,
    ) -> float:
        """Computes 2D mAP40 for a single class and difficulty level."""
        all_gt_boxes = []
        all_pred_boxes = []
        num_valid_gt = 0

        for img_idx, (gt, pred) in enumerate(zip(gt_annos, pred_annos)):
            if gt is None:
                continue

            for j in range(len(gt["name"])):
                bbox = gt["bbox"][j]
                h_2d = bbox[3] - bbox[1]
                trunc = gt["truncated"][j]
                occ = gt["occluded"][j]
                name = gt["name"][j]

                valid_match, ignored = self.is_gt_ignored(
                    name, cls_name, h_2d, trunc, occ, diff_level
                )
                if not valid_match:
                    continue

                if not ignored:
                    num_valid_gt += 1

                all_gt_boxes.append({
                    "img_idx": img_idx,
                    "bbox": bbox,
                    "ignored": ignored,
                    "matched": False,
                })

            if pred is not None:
                for k in range(len(pred["name"])):
                    if pred["name"][k] == cls_name:
                        all_pred_boxes.append({
                            "img_idx": img_idx,
                            "bbox": pred["bbox"][k],
                            "score": pred["score"][k],
                        })

        if num_valid_gt == 0 or not all_pred_boxes:
            return 0.0

        all_pred_boxes.sort(key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(all_pred_boxes))
        fp = np.zeros(len(all_pred_boxes))

        for p_idx, pred in enumerate(all_pred_boxes):
            img_idx = pred["img_idx"]
            best_iou = -1.0
            best_gt_idx = -1

            for g_idx, gt in enumerate(all_gt_boxes):
                if gt["img_idx"] != img_idx or gt["matched"]:
                    continue

                iou = self.compute_iou_2d(gt["bbox"], pred["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_thresh and best_gt_idx >= 0:
                matched_gt = all_gt_boxes[best_gt_idx]
                if not matched_gt["ignored"]:
                    tp[p_idx] = 1.0
                    matched_gt["matched"] = True
            else:
                fp[p_idx] = 1.0

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / num_valid_gt
        precisions = tp_cumsum / np.maximum(
            tp_cumsum + fp_cumsum, np.finfo(np.float64).eps
        )

        recall_thresholds = np.linspace(1 / 40, 1.0, 40)
        map40 = 0.0

        for r_thresh in recall_thresholds:
            prec_at_r = precisions[recalls >= r_thresh]
            if len(prec_at_r) > 0:
                map40 += np.max(prec_at_r)

        return float((map40 / 40.0) * 100.0)

    def evaluate(
        self,
        exp_dir: str,
        save_json: bool = True,
        save_exports: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Executes 2D mAP40 evaluation across all objects."""
        os.makedirs(exp_dir, exist_ok=True)
        pred_dir = os.path.join(exp_dir, "pred_labels")
        gt_dir = os.path.join(exp_dir, "gt_labels")

        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
        if not gt_files:
            raise FileNotFoundError(f"No .txt files found in {gt_dir}")

        gt_annos = []
        pred_annos = []

        for gt_path in gt_files:
            filename = os.path.basename(gt_path)
            pred_path = os.path.join(pred_dir, filename)

            gt_annos.append(self.load_kitti_txt(gt_path, is_pred=False))
            pred_annos.append(self.load_kitti_txt(pred_path, is_pred=True))

        results_dict = {}

        for cls in self.classes:
            cls_results = {}
            for diff_idx, diff in enumerate(self.diffs):
                map_val = self.eval_class_difficulty(
                    gt_annos,
                    pred_annos,
                    cls,
                    diff_idx,
                    self.iou_thresholds[cls],
                )
                cls_results[diff] = map_val
            results_dict[cls] = cls_results

        df = pd.DataFrame.from_dict(results_dict, orient="index")[self.diffs]
        df.index.name = "Class"

        if save_json:
            json_path = os.path.join(exp_dir, "map40_2d_results.json")
            with open(json_path, "w") as f:
                json.dump(results_dict, f, indent=4)

        if save_exports:
            exp_name = os.path.basename(os.path.normpath(exp_dir))
            df.to_csv(os.path.join(exp_dir, "map40_2d_results.csv"))
            df.round(2).to_markdown(os.path.join(exp_dir, "map40_2d_results.md"))
            df.round(2).to_latex(
                os.path.join(exp_dir, "map40_2d_results.tex"),
                caption=f"KITTI 2D Detection mAP40 - {exp_name}",
                label=f"tab:map40_2d_{exp_name}",
            )

        if verbose:
            exp_name = os.path.basename(os.path.normpath(exp_dir))
            print(f"\n OVERALL 2D DETECTION mAP40 [{exp_name}]\n")
            if HAS_IPYTHON:
                styled_df = df.style.format("{:.2f}%").set_caption(
                    f"KITTI 2D mAP40 Benchmark ({exp_name})"
                )
                display(styled_df)
            else:
                print(df.to_string())

            print(f"\n Reports saved in '{exp_dir}': .json, .csv, .md, .tex")

        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KITTI 2D mAP40 Evaluation via CLI"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Path to experiment folder containing pred_labels and gt_labels",
    )
    args = parser.parse_args()

    evaluator = CameraEvaluator()
    evaluator.evaluate(args.exp_dir)