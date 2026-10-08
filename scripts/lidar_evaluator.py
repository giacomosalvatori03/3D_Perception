import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from shapely.geometry import Polygon


class KittiEvaluator:
    """Evaluator nativo in Python/Shapely per le metriche KITTI 3D mAP40."""

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        iou_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.classes = classes or ["Car", "Pedestrian", "Cyclist"]
        self.diffs = ["Easy", "Moderate", "Hard"]
        self.iou_thresholds = iou_thresholds or {
            "Car": 0.70,
            "Pedestrian": 0.50,
            "Cyclist": 0.50,
        }

    @staticmethod
    def load_kitti_txt(file_path: str, is_pred: bool = False) -> Optional[dict]:
        """Legge un file .txt KITTI e restituisce le annotazioni in un dizionario NumPy."""
        if not os.path.exists(file_path):
            return None

        names, truncs, occs, alphas, bboxes, dims, locs, rys, scores = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or len(parts) < 14:
                    continue
                names.append(parts[0])
                truncs.append(float(parts[1]))
                occs.append(int(float(parts[2])))
                alphas.append(float(parts[3]))
                bboxes.append([float(x) for x in parts[4:8]])
                dims.append([float(x) for x in parts[8:11]])
                locs.append([float(x) for x in parts[11:14]])
                rys.append(float(parts[14]))
                if is_pred and len(parts) >= 16:
                    scores.append(float(parts[15]))

        return {
            "name": np.array(names, dtype=object),
            "truncated": np.array(truncs, dtype=np.float32),
            "occluded": np.array(occs, dtype=np.int32),
            "alpha": np.array(alphas, dtype=np.float32),
            "bbox": (
                np.array(bboxes, dtype=np.float32)
                if bboxes
                else np.zeros((0, 4), dtype=np.float32)
            ),
            "dimensions": (
                np.array(dims, dtype=np.float32)
                if dims
                else np.zeros((0, 3), dtype=np.float32)
            ),
            "location": (
                np.array(locs, dtype=np.float32)
                if locs
                else np.zeros((0, 3), dtype=np.float32)
            ),
            "rotation_y": np.array(rys, dtype=np.float32),
            "score": (
                np.array(scores, dtype=np.float32)
                if is_pred and scores
                else np.zeros(len(names), dtype=np.float32)
            ),
        }

    @staticmethod
    def compute_bev_polygon(
        loc: np.ndarray, dim: np.ndarray, ry: float
    ) -> Polygon:
        """Calcola il poligono BEV (xz) corretto nel riferimento Camera:

        dim = [h, w, l] -> w lungo l'asse X, l lungo l'asse Z.
        """
        h, w, l = dim
        x, y, z = loc

        x_corners = [w / 2, w / 2, -w / 2, -w / 2]
        z_corners = [l / 2, -l / 2, -l / 2, l / 2]

        R = np.array([[np.cos(ry), np.sin(ry)], [-np.sin(ry), np.cos(ry)]])

        corners = np.vstack([x_corners, z_corners])
        corners = np.dot(R, corners)
        corners[0, :] += x
        corners[1, :] += z

        return Polygon(zip(corners[0, :], corners[1, :]))

    def compute_iou_3d(
        self,
        gt_loc: np.ndarray,
        gt_dim: np.ndarray,
        gt_ry: float,
        pred_loc: np.ndarray,
        pred_dim: np.ndarray,
        pred_ry: float,
    ) -> float:
        """Calcola l'IoU 3D combinando l'overlap BEV e l'intersezione sull'asse Y."""
        gt_y_min, gt_y_max = gt_loc[1] - gt_dim[0], gt_loc[1]
        pred_y_min, pred_y_max = pred_loc[1] - pred_dim[0], pred_loc[1]

        inter_y = max(
            0.0, min(gt_y_max, pred_y_max) - max(gt_y_min, pred_y_min)
        )
        if inter_y <= 0:
            return 0.0

        try:
            poly_gt = self.compute_bev_polygon(gt_loc, gt_dim, gt_ry)
            poly_pred = self.compute_bev_polygon(pred_loc, pred_dim, pred_ry)

            if not poly_gt.is_valid or not poly_pred.is_valid:
                return 0.0

            inter_bev = poly_gt.intersection(poly_pred).area
            if inter_bev <= 0:
                return 0.0

            vol_inter = inter_bev * inter_y
            vol_gt = poly_gt.area * gt_dim[0]
            vol_pred = poly_pred.area * pred_dim[0]

            vol_union = vol_gt + vol_pred - vol_inter
            return float(vol_inter / vol_union) if vol_union > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def is_gt_ignored(
        name: str,
        cls_name: str,
        h_2d: float,
        trunc: float,
        occ: int,
        diff_level: int,
    ) -> Tuple[bool, bool]:
        """Determina se un GT appartiene alla classe e se deve essere ignorato per la difficoltà."""
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
        """Calcola mAP40 per una singola classe e livello di difficoltà."""
        all_gt_boxes = []
        all_pred_boxes = []
        num_valid_gt = 0

        for img_idx, (gt, pred) in enumerate(zip(gt_annos, pred_annos)):
            if gt is None:
                continue

            for j in range(len(gt["name"])):
                h_2d = gt["bbox"][j][3] - gt["bbox"][j][1]
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
                    "loc": gt["location"][j],
                    "dim": gt["dimensions"][j],
                    "ry": gt["rotation_y"][j],
                    "ignored": ignored,
                    "matched": False,
                })

            if pred is not None:
                for k in range(len(pred["name"])):
                    if pred["name"][k] == cls_name:
                        all_pred_boxes.append({
                            "img_idx": img_idx,
                            "loc": pred["location"][k],
                            "dim": pred["dimensions"][k],
                            "ry": pred["rotation_y"][k],
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

                iou = self.compute_iou_3d(
                    gt["loc"],
                    gt["dim"],
                    gt["ry"],
                    pred["loc"],
                    pred["dim"],
                    pred["ry"],
                )
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
        self, exp_dir: str, save_json: bool = True, verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """Esegue la valutazione completa mAP40 su una cartella esperimento."""
        pred_dir = os.path.join(exp_dir, "pred_labels")
        gt_dir = os.path.join(exp_dir, "gt_labels")

        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
        if not gt_files:
            raise FileNotFoundError(f"Nessun file .txt trovato in {gt_dir}")

        gt_annos = []
        pred_annos = []

        for gt_path in gt_files:
            filename = os.path.basename(gt_path)
            pred_path = os.path.join(pred_dir, filename)

            gt_annos.append(self.load_kitti_txt(gt_path, is_pred=False))
            pred_annos.append(self.load_kitti_txt(pred_path, is_pred=True))

        results_dict = {}

        if verbose:
            print("\n" + "=" * 55)
            print(" 📊 RISULTATI mAP40 3D DETECTION (Python / Shapely)")
            print("=" * 55)
            print(
                f"{'Class':<12} | {'Easy':<10} | {'Moderate':<10} | {'Hard':<10}"
            )
            print("-" * 55)

        for cls in self.classes:
            row_str = f"{cls:<12} | "
            cls_results = {}
            for diff_idx, diff in enumerate(self.diffs):
                map_val = self.eval_class_difficulty(
                    gt_annos,
                    pred_annos,
                    cls,
                    diff_idx,
                    self.iou_thresholds[cls],
                )
                row_str += f"{map_val:6.2f}%    | "
                cls_results[diff] = map_val

            if verbose:
                print(row_str)
            results_dict[cls] = cls_results

        if verbose:
            print("=" * 55)

        if save_json:
            report_json_path = os.path.join(exp_dir, "map40_results.json")
            with open(report_json_path, "w") as f:
                json.dump(results_dict, f, indent=4)
            if verbose:
                print(f"✅ Report salvato in: {report_json_path}")

        return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Valutazione KITTI mAP40 via CLI"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Percorso cartella esperimento",
    )
    args = parser.parse_args()

    evaluator = KittiEvaluator()
    evaluator.evaluate(args.exp_dir)