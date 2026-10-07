import os
import glob
import json
import argparse
import numpy as np
from shapely.geometry import Polygon


def parse_args():
    parser = argparse.ArgumentParser(description="Valutazione KITTI mAP40 in Python Puro")
    parser.add_argument("--exp_dir", type=str, required=True, help="Percorso cartella esperimento su Drive")
    return parser.parse_args()


def load_kitti_txt(file_path, is_pred=False):
    """Legge un file .txt KITTI e restituisce le annotazioni in formato NumPy."""
    if not os.path.exists(file_path):
        return None

    names, truncs, occs, alphas, bboxes, dims, locs, rys, scores = [], [], [], [], [], [], [], [], []
    with open(file_path, 'r') as f:
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
        'name': np.array(names, dtype=object),
        'truncated': np.array(truncs, dtype=np.float32),
        'occluded': np.array(occs, dtype=np.int32),
        'alpha': np.array(alphas, dtype=np.float32),
        'bbox': np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32),
        'dimensions': np.array(dims, dtype=np.float32) if dims else np.zeros((0, 3), dtype=np.float32),
        'location': np.array(locs, dtype=np.float32) if locs else np.zeros((0, 3), dtype=np.float32),
        'rotation_y': np.array(rys, dtype=np.float32),
        'score': np.array(scores, dtype=np.float32) if is_pred and scores else np.zeros(len(names), dtype=np.float32)
    }


def compute_bev_polygon(loc, dim, ry):
    """Calcola il poligono BEV (Bird's Eye View) xz dato il centro, le dimensioni e rotation_y."""
    h, w, l = dim
    x, y, z = loc

    x_corners = [l/2, l/2, -l/2, -l/2]
    z_corners = [w/2, -w/2, -w/2, w/2]

    R = np.array([
        [np.cos(ry), np.sin(ry)],
        [-np.sin(ry), np.cos(ry)]
    ])

    corners = np.vstack([x_corners, z_corners])
    corners = np.dot(R, corners)
    corners[0, :] += x
    corners[1, :] += z

    return Polygon(zip(corners[0, :], corners[1, :]))


def compute_iou_3d(gt_loc, gt_dim, gt_ry, pred_loc, pred_dim, pred_ry):
    """Calcola l'IoU 3D combinando l'overlap BEV con l'intersezione sull'asse Y."""
    # 1. Overlap Y (Altezza)
    gt_y_min, gt_y_max = gt_loc[1] - gt_dim[0], gt_loc[1]
    pred_y_min, pred_y_max = pred_loc[1] - pred_dim[0], pred_loc[1]

    inter_y = max(0.0, min(gt_y_max, pred_y_max) - max(gt_y_min, pred_y_min))
    if inter_y <= 0:
        return 0.0

    # 2. Overlap BEV (Shapely)
    try:
        poly_gt = compute_bev_polygon(gt_loc, gt_dim, gt_ry)
        poly_pred = compute_bev_polygon(pred_loc, pred_dim, pred_ry)

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


def eval_class_difficulty(gt_annos, pred_annos, cls_name, diff_level, iou_thresh):
    """Calcola l'mAP40 per una specifica classe e livello di difficoltà."""
    # Definizioni filtri KITTI
    # diff_level: 0=Easy, 1=Moderate, 2=Hard
    all_gt_boxes = []
    all_pred_boxes = []

    for img_idx, (gt, pred) in enumerate(zip(gt_annos, pred_annos)):
        if gt is None:
            continue

        # Filtra GT per classe e difficoltà
        for j in range(len(gt['name'])):
            if gt['name'][j] != cls_name:
                continue

            h_2d = gt['bbox'][j][3] - gt['bbox'][j][1]
            trunc = gt['truncated'][j]
            occ = gt['occluded'][j]

            # Criteri di difficoltà KITTI
            if diff_level == 0 and not (h_2d >= 40 and trunc <= 0.15 and occ == 0):
                continue
            elif diff_level == 1 and not (h_2d >= 25 and trunc <= 0.30 and occ <= 1):
                continue
            elif diff_level == 2 and not (h_2d >= 25 and trunc <= 0.50 and occ <= 2):
                continue

            all_gt_boxes.append({
                'img_idx': img_idx,
                'loc': gt['location'][j],
                'dim': gt['dimensions'][j],
                'ry': gt['rotation_y'][j],
                'matched': False
            })

        # Filtra Predizioni per classe
        if pred is not None:
            for k in range(len(pred['name'])):
                if pred['name'][k] == cls_name:
                    all_pred_boxes.append({
                        'img_idx': img_idx,
                        'loc': pred['location'][k],
                        'dim': pred['dimensions'][k],
                        'ry': pred['rotation_y'][k],
                        'score': pred['score'][k]
                    })

    if not all_gt_boxes:
        return 0.0

    if not all_pred_boxes:
        return 0.0

    # Ordina predizioni per score decrescente
    all_pred_boxes.sort(key=lambda x: x['score'], reverse=True)

    tp = np.zeros(len(all_pred_boxes))
    fp = np.zeros(len(all_pred_boxes))

    # Matching Greedy
    for p_idx, pred in enumerate(all_pred_boxes):
        img_idx = pred['img_idx']
        best_iou = -1.0
        best_gt_idx = -1

        for g_idx, gt in enumerate(all_gt_boxes):
            if gt['img_idx'] != img_idx or gt['matched']:
                continue

            iou = compute_iou_3d(gt['loc'], gt['dim'], gt['ry'], pred['loc'], pred['dim'], pred['ry'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            tp[p_idx] = 1.0
            all_gt_boxes[best_gt_idx]['matched'] = True
        else:
            fp[p_idx] = 1.0

    # Calcolo curve Precision-Recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / len(all_gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Calcolo mAP40 (40 punti di recall campionati)
    recall_thresholds = np.linspace(1/40, 1.0, 40)
    map40 = 0.0

    for r_thresh in recall_thresholds:
        prec_at_r = precisions[recalls >= r_thresh]
        if len(prec_at_r) > 0:
            map40 += np.max(prec_at_r)

    return float((map40 / 40.0) * 100.0)


def main():
    args = parse_args()
    pred_dir = os.path.join(args.exp_dir, "pred_labels")
    gt_dir = os.path.join(args.exp_dir, "gt_labels")

    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    if not gt_files:
        raise FileNotFoundError(f"Nessun file .txt trovato in {gt_dir}")

    gt_annos = []
    pred_annos = []

    for gt_path in gt_files:
        filename = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, filename)

        gt_annos.append(load_kitti_txt(gt_path, is_pred=False))
        pred_annos.append(load_kitti_txt(pred_path, is_pred=True))

    classes = ['Car', 'Pedestrian', 'Cyclist']
    diffs = ['Easy', 'Moderate', 'Hard']
    iou_thresholds = {'Car': 0.70, 'Pedestrian': 0.50, 'Cyclist': 0.50}

    print("\n" + "="*55)
    print(" 📊 RISULTATI mAP40 3D DETECTION (Python Puro / Shapely)")
    print("="*55)
    print(f"{'Class':<12} | {'Easy':<10} | {'Moderate':<10} | {'Hard':<10}")
    print("-" * 55)

    results_dict = {}

    for cls in classes:
        row_str = f"{cls:<12} | "
        cls_results = {}
        for diff_idx, diff in enumerate(diffs):
            map_val = eval_class_difficulty(gt_annos, pred_annos, cls, diff_idx, iou_thresholds[cls])
            row_str += f"{map_val:6.2f}%    | "
            cls_results[diff] = map_val
        print(row_str)
        results_dict[cls] = cls_results

    print("="*55)

    # Salvataggio JSON su Drive
    report_json_path = os.path.join(args.save_dir if hasattr(args, 'save_dir') else args.exp_dir, "map40_results.json")
    with open(report_json_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"\n✅ Report salvato in: {report_json_path}")


if __name__ == "__main__":
    main()