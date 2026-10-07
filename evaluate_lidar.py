import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from mmdet3d.evaluation.functional.kitti_utils import kitti_eval
from src import KittiDataset, LidarDetector

def parse_args():
    parser = argparse.ArgumentParser(description="Valutazione KITTI con Sparsificazione")
    parser.add_argument("--data_path", type=str, default="/content/drive/MyDrive/3D_Perception/data/kitti_validation")
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/3D_Perception/experiments")
    
    # Parametri di Sparsificazione
    parser.add_argument("--subsample_mode", type=str, default="none", choices=["none", "random", "beam", "distance"])
    parser.add_argument("--subsample_ratio", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=32)
    parser.add_argument("--max_distance", type=float, default=35.0)
    
    # Limite campioni per esecuzione CPU veloce (es. 200 o 500)
    parser.add_argument("--max_samples", type=int, default=-1, help="Numero massimo di campioni da valutare (-1 per tutti)")
    
    parser.add_argument("--conf_thresh", type=float, default=0.3)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    dataset = KittiDataset(
        data_root=args.data_path,
        subsample_mode=args.subsample_mode,
        subsample_ratio=args.subsample_ratio,
        num_beams=args.num_beams,
        max_distance=args.max_distance
    )
    detector = LidarDetector(conf_threshold=args.conf_thresh)

    # Naming file di output
    if args.subsample_mode == 'random':
        tag = f"random_{int(args.subsample_ratio * 100)}perc"
    elif args.subsample_mode == 'beam':
        tag = f"beam_{args.num_beams}rings"
    elif args.subsample_mode == 'distance':
        tag = f"dist_{int(args.max_distance)}m"
    else:
        tag = "baseline_100perc"

    total_samples = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))
    print(f"\n📊 Avvio Valutazione [{tag.upper()}] | Campioni analizzati: {total_samples}/{len(dataset)}")

    gt_annotations = []
    pred_annotations = []

    # Loop di Inferenza con barra di avanzamento tqdm
    for i in tqdm(range(total_samples), desc="Valutazione Frame"):
        sample = dataset[i]
        
        # Ground Truth (Dizionari)
        gt_objs = sample['gt_boxes']
        gt_ann = {
            'name': np.array([obj['type'] for obj in gt_objs]),
            'truncated': np.array([obj['truncation'] for obj in gt_objs]) if gt_objs else np.zeros(0),
            'occluded': np.array([obj['occlusion'] for obj in gt_objs]) if gt_objs else np.zeros(0),
            'alpha': np.array([obj['alpha'] for obj in gt_objs]) if gt_objs else np.zeros(0),
            'bbox': np.array([obj['bbox_2d'] for obj in gt_objs]) if gt_objs else np.zeros((0, 4)),
            'dimensions': np.array([obj['dimensions_3d'] for obj in gt_objs]) if gt_objs else np.zeros((0, 3)),
            'location': np.array([obj['location_3d'] for obj in gt_objs]) if gt_objs else np.zeros((0, 3)),
            'rotation_y': np.array([obj['rotation_y'] for obj in gt_objs]) if gt_objs else np.zeros(0)
        }
        gt_annotations.append(gt_ann)
        
        # Predizioni (Oggetti Detection3D)
        detections = detector.detect(sample)
        pred_ann = {
            'name': np.array([det.obj_type for det in detections]),
            'truncated': np.zeros(len(detections)),
            'occluded': np.zeros(len(detections)),
            'alpha': np.zeros(len(detections)),
            'bbox': np.zeros((len(detections), 4)),
            'dimensions': np.array([det.dimensions_3d for det in detections]) if detections else np.zeros((0, 3)),
            'location': np.array([det.location_3d for det in detections]) if detections else np.zeros((0, 3)),
            'rotation_y': np.array([det.rotation_y for det in detections]) if detections else np.zeros(0),
            'score': np.array([det.score for det in detections]) if detections else np.zeros(0)
        }
        pred_annotations.append(pred_ann)

    # Calcolo Metrike KITTI
    classes = ['Car', 'Pedestrian', 'Cyclist']
    result_str, ret_dict = kitti_eval(gt_annotations, pred_annotations, classes)

    print("\n" + result_str)

    # Salvataggio su Drive
    txt_path = os.path.join(args.save_dir, f"report_{tag}.txt")
    json_path = os.path.join(args.save_dir, f"metrics_{tag}.json")
    
    with open(txt_path, "w") as f:
        f.write(result_str)
        
    with open(json_path, "w") as f:
        clean_ret_dict = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in ret_dict.items()}
        json.dump(clean_ret_dict, f, indent=4)

    print(f"✅ Risultati salvati su Drive:\n  • {txt_path}\n  • {json_path}")

if __name__ == "__main__":
    main()