# Robustness Analysis of PointPillars Under LiDAR Sparsification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenMMLab](https://img.shields.io/badge/MMDetection3D-0.17.1-blueviolet.svg)](https://github.com/open-mmlab/mmdetection3d)
[![Dataset](https://img.shields.io/badge/Dataset-KITTI--3D-yellow.svg)](http://www.cvlibs.net/datasets/kitti/eval_3dobject.php)

An empirical investigation into the operational robustness of the **PointPillars** 3D object detection architecture under synthetic LiDAR point cloud sparsification without model retraining.

This project forms **Project #1** of the Master's research course *3D Perception, Learning-Based Data Fusion* at the **University of Parma (UNIPR)** within the **MUNER - Electric Vehicle Engineering (EEIV)** program in the **ADE - Autonomous Driving Engineering (ADE)** curriculum.

---

## 📌 Project Overview & Motivation

High-resolution 64-beam mechanical LiDAR sensors (e.g., Velodyne HDL-64E) provide accurate 3D geometry for autonomous perception but introduce significant monetary cost, computational load, and mechanical failure risks. Furthermore, production autonomous vehicles frequently deploy lower-cost sensors (32 or 16 beams) or encounter atmospheric point absorption due to rain, fog, and spray.

This repository evaluates how gracefully an off-the-shelf, pre-trained **PointPillars** detector degrades under synthetic point cloud subsampling on the **KITTI 3D Object Detection Benchmark** under a **zero-retraining protocol**.

---

## ⚙️ Key Features

- **Two-Stage Sparsification Pipeline:**
  - **Uniform Random Point Drop:** Simulates atmospheric point absorption by retaining fixed point fractions ($\rho \in \{1.0, 0.75, 0.50, 0.25\}$).
  - **Vertical Beam Reduction:** Simulates hardware downgrades ($64 \to 32 \to 16$ rings) via vertical elevation angle ($\theta$) filtering.
- **Range-Aware Evaluation Framework:** Extends standard KITTI $mAP_{40}$ metrics with spatial depth breakdown along the camera Z-axis:
  - **Near Zone:** $0 - 20\text{m}$ (High density, critical collision avoidance)
  - **Medium Zone:** $20 - 40\text{m}$ (Standard urban driving range)
  - **Far Zone:** $40 - 70\text{m}$ (Sparse point returns, long-range perception)
- **Automated Logging & Reporting:** Exports structured metrics to `.json`, `.csv`, `.md`, and `.tex` formats automatically.
- **Qualitative Visual Inspection:** Side-by-side rendering of 3D projected bounding boxes on 2D RGB imagery and Bird's Eye View (BEV) point clouds matching the exact experiment sparsification level.

---

## 📁 Repository Structure

```
.
├── notebooks
|   ├── 00_test_dataset.ipynb       # Test the correct initialization of the dataset
|   └── 01_lidar_degradation.ipynb  # LiDAR sparsification evaluation pipeline
├── scripts/
|   ├── experiment_comparer.py      # Aggregates experiment JSONs & plots degradation curves 
│   ├── lidar_evaluator.py          # Unified KITTI 3D mAP40 & distance-range evaluator
|   ├── sample_visualizer.py        # High-level visual inspector (Camera projection + BEV)
|   └── sparsifier.py               # LidarSparsifier module (Random drop & Beam reduction)
├── src/
|   ├── calibration.py              # Calibration functions between LiDAR and Camera frame
│   ├── detection.py                # 3D Bounding Box detection data structure 
│   ├── kitti_dataset.py            # KITTI validation set loader and parser
│   ├── visualizer.py               # OpenCV & Matplotlib drawing utilities for 3D boxes & BEV
|   └── detectors/
|       ├── base_detector.py        # Base class for the 3D detectors
|       └── lidar_detector.py       # 3D object detection based on PointPillars
├── validate_lidar.py               # Inference on the validation dataset with sparsification  
├── paper_1.tex                     # IEEE 2-column format conference paper draft
└── README.md                       # Project documentation
```

---

## 📊 Summary of Key Results

| Subsampling Regime | Car Easy ($mAP_{40}$) | Car Mod ($mAP_{40}$) | Pedestrian Mod ($mAP_{40}$) | Cyclist Mod ($mAP_{40}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (100% pts, 64 rings)** | **67.40** | **62.71** | **49.63** | **43.70** |
| Random Drop ($\rho = 0.75$) | 65.34 | 59.62 | 49.38 | 40.88 |
| Random Drop ($\rho = 0.50$) | 65.17 | 54.77 | 44.68 | 35.78 |
| Random Drop ($\rho = 0.25$) | 55.26 | 43.81 | 40.10 | 20.25 |
| Beam Reduction ($32\text{ rings}$) | 68.04 | 56.61 | 45.04 | 29.62 |
| **Beam Reduction ($16\text{ rings}$)** | **49.00** | **36.80** | **26.97** | **13.45** |

### Core Insights
1. **High Resilience to Random Drop:** PointPillars retains strong feature representations down to 50% random point drop because local PointNet pillar encoders capture geometric shape effectively.
2. **Vulnerability to Hardware Downscaling:** Vertical beam reduction causes far more severe degradation than random point drop. 16-beam sampling drops Moderate Car $mAP_{40}$ to $36.80\%$ and Pedestrian $mAP_{40}$ to $26.97\%$.
3. **Spatial Decay:** Performance remains highly resilient in the Near range ($0-20\text{m}$) but experiences severe recall collapse in the Far range ($>40\text{m}$) where objects return fewer than 5 points total.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- OpenMMLab (MMDetection3D framework)
- NumPy, Pandas, Matplotlib, Shapely, OpenCV

### 1. Global & Range-Based Evaluation
To run 3D $mAP_{40}$ evaluation on an experiment folder containing predicted label files:
```bash
python scripts/lidar_evaluator.py --exp_dir /path/to/experiment_folder --eval_ranges
```

### 2. Plotting Degradation Curves
```python
from src.experiment_comparer import ExperimentComparer

comparer = ExperimentComparer(experiments_root="/path/to/experiments")

# Plot overall degradation across difficulties
comparer.plot_degradation_curves(
    exp_order=["baseline_100perc", "beam_32rings", "beam_16rings"],
    target_class="Car",
)

# Plot distance-range degradation (Near, Medium, Far)
comparer.plot_range_degradation_curves(
    exp_order=["baseline_100perc", "beam_32rings", "beam_16rings"],
    target_class="Car",
    difficulty="Moderate",
)
```

### 3. Visualizing Predictions & BEV Point Clouds
```python
from src.sample_visualizer import SampleVisualizer

visualizer = SampleVisualizer(data_path="/path/to/kitti_validation")
visualizer.visualize_experiment(
    exp_dir="/path/to/experiment_folder",
    num_samples=2,
    conf_thresh=0.3,
    grid_layout=True,
)
```

---

## 🔮 Next Steps: Project #2 (Camera-LiDAR Fusion)

The findings from this project demonstrate that single-modality 3D point cloud detectors suffer severe geometric limits when beam resolution drops. 

**Project #2** will investigate multi-modal **Camera-LiDAR Sensor Fusion** by incorporating 2D RGB imagery (already available in the KITTI benchmark) to determine if feature-level cross-attention can recover the recall loss caused by LiDAR sparsification.

---

## 👤 Author & Acknowledgments

- **Author:** Giacomo Salvatori
- **Degree Program:** MUNER - Master's in Electric Vehicle Engineering (EEIV)
- **Institution:** University of Parma (UNIPR), Italy
- **Frameworks:** Built with [OpenMMLab / MMDetection3D](https://github.com/open-mmlab/mmdetection3d) and tested on the [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/).