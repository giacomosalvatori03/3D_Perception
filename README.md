# 3D Perception & Multi-Modal Late Fusion Under Hardware Degradation

![Miniconda](https://img.shields.io/badge/Environment-Miniconda-green.svg)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch 2.0.1](https://img.shields.io/badge/PyTorch-2.0.1-orange.svg)
![torchvision 0.15.2](https://img.shields.io/badge/torchvision-0.15.2-red.svg)
![NumPy < 2](https://img.shields.io/badge/NumPy-%3C2.0-013243.svg)
![OpenMMLab](https://img.shields.io/badge/MMDetection3D-0.17.1-blueviolet.svg)
![Google Drive](https://img.shields.io/badge/Storage-Google%20Drive-4285F4.svg)
![Google Colab](https://img.shields.io/badge/Platform-Google%20Colab%20GPU-yellow.svg)

This repository contains the **Project #1, #2** of the Master's course *3D Perception, Learning-Based Data Fusion* at the **University of Parma (UNIPR)** within the **MUNER - Electric Vehicle Engineering (EEIV)** program in the **ADE - Autonomous Driving Engineering (ADE)** curriculum.

---

## 📌 Repository Overview

This repository contains the complete experimental suite and source code for a two-stage research study evaluating the operational robustness of 3D object detection under synthetic sensor degradation and designing mitigation strategies via camera-LiDAR sensor fusion:

1. **Project #1: Robustness Analysis of PointPillars Under LiDAR Sparsification**  
   *Quantifying mAP40 degradation of an off-the-shelf PointPillars model under uniform random point drops (75%, 50%, 25%) and elevation-based vertical beam reductions (64 → 32 → 16 rings) across standard difficulties and range-segmented spatial intervals (0–20m, 20–40m, 40–70m).*

2. **Project #2: Class-Adaptive Multi-Modal Late Fusion for Severe Sparsification**  
   *Developing a training-free 2D-3D Late Fusion framework combining PointPillars (3D LiDAR) and YOLOv8m (2D RGB Camera) via calibration-guided projection (P₂ matrix). Introduces a Class-Adaptive Calibration strategy to solve "False Positive Promotion" on large vehicle bounding boxes while maximizing recall recovery for point-sparse categories (Pedestrians, Cyclists).*


---

## 🧪 Project #1: PointPillars Robustness Analysis

To evaluate the operational boundaries of deep-learning-based 3D perception architectures without retraining, this project investigates how an off-the-shelf PointPillars detector degrades when subjected to synthetic LiDAR point cloud sparsification. The study simulates real-world hardware downgrades and environmental point loss on the KITTI benchmark across two controlled subsampling regimes: uniform random point drops ($\rho \in \{1.0, 0.75, 0.50, 0.25\}$) and vertical elevation beam reductions ($64 \to 32 \to 16$ rings). The analysis quantifies $mAP_{40}$ performance decay across standard KITTI difficulty tiers and range-segmented spatial intervals ($0\text{--}20\text{m}$, $20\text{--}40\text{m}$, $40\text{--}70\text{m}$).

---

## 🔀 Project #2: Class-Adaptive Multi-Modal Late Fusion

To compensate for 3D point loss without model retraining, candidate 3D bounding boxes from PointPillars (extracted at a lowered candidate threshold $\tau = 0.10$) are projected onto the 2D image plane using calibration matrix $P_2$ and rescored using 2D detections from YOLOv8m:

$$S_{\text{fused}} = \min\left(1.0, \, S_{\text{3D}} + \alpha_c \cdot S_{\text{2D}} \cdot \text{IoU}_{\text{2D}}\right)$$

### The Class-Specific Dilemma & Solution
Applying uniform fusion parameters causes **False Positive Promotion** on Cars: large 2D bounding footprints allow depth-displaced 3D ghost boxes to exceed IoU thresholds and get artificially boosted. 

To resolve this, we implement a **Class-Adaptive Configuration**:

| Object Class | $2\text{D IoU}_{\text{thresh}}$ | Boost Factor ($\alpha$) | Final Cutoff ($\tau_{\text{final}}$) | Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Car** | **$0.50$** | **$0.10$** | **$0.30$** | Restrictive matching suppresses depth-displaced false positives. |
| **Pedestrian** | **$0.25$** | **$0.35$** | **$0.20$** | Permissive recovery for point-sparse human structures. |
| **Cyclist** | **$0.25$** | **$0.35$** | **$0.20$** | Permissive recovery for thin object profiles. |

---

## 📊 Summary Results (Moderate $mAP_{40}$)

Comparison of **LiDAR-Only** vs. **Class-Adaptive Late Fusion** on the KITTI Validation Set across severe degradation regimes:

| Subsampling Configuration | Car (LiDAR) | Car (Fused) | Pedestrian (LiDAR) | Pedestrian (Fused) | Cyclist (LiDAR) | Cyclist (Fused) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (100% / 64 Beams)** | 62.71% | **62.75%** | 50.11% | **50.35%** | 43.70% | **44.02%** |
| **Random Drop 50%** | 54.77% | **56.10%** | 46.12% | **46.87%** | 28.15% | **28.70%** |
| **Random Drop 25%** | 43.81% | **44.48%** | 35.57% | **36.43%** | 20.25% | **20.48%** |
| **Beam Reduction (32 Rings)** | 56.61% | **56.85%** | 45.10% | **45.59%** | 29.62% | **30.10%** |
| **Beam Reduction (16 Rings)** | 36.80% | **37.75%** | 26.97% | **28.16%** | 13.45% | **14.33%** |

---

## 🚀 Quickstart & Usage

This project was executed on a **Google Colab GPU** instance for high-performance processing, leveraging an attached **Google Drive** for persistent storage and dataset management.

### Tech Stack & Dependencies:
- **Environment**: Miniconda / Google Colab (GPU Runtime recommended)
- **Python Version**: `3.10`
- **Core Package Specifications**:
  - `torch == 2.0.1`
  - `torchvision == 0.15.2`
  - `numpy < 2.0` (required for OpenMMLab / MMDetection3D compatibility)
  - `mmdetection3d` / OpenMMLab ecosystem
  - `google-colab` & `google-drive` (for storage mounting)

### Dataset Requirements & Setup
This project uses the **KITTI 3D Object Detection Validation Set** with the reduced point cloud configuration (`velodyne_reduced`).

The dataset is stored on **Google Drive** and mounted into the execution environment. You can automatically download, extract, and structure the dataset by running the provided setup notebook `notebooks/00_dataset_setup.ipynb`

### Environment Setup (Local / Miniconda / Colab)
If setting up a local environment using **Miniconda** or preparing a fresh Colab session follow the first cell of `notebooks/01_lidar_degradation.ipynb`


---

## 🔮 Next Steps: Project #3 (Learning-Based Data Fusion)

While Project #2 successfully eliminates *False Positive Promotion* using handcrafted class-adaptive rules, rule-based fusion relies on static geometric thresholds that require manual tuning and cannot adjust dynamically to varying environment conditions or depth levels.

**Project #3** will explore **Learning-Based Multi-Modal Data Fusion** architectures to transition from rule-based scoring to a data-driven paradigm.


---

## 👤 Author & Acknowledgments

- **Author:** Giacomo Salvatori
- **Degree Program:** MUNER - Master's in Electric Vehicle Engineering (EEIV)
- **Institution:** University of Parma (UNIPR), Italy
- **Frameworks:** Built with [OpenMMLab / MMDetection3D](https://github.com/open-mmlab/mmdetection3d) and tested on the [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/).