# 3D Perception & Multi-Modal Data Fusion Under Hardware Degradation

![Miniconda](https://img.shields.io/badge/Environment-Miniconda-green.svg)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch 2.0.1](https://img.shields.io/badge/PyTorch-2.0.1-orange.svg)
![torchvision 0.15.2](https://img.shields.io/badge/torchvision-0.15.2-red.svg)
![NumPy < 2](https://img.shields.io/badge/NumPy-%3C2.0-013243.svg)
![OpenMMLab](https://img.shields.io/badge/MMDetection3D-0.17.1-blueviolet.svg)
![Google Drive](https://img.shields.io/badge/Storage-Google%20Drive-4285F4.svg)
![Google Colab](https://img.shields.io/badge/Platform-Google%20Colab%20GPU-yellow.svg)

This repository contains **Project #1, #2, #3** (branch `3`) of the Master's course *3D Perception, Learning-Based Data Fusion* at the **University of Parma (UNIPR)** within the **MUNER - Electric Vehicle Engineering (EEIV)** program in the **ADE - Autonomous Driving Engineering (ADE)** curriculum.

---

## 📌 Repository Overview

This repository contains the complete experimental suite and source code for a three-stage research study evaluating the operational robustness of 3D object detection under synthetic sensor degradation and designing mitigation strategies via camera-LiDAR sensor fusion:

1. **Project #1: Robustness Analysis of PointPillars Under LiDAR Sparsification**  
   *Quantifying $mAP_{40}$ degradation of an off-the-shelf PointPillars model under uniform random point drops (75%, 50%, 25%) and elevation-based vertical beam reductions (64 → 32 → 16 rings) across standard difficulties and range-segmented spatial intervals (0–20m, 20–40m, 40–70m).*

2. **Project #2: Class-Adaptive Multi-Modal Late Fusion for Severe Sparsification**  
   *Developing a training-free 2D-3D Late Fusion framework combining PointPillars (3D LiDAR) and YOLOv8m (2D RGB Camera) via calibration-guided projection ($P_2$ matrix). Introduces a Class-Adaptive Calibration strategy to solve "False Positive Promotion" on large vehicle bounding boxes while maximizing recall recovery for point-sparse categories (Pedestrians, Cyclists).*

3. **Project #3: Learned 2D-3D Candidate Alignment (FastCLOCs) Under LiDAR Degradation**  
   *Transitioning from handcrafted static heuristics to a data-driven candidate-level fusion architecture (FastCLOCs). Utilizes a lightweight Multi-Layer Perceptron (MLP) regularized with LayerNorm to dynamically rescore 3D proposals by learning non-linear alignment patterns across 2D/3D confidences, projected IoU, normalized depth, box volume, and directional overlap ratios.*

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

## 🧠 Project #3: Learned 2D-3D Candidate Alignment (FastCLOCs)

While Project #2's handcrafted rules mitigate False Positive Promotion, static linear scoring functions cannot capture non-linear cross-modal dependencies or adapt dynamically to object depth and scale. **FastCLOCs** replaces manual threshold grids with a learned candidate-level rescoring network.

### Feature Engineering & MLP Architecture
3D candidates from PointPillars ($\tau_{\text{raw}} = 0.15$) are projected onto the 2D image plane via $P_2$ and matched with the highest-overlapping YOLOv8m detection to construct a **7-dimensional feature vector** $x \in \mathbb{R}^7$:

$$x = \left[ S_{3D}, \; S_{2D}, \; \text{IoU}_{2D}, \; \frac{\text{Depth}}{80.0}, \; \frac{\text{Volume}}{20.0}, \; x_{\text{overlap}}, \; y_{\text{overlap}} \right]^T$$

* **Network Architecture:** `Linear(7 → 64) → LayerNorm → ReLU → Linear(64 → 32) → LayerNorm → ReLU → Dropout(0.1) → Linear(32 → 1)` outputting raw logit $z_i$ ($S_{\text{learned}} = \sigma(z_i)$). Using `LayerNorm` ensures numerical stability during single-sample inference (`batch_size = 1`).
* **Training & Post-Processing:** Trained on 3,798 candidate pairs (400 KITTI frames) and validated on 983 pairs (100 frames) using Weighted Binary Cross-Entropy ($w_{\text{pos}} = 1.5$, AdamW, $\text{lr} = 10^{-3}$), achieving a validation **$F_1$-score of $87.41\%$** ($\text{Precision}: 82.80\%$, $\text{Recall}: 92.60\%$) at Epoch 12. Final detections are filtered via 3D BEV NMS ($\text{NMS}_{\text{thresh}} = 0.20$, $\tau_{\text{final}} = 0.25$).

---

## 📊 Summary Results ($mAP_{40}$ on KITTI Validation)

### 1. Cross-Paradigm Comparison (Moderate $mAP_{40}$): LiDAR-Only vs. Class-Adaptive vs. Learned Fusion
Comparison across **Single-Modality LiDAR-Only (Project #1)**, **Traditional Class-Adaptive Late Fusion (Project #2)**, and **FastCLOCs Learned Fusion (Project #3)** on the KITTI Validation Set:

| Subsampling Regime | Class | LiDAR-Only (E / M / H) | Adaptive Traditional (E / M / H) | Learned Fusion (E / M / H) |
| :--- | :--- | :---: | :---: | :---: |
| **`baseline_100perc`** | **Car** | 67.40% / 62.71% / 59.77% | 67.12% / 62.81% / 59.95% | **73.03% / 66.43% / 63.82%** |
| | **Pedestrian** | 53.82% / 49.63% / 44.13% | 62.09% / 56.29% / 50.14% | **67.90% / 63.97% / 63.55%** |
| | **Cyclist** | 48.11% / 43.70% / 42.59% | **55.04% / 47.97%** / 47.23% | 48.57% / 45.00% / **48.99%** |
| **`random_75perc`** | **Car** | 65.34% / 59.62% / 56.88% | 65.27% / 59.85% / 55.57% | **67.07% / 62.76% / 58.57%** |
| | **Pedestrian** | 55.25% / 49.38% / 44.93% | 61.63% / 55.70% / 50.22% | **66.47% / 63.61% / 62.07%** |
| | **Cyclist** | 50.24% / 40.88% / 40.34% | **55.18%** / 42.85% / 42.49% | 48.57% / **45.00% / 51.14%** |
| **`random_50perc`** | **Car** | 65.17% / 54.77% / 51.99% | 65.37% / 55.33% / 50.89% | **68.22% / 58.24% / 54.11%** |
| | **Pedestrian** | 54.77% / 49.56% / 44.68% | 58.10% / 52.10% / 46.92% | **70.56% / 66.54% / 64.25%** |
| | **Cyclist** | 46.91% / 35.78% / 35.57% | 48.17% / 33.53% / 35.40% | **50.00% / 38.21% / 46.67%** |
| **`random_25perc`** | **Car** | 55.26% / 43.81% / 39.42% | 57.10% / 44.42% / 38.47% | **58.06% / 46.42% / 40.62%** |
| | **Pedestrian** | 45.55% / 40.10% / 35.57% | 48.21% / 42.55% / 38.03% | **48.27% / 42.81% / 40.35%** |
| | **Cyclist** | 28.50% / 20.25% / 21.25% | **34.81% / 24.25%** / 25.10% | 20.00% / 20.00% / **32.50%** |
| **`beam_32rings`** | **Car** | 68.04% / 56.61% / 52.01% | 68.45% / 57.13% / 52.60% | **76.96% / 60.34% / 53.76%** |
| | **Pedestrian** | 50.47% / 45.04% / 40.38% | **54.38%** / 48.77% / 44.45% | 52.55% / **50.53% / 50.52%** |
| | **Cyclist** | 41.82% / 29.62% / 30.63% | **45.05%** / 33.26% / 34.06% | 33.33% / **38.00% / 47.50%** |
| **`beam_16rings`** | **Car** | 49.00% / 36.80% / 33.85% | 49.44% / 38.11% / 34.19% | **59.55% / 40.13% / 35.86%** |
| | **Pedestrian** | 31.75% / 26.97% / 24.00% | **38.55% / 33.34% / 30.48%** | 35.57% / 29.84% / 29.51% |
| | **Cyclist** | 18.90% / 13.45% / 14.93% | **22.40% / 17.01%** / 18.57% | 10.00% / 13.33% / **26.00%** |

### 2. Full Multi-Difficulty Breakdown Gain Analysis: LiDAR-Only vs. Class-Adaptive vs. Learned Fusion
Complete 3D $mAP_{40}$ across all KITTI difficulty tiers (*Easy / Moderate / Hard*), highlighting the relative (up to **+21.53%** under 16-beam reduction):

#### `Car` — Moderate Difficulty (Benchmark Principale KITTI)

| Subsampling Regime | LiDAR-Only ($\text{mAP}_{40}$) | Adaptive Traditional ($\text{mAP}_{40}$) | Learned Fusion ($\text{mAP}_{40}$) | $\Delta$ Learned vs LiDAR (Ass. / Rel.) | $\Delta$ Learned vs Trad. (Ass. / Rel.) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`baseline_100perc`** | 62.71% | 62.81% | **66.43%** | +3.72% / +5.93% | +3.61% / +5.75% |
| **`random_75perc`** | 59.62% | 59.85% | **62.76%** | +3.14% / +5.27% | +2.91% / +4.87% |
| **`random_50perc`** | 54.77% | 55.33% | **58.24%** | +3.47% / +6.34% | +2.91% / +5.26% |
| **`random_25perc`** | 43.81% | 44.42% | **46.42%** | +2.61% / +5.96% | +2.00% / +4.51% |
| **`beam_32rings`** | 56.61% | 57.13% | **60.34%** | +3.73% / +6.59% | +3.21% / +5.62% |
| **`beam_16rings`** | 36.80% | 38.11% | **40.13%** | +3.33% / +9.05% | +2.02% / +5.29% |

#### `Car` — Easy Difficulty

| Subsampling Regime | LiDAR-Only ($\text{mAP}_{40}$) | Adaptive Traditional ($\text{mAP}_{40}$) | Learned Fusion ($\text{mAP}_{40}$) | $\Delta$ Learned vs LiDAR (Ass. / Rel.) | $\Delta$ Learned vs Trad. (Ass. / Rel.) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`baseline_100perc`** | 67.40% | 67.12% | **73.03%** | +5.63% / +8.35% | +5.91% / +8.80% |
| **`random_75perc`** | 65.34% | 65.27% | **67.07%** | +1.73% / +2.65% | +1.79% / +2.74% |
| **`random_50perc`** | 65.17% | 65.37% | **68.22%** | +3.05% / +4.68% | +2.85% / +4.36% |
| **`random_25perc`** | 55.26% | 57.10% | **58.06%** | +2.80% / +5.07% | +0.96% / +1.68% |
| **`beam_32rings`** | 68.04% | 68.45% | **76.96%** | +8.92% / +13.11% | +8.51% / +12.43% |
| **`beam_16rings`** | 49.00% | 49.44% | **59.55%** | +10.55% / +21.53% | +10.10% / +20.43% |

#### `Car` — Hard Difficulty

| Subsampling Regime | LiDAR-Only ($\text{mAP}_{40}$) | Adaptive Traditional ($\text{mAP}_{40}$) | Learned Fusion ($\text{mAP}_{40}$) | $\Delta$ Learned vs LiDAR (Ass. / Rel.) | $\Delta$ Learned vs Trad. (Ass. / Rel.) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`baseline_100perc`** | 59.77% | 59.95% | **63.82%** | +4.05% / +6.78% | +3.87% / +6.46% |
| **`random_75perc`** | 56.88% | 55.57% | **58.57%** | +1.69% / +2.97% | +3.01% / +5.41% |
| **`random_50perc`** | 51.99% | 50.89% | **54.11%** | +2.12% / +4.08% | +3.22% / +6.32% |
| **`random_25perc`** | 39.42% | 38.47% | **40.62%** | +1.20% / +3.04% | +2.16% / +5.60% |
| **`beam_32rings`** | 52.01% | 52.60% | **53.76%** | +1.75% / +3.36% | +1.16% / +2.21% |
| **`beam_16rings`** | 33.85% | 34.19% | **35.86%** | +2.01% / +5.94% | +1.67% / +4.88% |

---

#### `Pedestrian` — Moderate Difficulty

| Subsampling Regime | LiDAR-Only ($\text{mAP}_{40}$) | Adaptive Traditional ($\text{mAP}_{40}$) | Learned Fusion ($\text{mAP}_{40}$) | $\Delta$ Learned vs LiDAR (Ass. / Rel.) | $\Delta$ Learned vs Trad. (Ass. / Rel.) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`baseline_100perc`** | 49.63% | 56.29% | **63.97%** | +14.34% / +28.89% | +7.67% / +13.63% |
| **`random_75perc`** | 49.38% | 55.70% | **63.61%** | +14.23% / +28.82% | +7.91% / +14.20% |
| **`random_50perc`** | 49.56% | 52.10% | **66.54%** | +16.98% / +34.26% | +14.43% / +27.70% |
| **`random_25perc`** | 40.10% | 42.55% | **42.81%** | +2.71% / +6.76% | +0.25% / +0.60% |
| **`beam_32rings`** | 45.04% | 48.77% | **50.53%** | +5.49% / +12.19% | +1.76% / +3.60% |
| **`beam_16rings`** | 26.97% | **33.34%** | 29.84% | +2.87% / +10.64% | -3.50% / -10.50% |

#### `Pedestrian` — Easy Difficulty

| Subsampling Regime | LiDAR-Only ($\text{mAP}_{40}$) | Adaptive Traditional ($\text{mAP}_{40}$) | Learned Fusion ($\text{mAP}_{40}$) | $\Delta$ Learned vs LiDAR (Ass. / Rel.) | $\Delta$ Learned vs Trad. (Ass. / Rel.) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`baseline_100perc`** | 53.82% | 62.09% | **67.90%** | +14.08% / +26.16% | +5.81% / +9.36% |
| **`random_75perc`** | 55.25% | 61.63% | **66.47%** | +11.22% / +20.31% | +4.85% / +7.87% |
| **`random_50perc`** | 54.77% | 58.10% | **70.56%** | +15.79% / +28.83% | +12.46% / +21.44% |
| **`random_25perc`** | 45.55% | 48.21% | **48.27%** | +2.72% / +5.97% | +0.06% / +0.12% |
| **`beam_32rings`** | 50.47% | **54.38%** | 52.55% | +2.08% / +4.12% | -1.83% / -3.36% |
| **`beam_16rings`** | 31.75% | **38.55%** | 35.57% | +3.82% / +12.03% | -2.98% / -7.74% |

#### `Pedestrian` — Hard Difficulty

| Subsampling Regime | LiDAR-Only ($\text{mAP}_{40}$) | Adaptive Traditional ($\text{mAP}_{40}$) | Learned Fusion ($\text{mAP}_{40}$) | $\Delta$ Learned vs LiDAR (Ass. / Rel.) | $\Delta$ Learned vs Trad. (Ass. / Rel.) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`baseline_100perc`** | 44.13% | 50.14% | **63.55%** | +19.42% / +44.01% | +13.41% / +26.75% |
| **`random_75perc`** | 44.93% | 50.22% | **62.07%** | +17.14% / +38.15% | +11.85% / +23.59% |
| **`random_50perc`** | 44.68% | 46.92% | **64.25%** | +19.57% / +43.80% | +17.33% / +36.93% |
| **`random_25perc`** | 35.57% | 38.03% | **40.35%** | +4.78% / +13.44% | +2.32% / +6.09% |
| **`beam_32rings`** | 40.38% | 44.45% | **50.52%** | +10.14% / +25.11% | +6.07% / +13.66% |
| **`beam_16rings`** | 24.00% | **30.48%** | 29.51% | +5.51% / +22.96% | -0.97% / -3.19% |

---

#### `Cyclist` — Moderate Difficulty

| Subsampling Regime | LiDAR-Only ($\text{mAP}_{40}$) | Adaptive Traditional ($\text{mAP}_{40}$) | Learned Fusion ($\text{mAP}_{40}$) | $\Delta$ Learned vs LiDAR (Ass. / Rel.) | $\Delta$ Learned vs Trad. (Ass. / Rel.) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`baseline_100perc`** | 43.70% | **47.97%** | 45.00% | +1.30% / +2.97% | -2.97% / -6.18% |
| **`random_75perc`** | 40.88% | 42.85% | **45.00%** | +4.12% / +10.08% | +2.15% / +5.02% |
| **`random_50perc`** | 35.78% | 33.53% | **38.21%** | +2.43% / +6.79% | +4.69% / +13.98% |
| **`random_25perc`** | 20.25% | **24.25%** | 20.00% | -0.25% / -1.23% | -4.25% / -17.51% |
| **`beam_32rings`** | 29.62% | 33.26% | **38.00%** | +8.38% / +28.29% | +4.74% / +14.26% |
| **`beam_16rings`** | 13.45% | **17.01%** | 13.33% | -0.12% / -0.89% | -3.68% / -21.61% |

#### `Cyclist` — Easy Difficulty

| Subsampling Regime | LiDAR-Only ($\text{mAP}_{40}$) | Adaptive Traditional ($\text{mAP}_{40}$) | Learned Fusion ($\text{mAP}_{40}$) | $\Delta$ Learned vs LiDAR (Ass. / Rel.) | $\Delta$ Learned vs Trad. (Ass. / Rel.) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`baseline_100perc`** | 48.11% | **55.04%** | 48.57% | +0.46% / +0.96% | -6.47% / -11.75% |
| **`random_75perc`** | 50.24% | **55.18%** | 48.57% | -1.67% / -3.32% | -6.61% / -11.97% |
| **`random_50perc`** | 46.91% | 48.17% | **50.00%** | +3.09% / +6.59% | +1.83% / +3.80% |
| **`random_25perc`** | 28.50% | **34.81%** | 20.00% | -8.50% / -29.82% | -14.81% / -42.54% |
| **`beam_32rings`** | 41.82% | **45.05%** | 33.33% | -8.49% / -20.30% | -11.71% / -26.00% |
| **`beam_16rings`** | 18.90% | **22.40%** | 10.00% | -8.90% / -47.09% | -12.40% / -55.35% |

#### `Cyclist` — Hard Difficulty

| Subsampling Regime | LiDAR-Only ($\text{mAP}_{40}$) | Adaptive Traditional ($\text{mAP}_{40}$) | Learned Fusion ($\text{mAP}_{40}$) | $\Delta$ Learned vs LiDAR (Ass. / Rel.) | $\Delta$ Learned vs Trad. (Ass. / Rel.) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`baseline_100perc`** | 42.59% | 47.23% | **48.99%** | +6.40% / +15.03% | +1.76% / +3.73% |
| **`random_75perc`** | 40.34% | 42.49% | **51.14%** | +10.80% / +26.77% | +8.64% / +20.34% |
| **`random_50perc`** | 35.57% | 35.40% | **46.67%** | +11.10% / +31.21% | +11.26% / +31.82% |
| **`random_25perc`** | 21.25% | 25.10% | **32.50%** | +11.25% / +52.94% | +7.40% / +29.50% |
| **`beam_32rings`** | 30.63% | 34.06% | **47.50%** | +16.87% / +55.08% | +13.44% / +39.48% |
| **`beam_16rings`** | 14.93% | 18.57% | **26.00%** | +11.07% / +74.15% | +7.43% / +40.01% |

### 3. Summary of Experimental Findings

- **`Car` (Across-the-Board Robustness):** Learned Fusion (Fast-CLOCs) consistently outperforms both the LiDAR-Only baseline and Adaptive Traditional Fusion across **18 out of 18 combinations** (all difficulty levels and sparsity regimes), achieving gains of up to **+10.55 pts (+21.53%)** over LiDAR-Only and **+10.10 pts (+20.43%)** over Adaptive Traditional Fusion under `beam_16rings` (Easy).
- **`Pedestrian` (Peak Gains Under Occlusion and Medium-to-High Density):** Learned Fusion substantially improves detection performance on **Moderate** and **Hard** difficulties down to `beam_32rings` and `random_25perc` (peaking at **+16.98 pts / +34.26%** on Moderate and **+19.57 pts / +43.80%** on Hard under the `random_50perc` regime vs. LiDAR-Only), whereas Adaptive Traditional Fusion retains an edge in the extreme `beam_16rings` configuration (+3.50 pts on Moderate vs. Learned Fusion).
- **`Cyclist` (Cross-Regime Complementarity):** On **Hard** difficulty, Learned Fusion dominates across every regime (+6.40 pts up to +16.87 pts vs. LiDAR-Only; +1.76 pts up to +13.44 pts vs. Adaptive Traditional Fusion), whereas on **Easy** and ultra-sparse regimes (`random_25perc`, `beam_16rings`), the adaptive heuristic approach prevents confidence degradation in 3D candidates.

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
  - `ultralytics` (YOLOv8m 2D candidate extraction)
  - `google-colab` & `google-drive` (for storage mounting)

### Dataset Requirements & Setup
This project uses the **KITTI 3D Object Detection Validation Set** with the reduced point cloud configuration (`velodyne_reduced`).

The dataset is stored on **Google Drive** and mounted into the execution environment. You can automatically download, extract, and structure the dataset by running the provided setup notebook `notebooks/00_dataset_setup.ipynb`.

### Environment Setup & Execution (Local / Miniconda / Colab)
If setting up a local environment using **Miniconda** or preparing a fresh Colab session, follow the initialization steps in `notebooks/01_lidar_degradation.ipynb` and execute the FastCLOCs candidate extraction, MLP training, and 6-regime evaluation notebooks available on branch `3`.

---

## 🔮 Next Steps: Future Work

While **Project #3 (FastCLOCs)** eliminates manual rule tuning and suppresses LiDAR-only False Positives across all degradation regimes, candidate-level (late) fusion remains bounded by initial 3D proposal generation: when extreme point loss prevents the 3D backbone from generating a candidate box, camera confidence alone cannot recover the object.

Future research directions will explore **Intermediate Feature-Level Fusion**, projecting 2D camera feature maps directly into the 3D voxel/pillar backbone prior to candidate generation, alongside evaluation on benchmarks featuring dense far-range annotations (e.g., *nuScenes*, *Waymo*).

---

## 👤 Author & Acknowledgments

- **Author:** Giacomo Salvatori
- **Degree Program:** MUNER - Master's in Electric Vehicle Engineering (EEIV)
- **Institution:** University of Parma (UNIPR), Italy
- **Frameworks:** Built with [OpenMMLab / MMDetection3D](https://github.com/open-mmlab/mmdetection3d) and tested on the [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/).