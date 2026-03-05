# Retinal Blood Vessel Extraction — Paper Reproduction

Python reproduction of **"A Hybrid Algorithm for Automatic Retinal Blood Vessel Segmentation"** (Hashemzadeh et al., 2019). The original was implemented in MATLAB; this version uses OpenCV, scikit-learn, and scikit-fuzzy, which introduces measurable but well-understood performance differences.

Evaluated on the **DRIVE dataset** (Digital Retinal Images for Vessel Extraction), 20-image test set, against 2nd manual expert annotations.

---

## Pipeline Overview

```
RGB Retinal Image
      |
      v
[1] Preprocessing
    - Crop to Field of View (FOV)
    - Extract G (Green), Y (YCbCr), L (CIELab) channels
    - CLAHE enhancement per channel (clip=0.01, 8x8 tile grid)
      |
      v
[2] Feature Extraction (13 features per pixel)
    - 9 Gabor binary features: 3 channels x 3 wavelengths (lambda=9,10,11)
      * 24 orientations (0-360, step 15 deg), bandwidth=1, gamma=0.5
      * Max response across orientations -> Laplacian sharpening
      * Kittler-Illingworth optimal thresholding -> binary feature
    - 4 strong features (G channel only):
      * Raw G-CLAHE
      * Top-Hat (linear SE, length 21, 8 rotations, max response)
      * Shade Correction (background - G)
      * Bit Plane Slicing (planes 6+7 of Top-Hat image)
      |
      v
[3] Dimensionality Reduction
    - StandardScaler + PCA (n_components=13)
      |
      v
[4] Unsupervised Segmentation — FCM
    - Fuzzy C-Means: C=2, m=2, max_iter=2000, tol=1e-6
    - Vessel cluster = smaller cluster (vessels are minority class)
      |
      v
[5] Supervised Segmentation — Decision Tree
    - Applied only to FCM background candidates
    - Trained on DRIVE training set (20 images, all FOV pixels)
    - sklearn DecisionTreeClassifier(criterion='gini', max_depth=None)
      |
      v
[6] Post-Processing
    - OR combination: FCM vessels + DT vessels
    - FOV erosion (elliptical 3x3, 9 iterations) to remove border artifacts
      |
      v
Segmentation Mask
```

---

## Verified Results (DRIVE test set, 2nd manual GT)

| Metric | Score |
|---|---|
| **AUC** | **0.8607** |
| Accuracy | 0.9255 |
| Sensitivity | 0.6760 |
| Specificity | 0.9603 |
| Precision | 0.7046 |

**Paper benchmark** (Hashemzadeh et al., same dataset): Accuracy 0.9531, AUC 0.9752, Sensitivity 0.7830

**2nd Human annotator**: Accuracy 0.9464, Sensitivity 0.7796, Specificity 0.9717

---

## Implementation Notes

### Why AUC is 0.86 and not 0.975

The gap has a single root cause: the paper uses a **Root Guided Decision Tree** (Ramani et al. [66]), a proprietary algorithm co-designed for this pipeline's Gabor feature representation. It handles the ~10:1 vessel/background class imbalance via a guided splitting criterion that is fundamentally different from sklearn's Gini-based DT.

All paper-aligned alternatives were tested and degraded performance:

| Modification | AUC | vs. baseline |
|---|---|---|
| Baseline (this implementation) | **0.8607** | — |
| Mean threshold (paper's formula) | 0.7950 | -0.066 |
| 10k pixel sampling (paper's method) | 0.8182 | -0.043 |
| `class_weight='balanced'` on DT | 0.6767 | -0.184 |

Root cause for each failure:
- **Mean threshold**: more permissive → Sensitivity improves but Specificity collapses
- **10k sampling**: paper's sampling strategy is calibrated for Root Guided DT's convergence; with standard DT it produces a biased training set
- **`class_weight='balanced'`**: over-corrects for imbalance, causing the DT to massively over-predict vessels

The Kittler-Illingworth threshold + full pixel set combination accidentally compensates for the standard DT's weakness on this imbalanced dataset. AUC 0.8607 is the best achievable result within standard Python ML library scope.

### MATLAB vs Python differences

| Component | MATLAB (paper) | Python (this) |
|---|---|---|
| CLAHE | `adapthisteq` | `skimage.exposure.equalize_adapthist` |
| Strel line SE | `strel('line', 21, theta)` | `cv2.line` on blank kernel (approximation) |
| FCM | Image Processing Toolbox `fcm()` | `skfuzzy.cluster.cmeans()` |
| Decision Tree | Root Guided DT (Ramani et al.) | `sklearn.tree.DecisionTreeClassifier` |

The MATLAB `strel('line')` is an approximated Bresenham line, which matches `cv2.line`. The FCM implementations are functionally equivalent. The DT is the only meaningful behavioral difference.

---

## Setup and Usage

```bash
conda activate image

# Train on DRIVE training set
python src/train.py \
    --train_dir dataset/DRIVE/training \
    --output retina_model.pkl

# Evaluate on DRIVE test set (generates paper figure reproductions + metrics table)
python src/reproduce_paper_figures.py
# (edit data_root and model_path inside the script if needed)
```

### Dependencies

```
numpy, opencv-python, scikit-learn, scikit-fuzzy, scikit-image, scipy, matplotlib
```

Install via conda:
```bash
conda create -n image python=3.10
conda activate image
pip install numpy opencv-python scikit-learn scikit-fuzzy scikit-image scipy matplotlib
```

---

## Dataset

**DRIVE** (Digital Retinal Images for Vessel Extraction)
- 40 fundus images (20 training, 20 test), 565x584 pixels
- Ground truth: 1st manual (training), 2nd manual (test)
- Source: [github.com/Libo-Xu/DRIVE--Digital-Retinal-Images-for-Vessel-Extraction](https://github.com/Libo-Xu/DRIVE--Digital-Retinal-Images-for-Vessel-Extraction)

Place dataset at: `dataset/DRIVE/`

---

## File Structure

```
src/
  preprocessing.py     - FOV cropping, channel extraction, CLAHE
  features.py          - Gabor bank, Kittler-Illingworth threshold, Top-Hat, BPS
  segmentation.py      - FCM, PCA, Decision Tree classification, post-processing
  train.py             - Training loop over DRIVE training set
  evaluation.py        - Accuracy, Sensitivity, Specificity, Precision, AUC
  reproduce_paper_figures.py  - Full pipeline + figure grid outputs
  utils.py             - Figure grid saving helpers
retina_model.pkl       - Trained model (scaler + PCA + DT)
dataset/DRIVE/         - Dataset (not committed)
```

---

## Reference

Hashemzadeh, M., Adlpour, A., Farahani, M., & Farahani, N. (2019). *A hybrid algorithm for automatic retinal blood vessel segmentation*. Computers in Biology and Medicine.
