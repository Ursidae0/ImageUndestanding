# Image Understanding

Coursework projects (CENG 391 / Image Understanding) implementing classical and learning-based computer vision techniques. All implementations are in Python (OpenCV, scikit-learn, scikit-fuzzy) without deep learning frameworks — the focus is on understanding the underlying signal processing and machine learning algorithms.

---

## Projects

| Directory | Topic | Technique |
|---|---|---|
| `retinal_blood_vessel_extraction/` | Medical image segmentation | FCM + Decision Tree + Gabor (paper reproduction) |
| `people_counting_on_a_mass_site/` | Crowd density estimation | Morphological features + MLP (paper reproduction) |
| `ball_tracking_program/` | Real-time object tracking | Background subtraction + contour detection |
| `fire_detection/` | Video fire detection | Multi-colorspace (YCbCr + HSV) rule fusion |

---

## Key Results

### Retinal Blood Vessel Extraction (DRIVE test set, 2nd manual annotations)

| Metric | This Implementation | Paper (Hashemzadeh 2019) | 2nd Human |
|---|---|---|---|
| Accuracy | 0.9255 | 0.9531 | 0.9464 |
| AUC | **0.8607** | 0.9752 | — |
| Sensitivity | 0.6760 | 0.7830 | 0.7796 |
| Specificity | 0.9603 | 0.9800 | 0.9717 |

The AUC gap (0.86 vs 0.975) is attributable to the paper's proprietary Root Guided Decision Tree, which is co-designed with its Gabor feature representation and has no direct sklearn equivalent.

### People Counting (Method 3 — best overall)

| Metric | Result |
|---|---|
| MAPE | 10.64% |
| Accuracy within 10% | 66.67% |
| Accuracy within 15% | **86.27%** |

---

## Shared Technical Stack

- Python 3.x, OpenCV, NumPy, scikit-learn, scikit-fuzzy, scikit-image, Matplotlib
- Conda environment: `image`

---

## Skills Demonstrated

- Medical image processing: CLAHE, Gabor filterbanks, morphological operators, Kittler-Illingworth optimal thresholding
- Unsupervised learning: Fuzzy C-Means (FCM) clustering
- Supervised classification: Decision Tree, MLP regression
- Multi-colorspace video analysis (RGB, YCbCr, HSV, CIELab)
- Quantitative evaluation: AUC, sensitivity/specificity, MAPE, pixel-level accuracy
- Paper reproduction methodology: understanding the gap between MATLAB and Python implementations
