# People Counting on a Mass Site — Paper Reproduction

Python reproduction of the crowd counting methods in **"Automated People Counting at a Mass Site"** (Hou & Pang, 2008, CENG 391 Assignment #5). Implements three feature extraction approaches paired with an MLP regressor to estimate crowd size from foreground segmentation images.

---

## Method Overview

The pipeline operates on pre-segmented foreground images (binary BMP, crowd pixels white). Three feature sets are extracted and fed to an MLP regressor trained on a 2:1 train/test split (first 2 of every 3 consecutive frames = train).

| Method | Feature(s) | Description |
|---|---|---|
| Method 1 | `X` = foreground pixel count | Raw pixel count (Eq. 1) |
| Method 2 (disk-4/5) | `C` = closed foreground count | Morphological closing with disk SE radius 4 or 5 |
| Method 3 | `C`, `S/X` ratio | Closing (disk-4) + opening ratio to separate isolated vs. merged blobs |

**MLP configuration**: single hidden layer (10 neurons), tanh activation, L-BFGS solver, StandardScaler, 10-run averaged predictions.

---

## Results (Table I Reproduction)

| Method | MAPE | Acc < 10% | Acc < 15% |
|---|---|---|---|
| Method 1 | 16.29% | 45.10% | 62.75% |
| Method 2 (disk-4) | 10.43% | 56.86% | 80.39% |
| Method 2 (disk-5) | 10.21% | 58.82% | 84.31% |
| **Method 3** | **10.64%** | **66.67%** | **86.27%** |

**Best method overall: Method 3** — highest threshold accuracy (86.27% within 15% of ground truth) despite a MAPE 0.43 pp higher than Method 2's best. Method 3's two-feature representation (area + structural ratio) gives the MLP more discriminative signal about crowd density than raw area alone.

---

## Implementation Notes

**Paper ambiguity — opening vs. erosion:**
The paper text (Section III) says "erosion operation" but Figure 11's caption says "opening." This implementation follows the figure (opening = erode then dilate). Opening is semantically more appropriate here: it removes small isolated noise pixels while preserving large foreground blobs, which is what the ratio `S/X` is meant to measure.

**Averaging over 10 runs:**
MLP with L-BFGS is sensitive to random weight initialization on small datasets. Results are averaged over 10 independent fits to reduce variance. This matches the paper's reported methodology.

**MAPE formula:**
`MAPE = mean(|pred - gt| / (gt + 1e-5)) * 100` — the epsilon prevents division by zero on frames with zero ground truth count.

---

## Usage

```bash
conda activate image  # or: pip install numpy pandas opencv-python scikit-learn matplotlib

# Expects dataset at:
#   ./data/foreground_images/ximagepixel*.bmp
#   ./data/groundtruth.txt

python people_counting.py
# Outputs: ./outputs/Figure_*.png, ./outputs/Table_I.png
```

---

## Output Files

- `Figure_05_Training.png` / `Figure_06_Test.png` — Method 1 training relationship and test predictions
- `Figure_07–08` — Method 2 (disk-4)
- `Figure_09–10` — Method 2 (disk-5)
- `Figure_11–12` — Method 3
- `Table_I.png` — Reproduced Table I

---

## Reference

Hou, Y., & Pang, Z. (2008). *People Counting and Human Detection in a Challenging Situation*. IEEE Transactions on Systems, Man, and Cybernetics.
