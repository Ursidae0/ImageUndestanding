# Ball Tracking Program

Real-time ball detection and trajectory tracking from a sequence of frames using background subtraction and contour-based detection. Implements a classical computer vision pipeline without machine learning — the goal is understanding the fundamental building blocks of motion tracking.

---

## Pipeline

```
Frame sequence (ball00000NNN.jpg)
      |
      v
[1] Background Model
    - First 5 frames absorbed into a rolling buffer
    - Per-pixel mean -> static background image (grayscale)
      |
      v
[2] Foreground Detection
    - Gaussian blur (5x5) to suppress high-frequency noise
    - Absolute difference: |current_gray - background|
    - Binary threshold (T=25) -> raw foreground mask
    - Morphological opening (5x5 kernel) -> cleaned mask
      |
      v
[3] Ball Localization
    - findContours on cleaned mask
    - Filter by area (> 500 px) to reject small noise
    - minEnclosingCircle on largest valid contour
    - Draw circle + center dot on detection frame
      |
      v
[4] Trajectory Recording + Plotting
    - (x, y) center appended per frame
    - Matplotlib: X and Y position vs. frame number
```

---

## Design Decisions

**Static background model (mean of 5 frames):** Simple and effective when the camera is static and the scene is initially ball-free. A more robust approach (MOG2, KNN) would handle dynamic backgrounds but is unnecessary here — the scene is a controlled overhead shot.

**Morphological opening over closing:** Opening (erode → dilate) removes small noise specks while preserving the circular ball blob. Closing would fill holes inside the blob — not needed since the ball has consistent reflectance.

**Minimum contour area filter (500 px):** Balances sensitivity vs. false positives. Too low → every dust speck triggers; too high → small or distant ball is missed. 500 px corresponds to a ~13-pixel diameter circle.

**Y-axis inversion in trajectory plot:** OpenCV uses image coordinates (Y increases downward). Matplotlib's `invert_yaxis()` ensures the trajectory plot matches the physical scene geometry.

---

## Configuration (top of file)

```python
THRESHOLD_VAL = 25        # Background subtraction sensitivity
FRAMES_FOR_BG = 5         # Frames used to build background model
MIN_CONTOUR_AREA = 500    # Minimum blob area to count as ball
BLUR_KERNEL_SIZE = (5, 5) # Gaussian blur kernel
MORPH_KERNEL_SIZE = (5, 5)# Morphological operation kernel
```

---

## Usage

```bash
pip install opencv-python numpy matplotlib

# Place frames at: ./data/ball00000NNN.jpg (starting from index 100)
python suggested.py
# Press 'q' to quit real-time display
# Trajectory plot saved to: ./assignment_output/05_plots.png
```

---

## Output Structure

```
assignment_output/
  02_background_removed/   - Raw binary difference masks
  03_cleaned/              - After morphological opening
  04_detected/             - Original frame with detection overlay
  05_plots.png             - X and Y trajectory over time
```

---

## Skills Demonstrated

- Background subtraction and temporal frame differencing
- Morphological image processing (open, close, erode, dilate)
- Contour analysis and minimum enclosing circle
- Real-time OpenCV display pipeline (imshow + waitKey)
- Trajectory visualization with matplotlib
