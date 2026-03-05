# Fire Detection in Video

Real-time fire detection from video streams using a multi-colorspace pixel classification approach. Implements and fuses the YCbCr rule set from Celik et al. with HSV range masking, followed by morphological noise cleaning and contour-based localization.

No training data or neural network is used — the detector runs in real time on any CPU with OpenCV.

---

## Detection Pipeline

```
Video frame (BGR)
      |
      +--------> [1] Convert to HSV
      |                - Two range masks for fire-like hues (reddish + yellowish)
      |                - OR-combined -> hsv_mask
      |
      +--------> [2] Convert to YCbCr
                       - Per-frame channel means computed
                       - 5 per-pixel rules applied (Celik et al.):
                           R1: Y > Cb
                           R2: Cr > Cb
                           R3a: Y > mean(Y)
                           R3b: Cb < mean(Cb)
                           R3c: Cr > mean(Cr)
                           R4: |Cr - Cb| >= tau (default: 35)
                       - AND-combined -> ycrcb_mask
                           |
                           v
                    [3] Fuse: OR(hsv_mask, ycrcb_mask) -> final_mask
                           |
                           v
                    [4] Morphological Cleaning
                        - MORPH_CLOSE (2 iters, 4x4 kernel): fill holes
                        - MORPH_OPEN  (2 iters, 4x4 kernel): remove noise
                           |
                           v
                    [5] Contour Detection + Filtering
                        - findContours
                        - Area threshold: > 200 px
                        - Bounding box + "Fire" label drawn on detection frame
```

---

## Key Algorithm: YCbCr Fire Rules (Celik et al.)

Fire pixels have characteristic chrominance: they are bright (high Y), warm-toned (high Cr), and have low blue chrominance (low Cb). The per-frame mean comparison in R3a/R3b/R3c makes the detector **adaptive to scene illumination** — a brightly lit room won't trigger because the relative relationship between the pixel and the scene mean is what matters, not the absolute value.

The `|Cr - Cb|` threshold (`tau`) is the most discriminative single rule. The paper's ROC analysis found tau=40 optimal; tau=35 is used here for slightly higher sensitivity, appropriate for safety-critical applications where missed detections cost more than false alarms.

---

## Configuration

All tuning parameters are at the top of the file:

```python
YCBCR_TAU_THRESHOLD = 35      # |Cr - Cb| >= tau. Increase to reduce false positives.
HSV_LOWER_BOUND_1 = [0, 135, 0]   # Reddish fire
HSV_UPPER_BOUND_1 = [10, 144, 255]
HSV_LOWER_BOUND_2 = [0, 195, 0]   # Yellowish fire
HSV_UPPER_BOUND_2 = [10, 255, 255]
MIN_CONTOUR_AREA = 200        # Minimum blob area. Increase to filter small noise.
MORPH_KERNEL_SIZE = (4, 4)    # Morphological operation kernel size.
```

---

## Usage

```bash
pip install opencv-python numpy

# Set your video path at the top of the file:
# VIDEO_PATH = "./your_video.avi"

python fire_detection.py
# Press 'q' to quit
```

Supported video formats: any codec supported by your OpenCV build (AVI, MP4, etc.)

---

## Limitations and Honest Assessment

- **Color-only detection**: Sunsets, red/orange clothing, candles, and brake lights can all trigger false positives
- **No temporal consistency**: Each frame is classified independently — a single-frame flash can produce a detection
- **No depth/size estimation**: Cannot distinguish a large distant fire from a small nearby candle
- Adding temporal filtering (e.g., require N consecutive positive frames) would reduce false alarm rate at the cost of detection latency

---

## Skills Demonstrated

- Multi-colorspace image analysis (BGR, HSV, YCbCr)
- Adaptive thresholding using per-frame statistics
- Morphological noise filtering pipeline design
- Real-time video processing loop with OpenCV
- Parameter-driven configuration for tunable detection sensitivity

---

## Reference

Celik, T., & Demirel, H. (2009). *Fire detection in video sequences using a generic colour model*. Fire Safety Journal.
