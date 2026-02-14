import cv2
import numpy as np

# ==============================================================================
# --- CONFIGURATION & TUNING PARAMETERS ---
# ==============================================================================

# --- File and Playback ---

# !!! IMPORTANT: Change this to the full path of your AVI file !!!
# VIDEO_PATH = "./40m_PanFire_20060824.avi"
# VIDEO_PATH = "./barbeq.avi"
# VIDEO_PATH = "./controlled1.avi"
# VIDEO_PATH = "./controlled2.avi"
# VIDEO_PATH = "./controlled3.avi"
# VIDEO_PATH = "./fBackYardFire.avi"
# VIDEO_PATH = "./forest1.avi"
# VIDEO_PATH = "./ForestFire1.avi"

# Playback speed (wait time in milliseconds).
# 1 = max speed, 30 = ~33fps playback, 0 = wait for keypress per frame
PLAYBACK_SPEED_MS = 30

# --- YCbCr Color Rules (Based on Çelik et al. papers) ---

# This is the threshold 'τ' (tau) for the rule |Cr - Cb| >= τ.
# The paper (source 2) found τ=40 to be a good value from their ROC curve.
# - Increase to be *less* sensitive (fewer false positives, more missed fire).
# - Decrease to be *more* sensitive (more detections, more false positives).
YCBCR_TAU_THRESHOLD = 35

# --- HSV Color Rules ---
# These rules define the "fire-like" colors to look for.
# Format is np.array([HUE, SATURATION, VALUE])

# First HSV rule (e.g., for reddish colors)
HSV_LOWER_BOUND_1 = np.array([0, 135, 0])
HSV_UPPER_BOUND_1 = np.array([10, 144, 255])

# Second HSV rule (e.g., for yellowish/whiter colors)
HSV_LOWER_BOUND_2 = np.array([0, 195, 0])
HSV_UPPER_BOUND_2 = np.array([10, 255, 255])

# --- Blob Detection & Filtering ---

# The smallest fire blob size (in pixels) to detect.
# This is the *most important* parameter for filtering out noise.
# - Increase to filter out smaller (likely false) detections.
# - Decrease to detect smaller (potentially real) fires.
MIN_CONTOUR_AREA = 200

# Kernel size for morphological operations (noise cleaning).
# A larger kernel (e.g., (7, 7)) will be more aggressive in
# removing noise and merging nearby blobs. (Width, Height).
MORPH_KERNEL_SIZE = (4, 4)

# Number of iterations for MORPH_CLOSE.
# Fills small black holes *inside* a detected blob.
# Increase to fill larger holes.
MORPH_CLOSE_ITERATIONS = 2

# Number of iterations for MORPH_OPEN.
# Removes small white specks (noise) *outside* the main blobs.
# Increase to remove larger noise specks.
MORPH_OPEN_ITERATIONS = 2

# --- Visualization (Bounding Box) ---

# Text to display above the bounding box
BBOX_TEXT = "Fire"
# Color of the bounding box and text (in BGR format)
BBOX_COLOR = (0, 0, 255)  # Red
# Thickness of the bounding box lines
BBOX_THICKNESS = 2
# Font for the text
BBOX_FONT = cv2.FONT_HERSHEY_SIMPLEX
# Font scale (size)
BBOX_FONT_SCALE = 0.7
# Thickness of the text
BBOX_TEXT_THICKNESS = 2

# ==============================================================================
# --- MAIN VIDEO PROCESSING (No need to edit below here) ---
# ==============================================================================

# 1. Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    print("Please check the file path in the VIDEO_PATH variable.")
    exit()

print(f"Video loaded: {VIDEO_PATH}")
print("Tuning Parameters:")
print(f"  - YCbCr Tau Threshold: {YCBCR_TAU_THRESHOLD}")
print(f"  - Min Blob Area: {MIN_CONTOUR_AREA}")
print(f"  - Morph Kernel: {MORPH_KERNEL_SIZE}")
print("Press 'q' to quit.")

# Create the morphological kernel from the config
morph_kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)

while True:
    # 2. Read one frame from the video
    ret, frame = cap.read()

    # If 'ret' is False, it means we're at the end of the video
    if not ret:
        print("End of video stream.")
        # Reset video to loop (optional)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # continue
        break

    # Create a copy of the frame to draw detections on
    detection_frame = frame.copy()

    # 3. Convert to HSV and YCrCb color spaces
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # 4. Apply HSV rules
    mask1_hsv = cv2.inRange(hsv, HSV_LOWER_BOUND_1, HSV_UPPER_BOUND_1)
    mask2_hsv = cv2.inRange(hsv, HSV_LOWER_BOUND_2, HSV_UPPER_BOUND_2)
    hsv_mask = cv2.bitwise_or(mask1_hsv, mask2_hsv)

    # 5. Apply YCbCr rules
    (Y, Cr, Cb) = cv2.split(ycrcb)

    # Calculate channel means for the *current frame*
    Y_mean = np.mean(Y)
    Cr_mean = np.mean(Cr)
    Cb_mean = np.mean(Cb)

    # Build masks for each rule
    rule1_mask = cv2.compare(Y, Cb, cv2.CMP_GT)           # Y > Cb
    rule2_mask = cv2.compare(Cr, Cb, cv2.CMP_GT)           # Cr > Cb
    rule3a_mask = cv2.compare(Y, Y_mean, cv2.CMP_GT)      # Y > Y_mean
    rule3b_mask = cv2.compare(Cb, Cb_mean, cv2.CMP_LT)    # Cb < Cb_mean
    rule3c_mask = cv2.compare(Cr, Cr_mean, cv2.CMP_GT)    # Cr > Cr_mean

    abs_diff = cv2.absdiff(Cr, Cb)
    rule4_mask = cv2.compare(abs_diff, YCBCR_TAU_THRESHOLD, cv2.CMP_GE) # |Cr - Cb| >= τ

    # Combine YCbCr rules as you did
    mask12 = cv2.bitwise_and(rule1_mask, rule2_mask)
    maskab = cv2.bitwise_and(rule3b_mask, rule3a_mask)
    mask12ab = cv2.bitwise_and(mask12, maskab)
    maskc4 = cv2.bitwise_and(rule4_mask, rule3c_mask)
    ycrcb_mask = cv2.bitwise_and(mask12ab, maskc4)

    # 6. Combine HSV and YCbCr masks
    final_mask = cv2.bitwise_or(ycrcb_mask, hsv_mask)

    # 7. *** BLOB DETECTION & CLEANING ***

    # Clean the mask using Morphological Operations
    # Closing (dilate -> erode) fills small holes inside blobs
    cleaned_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=MORPH_CLOSE_ITERATIONS)
    # Opening (erode -> dilate) removes small specks (noise)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, morph_kernel, iterations=MORPH_OPEN_ITERATIONS)

    # Find the contours (outlines of the blobs)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Filter out small contours that are likely noise
        if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
            # Get the bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(cnt)

            # Draw the red bounding box on the original frame
            cv2.rectangle(detection_frame, (x, y), (x + w, y + h), BBOX_COLOR, BBOX_THICKNESS)
            cv2.putText(detection_frame, BBOX_TEXT, (x, y - 10), BBOX_FONT, BBOX_FONT_SCALE, BBOX_COLOR, BBOX_TEXT_THICKNESS)

    # 8. Display the results
    # Show the original frame with detections
    cv2.imshow("Fire Detection", detection_frame)
    # Show the final pixel mask (before cleaning)
    # cv2.imshow("Fire Detection Mask (Raw)", final_mask)
    # Show the final pixel mask (after cleaning)
    cv2.imshow("Fire Detection Mask (Cleaned)", cleaned_mask)

    # 9. Handle playback and exit
    key = cv2.waitKey(PLAYBACK_SPEED_MS) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break

# 10. Clean up
cap.release()
cv2.destroyAllWindows()