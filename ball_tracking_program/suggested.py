import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_DIR = './data/'
OUTPUT_DIRS = {
    "raw_mask": "./assignment_output/02_background_removed", # Raw threshold
    "cleaned": "./assignment_output/03_cleaned",             # Morphological ops
    "detected": "./assignment_output/04_detected"            # Final Result
}

START_INDEX = 100
FRAMES_FOR_BG = 5
THRESHOLD_VAL = 25
BLUR_KERNEL_SIZE = (5, 5)
MORPH_KERNEL_SIZE = (5, 5)
MIN_CONTOUR_AREA = 500
WAIT_KEY_DELAY = 30
DISPLAY_SIZE = (320, 240)  # (width, height)


# --- HELPER FUNCTIONS ---
def initialize_output_dirs():
    """Create output directories if they don't exist."""
    for path in OUTPUT_DIRS.values():
        os.makedirs(path, exist_ok=True)


def load_frame(index):
    """Load a frame by index, return (filename, frame) or (None, None) if not found."""
    filename = f"ball00000{index}.jpg"
    filepath = os.path.join(INPUT_DIR, filename)
    
    if not os.path.exists(filepath):
        return None, None
    
    frame = cv2.imread(filepath)
    return filename, frame


def preprocess_frame(frame):
    """Convert to grayscale and apply Gaussian blur."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)


def build_background_model(bg_buffer):
    """Build background model by averaging frames."""
    return np.mean(bg_buffer, axis=0).astype(np.uint8)


def detect_foreground(gray, background_model):
    """Detect foreground using background subtraction."""
    diff = cv2.absdiff(gray, background_model)
    _, raw_mask = cv2.threshold(diff, THRESHOLD_VAL
                                , 255, cv2.THRESH_BINARY)
    
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    
    return raw_mask, cleaned_mask


def find_ball(cleaned_mask, frame):
    """Find and draw the ball on the frame, return center position."""
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_frame = frame.copy()
    detected_center = None

    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            
            cv2.circle(final_frame, center, radius, (0, 255, 0), 2)
            cv2.circle(final_frame, center, 2, (0, 0, 255), 3)
            detected_center = center

    return final_frame, detected_center


def create_combined_display(frame, raw_mask, cleaned_mask, final_frame):
    """Combine 4 views into a single display window."""
    width, height = DISPLAY_SIZE
    
    # Convert masks to BGR
    raw_mask_bgr = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
    cleaned_mask_bgr = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
    
    # Resize all frames
    frames = [
        ("Original", frame),
        ("Raw Mask", raw_mask_bgr),
        ("Cleaned Mask", cleaned_mask_bgr),
        ("Tracking", final_frame)
    ]
    
    resized = []
    for label, img in frames:
        img_resized = cv2.resize(img, (width, height))
        cv2.putText(img_resized, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        resized.append(img_resized)
    
    # Create 2x2 grid
    top_row = np.hstack([resized[0], resized[1]])
    bottom_row = np.hstack([resized[2], resized[3]])
    return np.vstack([top_row, bottom_row])


def save_outputs(filename, raw_mask, cleaned_mask, final_frame):
    """Save processing step outputs to files."""
    cv2.imwrite(f"{OUTPUT_DIRS['raw_mask']}/{filename}", raw_mask)
    cv2.imwrite(f"{OUTPUT_DIRS['cleaned']}/{filename}", cleaned_mask)
    cv2.imwrite(f"{OUTPUT_DIRS['detected']}/{filename}", final_frame)


def plot_trajectory(ball_trajectory):
    """Plot ball trajectory using matplotlib."""
    if len(ball_trajectory) == 0:
        print("No trajectory data collected. Check detection threshold.")
        return
    
    x_vals = [p[0] for p in ball_trajectory]
    y_vals = [p[1] for p in ball_trajectory]
    frame_nums = range(START_INDEX + FRAMES_FOR_BG + 2, 
                       START_INDEX + FRAMES_FOR_BG + 2 + len(ball_trajectory))

    plt.figure(figsize=(10, 6))
    plt.plot(frame_nums, x_vals, color='green', linestyle='--', label='X Position')
    plt.plot(frame_nums, y_vals, color='red', linestyle='--', label='Y Position')
    plt.title("Position over Time (Frame vs Coordinates)")
    plt.xlabel("Frame Number")
    plt.ylabel("Pixel Position")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("./assignment_output/05_plots.png")
    plt.show()
    print("Plots saved to ./assignment_output/05_plots.png")


# --- MAIN FUNCTION ---
def main():
    initialize_output_dirs()
    
    i = START_INDEX
    bg_buffer = []
    background_model = None
    ball_trajectory = []

    print("Step 1: Building Background Model from first 5 frames...")

    while True:
        filename, frame = load_frame(i)
        
        if filename is None or frame is None:
            print(f"Sequence ended or file not found at index {i}")
            break
        
        gray = preprocess_frame(frame)

        # --- PHASE 1: BACKGROUND INITIALIZATION ---
        if len(bg_buffer) < FRAMES_FOR_BG:
            bg_buffer.append(gray)
            print(f"  > Absorbed frame {i} into background model.")
            
            if len(bg_buffer) == FRAMES_FOR_BG:
                background_model = build_background_model(bg_buffer)
                print("Background Model Created! Starting detection...")
            
            i += 1
            continue

        # --- PHASE 2: DETECTION ---
        raw_mask, cleaned_mask = detect_foreground(gray, background_model)
        final_frame, detected_center = find_ball(cleaned_mask, frame)
        
        if detected_center:
            ball_trajectory.append(detected_center)
            print(f"Frame {i}: Ball detected at {detected_center}")

        # Display
        combined = create_combined_display(frame, raw_mask, cleaned_mask, final_frame)
        cv2.imshow("Ball Tracking - All Views", combined)
        
        # Save outputs
        save_outputs(filename, raw_mask, cleaned_mask, final_frame)

        if cv2.waitKey(WAIT_KEY_DELAY) & 0xFF == ord('q'):
            break
        i += 1

    cv2.destroyAllWindows()

    # Plot results
    print("Plotting results...")
    plot_trajectory(ball_trajectory)


if __name__ == "__main__":
    main()