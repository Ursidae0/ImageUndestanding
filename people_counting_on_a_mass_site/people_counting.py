"""
Assignment #5: Automated People Counting at a Mass Site
Course: CENG 391
Student Name: Eren IŞIK

This script reproduces the people counting methods described in Hou & Pang (2008).
It implements foreground extraction, morphological operations, and neural network
regression (MLP) to estimate crowd density based on the paper's specific methods.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Configuration
BASE_DIR = './data'
IMG_SUBDIR = 'foreground_images'
GT_FILENAME = 'groundtruth.txt'
OUTPUT_DIR = './outputs'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_and_split_dataset(base_dir):
    """
    Loads data and splits it into training/testing sets.
    Rule: First 2 images of every 3 consecutive images are for training (Paper Section IV).
    """
    # 1. Load Ground Truth
    gt_path = os.path.join(base_dir, GT_FILENAME)
    if not os.path.exists(gt_path):
        gt_path = os.path.join(base_dir, 'groundtruth')
        
    ground_truth = []
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    ground_truth.append(int(line))
    else:
        print(f"Error: GT file not found in {base_dir}")
        return None

    # 2. Load Images
    img_pattern = os.path.join(base_dir, IMG_SUBDIR, 'ximagepixel*.bmp')
    image_files = sorted(glob.glob(img_pattern))

    # 3. Sync Data
    n_samples = min(len(image_files), len(ground_truth))
    image_files = image_files[:n_samples]
    ground_truth = ground_truth[:n_samples]

    # 4. Split (Train:Test = 2:1 pattern)
    train_X, train_y = [], []
    test_X, test_y = [], []

    for i in range(n_samples):
        # Index 0, 1 -> Train; Index 2 -> Test
        if (i % 3) == 2:
            test_X.append(image_files[i])
            test_y.append(ground_truth[i])
        else:
            train_X.append(image_files[i])
            train_y.append(ground_truth[i])

    print(f"Data Loaded: {len(train_X)} Training, {len(test_X)} Testing images.")
    return (np.array(train_X), np.array(train_y), 
            np.array(test_X), np.array(test_y))

def extract_features(image_paths, method, disk_radius=4):
    """
    Extracts features based on the methods defined in Section III.
    """
    features = []
    
    # Structuring elements
    k_size = (disk_radius * 2) + 1
    kernel_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    # Paper Section IV (Method 3): "A square of 2*2 pixels was used for erosion"
    kernel_square = np.ones((2, 2), np.uint8) 

    for path in image_paths:
        img = cv2.imread(path, 0)
        if img is None: continue

        if method == 'method1':
            # Method 1: Just count foreground pixels (Eq. 1)
            val = cv2.countNonZero(img)
            features.append([val])

        elif method == 'method2':
            # Method 2: Closing -> Count (Eq. 2)
            closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_disk)
            val = cv2.countNonZero(closed_img)
            features.append([val])

        elif method == 'method3':
            # ---------------------------------------------------------
            # Note on Method 3 Implementation:
            # I noticed a conflict in the paper: The text (Section III) says 
            # "erosion operation", but the caption of Fig 11 says "opening".
            # I chose to follow the figure description (opening) for this implementation.
            # ---------------------------------------------------------
            
            # Feature 1: C (Closed with disk-4)
            kernel_d4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_d4)
            C = cv2.countNonZero(closed_img)

            # Feature 2: S (Erosion Ratio)
            # eroded_img = cv2.erode(img, kernel_square, iterations=2)
            openned_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_square)
            S = cv2.countNonZero(openned_img)
            X = cv2.countNonZero(img)
            
            ratio = (S / X) if X > 0 else 0
            features.append([C, ratio])

    return np.array(features)

def run_experiment(X_train, y_train, X_test, y_test, label, neurons, fig_train_no, fig_test_no):
    """
    Runs the neural network experiment 10 times, averages results, calculates stats,
    and generates plots with correct Figure numbers and axis labels.
    """
    print(f"Running {label} (Generating Figure {fig_train_no} & {fig_test_no})...")
    
    # Accumulate predictions over 10 runs to handle random initialization
    cumulative_preds = np.zeros(len(y_test))
    errors = []
    last_model = None

    for _ in range(10):
        # Using StandardScaler helps MLP convergence on this small dataset
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(neurons,), activation='tanh', 
                         solver='lbfgs', max_iter=5000)
        )
        model.fit(X_train, y_train)
        preds = np.maximum(model.predict(X_test), 0) # Clip negatives
        
        cumulative_preds += preds
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs(preds - y_test) / (y_test + 1e-5)) * 100
        errors.append(mape)
        last_model = model

    avg_preds = cumulative_preds / 10
    avg_error = np.mean(errors)
    
    # Calculate accuracy buckets for Table I
    sample_errors = np.abs(avg_preds - y_test) / (y_test + 1e-5)
    acc_10 = np.sum(sample_errors <= 0.10) / len(y_test) * 100
    acc_15 = np.sum(sample_errors <= 0.15) / len(y_test) * 100

    # --- PLOTTING ---
    
    # 1. Training Relationship Plot (Figure X)
    train_preds = last_model.predict(X_train)
    plt.figure(figsize=(8, 6))
    
    # Determine axes based on feature count
    if X_train.shape[1] > 1:
        # Method 3 (2 features): Sort by main feature C for cleaner plot
        sort_feat = X_train[:, 0]
        x_label = 'Closed Foreground Pixels (C)'
        plot_x = X_train[:, 0]
    else:
        # Method 1 & 2 (1 feature): Flatten
        sort_feat = X_train.flatten()
        # Specific label for Method 1 vs Method 2
        if "Method 1" in label:
            x_label = 'Number of Foreground Pixels'
        else:
            x_label = 'Number of Foreground Pixels after Closing Operation'
        plot_x = sort_feat
    
    sort_idx = np.argsort(sort_feat)
    
    plt.plot(sort_feat[sort_idx], y_train[sort_idx], 'b-', alpha=0.7, label='Ground Truth')
    plt.scatter(plot_x, train_preds, c='red', s=15, label='Estimated')
    
    plt.xlabel(x_label)
    plt.ylabel('Number of People')
    plt.title(f'Figure {fig_train_no}: {label} - Training Relationship')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{OUTPUT_DIR}/Figure_{fig_train_no:02d}_Training.png")
    plt.close()

    # 2. Test Results Plot (Figure Y)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Upper Plot: Count vs Ground Truth
    ax1.plot(y_test, 'b-', label='Ground Truth')
    ax1.plot(avg_preds, 'r.', markersize=8, label='Estimated')
    ax1.set_title(f'Figure {fig_test_no}: {label} - Test Results')
    ax1.set_ylabel('Number of People')   # Matches paper axis label
    ax1.set_xlabel('Sample Sequence')    # Matches paper axis label
    ax1.legend()
    ax1.grid(True, alpha=0.5)
    
    # Lower Plot: Error Percentage
    ax2.plot(sample_errors * 100, 'g.', markersize=8, label='Error %')
    ax2.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='10% Threshold')
    ax2.set_ylabel('Error Percentage (%)') # Matches paper axis label
    ax2.set_xlabel('Sample Sequence')      # Matches paper axis label
    ax2.legend()
    ax2.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Figure_{fig_test_no:02d}_Test.png")
    plt.close()

    return {
        "Method": label,
        "Average Error (%)": round(avg_error, 2),
        "Accuracy (<10%)": round(acc_10, 2),
        "Accuracy (<15%)": round(acc_15, 2)
    }

def main():
    # Load Data
    data = load_and_split_dataset(BASE_DIR)
    if not data: return
    train_X_paths, train_y, test_X_paths, test_y = data
    
    results = []

    # --- Method 1 (Figs 5 & 6) ---
    # Section IV-A: 10 neurons
    X_train = extract_features(train_X_paths, 'method1')
    X_test = extract_features(test_X_paths, 'method1')
    results.append(run_experiment(X_train, train_y, X_test, test_y, 
                                  "Method 1", 10, fig_train_no=5, fig_test_no=6))

    # --- Method 2 disk-4 (Figs 7 & 8) ---
    X_train = extract_features(train_X_paths, 'method2', disk_radius=4)
    X_test = extract_features(test_X_paths, 'method2', disk_radius=4)
    results.append(run_experiment(X_train, train_y, X_test, test_y, 
                                  "Method 2 (disk-4)", 10, fig_train_no=7, fig_test_no=8))

    # --- Method 2 disk-5 (Figs 9 & 10) ---
    X_train = extract_features(train_X_paths, 'method2', disk_radius=5)
    X_test = extract_features(test_X_paths, 'method2', disk_radius=5)
    results.append(run_experiment(X_train, train_y, X_test, test_y, 
                                  "Method 2 (disk-5)", 10, fig_train_no=9, fig_test_no=10))

    # --- Method 3 (Figs 11 & 12) ---
    # Section IV-A: 10 neurons
    X_train = extract_features(train_X_paths, 'method3')
    X_test = extract_features(test_X_paths, 'method3')
    results.append(run_experiment(X_train, train_y, X_test, test_y, 
                                  "Method 3", 10, fig_train_no=11, fig_test_no=12))

    # --- Report ---
    df = pd.DataFrame(results)
    print("\nTable I - Reproduced Results:")
    print(df.to_string(index=False))

    # Save Table as Image
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    tbl.scale(1, 1.5)
    plt.title("Reproduced Table I")
    plt.savefig(f"{OUTPUT_DIR}/Table_I.png", bbox_inches='tight')
    
    print(f"\nDone. Outputs saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()