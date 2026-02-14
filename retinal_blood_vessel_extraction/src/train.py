
import cv2
import numpy as np
import os
import sys
import pickle
import argparse
from sklearn.tree import DecisionTreeClassifier

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import preprocess_image
from features import extract_features
from segmentation import flatten_features, apply_pca

def train_model(train_dir, output_model_path):
    print("Starting training...")
    
    images_dir = os.path.join(train_dir, 'images')
    masks_dir = os.path.join(train_dir, 'mask')
    manual_dir = os.path.join(train_dir, '1st_manual')
    
    X_all = []
    y_all = []
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])
    
    print(f"Found {len(image_files)} training images.")
    
    for img_file in image_files:
        img_id = img_file.split('_')[0] 
        img_path = os.path.join(images_dir, img_file)
        mask_filename = f"{img_id}_training_mask.gif"
        mask_path = os.path.join(masks_dir, mask_filename)
        manual_filename = f"{img_id}_manual1.gif"
        manual_path = os.path.join(manual_dir, manual_filename)
        
        if not os.path.exists(mask_path) or not os.path.exists(manual_path):
            print(f"Skipping {img_id}: Mask or manual label not found.")
            continue
            
        print(f"Processing {img_id}...")
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fov_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        manual_label = cv2.imread(manual_path, cv2.IMREAD_GRAYSCALE)
        
        crop_img, crop_mask, g_clahe, y_clahe, l_clahe, _ = preprocess_image(image, fov_mask, debug_mode=True)
        
        from preprocessing import get_fov_bounding_box
        rmin, rmax, cmin, cmax = get_fov_bounding_box(fov_mask)
        manual_cropped = manual_label[rmin:rmax+1, cmin:cmax+1]
        
        feature_maps, _ = extract_features(g_clahe, y_clahe, l_clahe, crop_mask, debug_mode=True)
        features_fov, indices, _ = flatten_features(feature_maps, crop_mask)
        
        rows, cols = indices
        labels_fov = manual_cropped[rows, cols]
        labels_fov = (labels_fov > 127).astype(np.uint8)
        
        X_all.append(features_fov)
        y_all.append(labels_fov)
        
    print("Concatenating data...")
    X_train = np.vstack(X_all)
    y_train = np.concatenate(y_all)
    
    print(f"Training data shape: {X_train.shape}")
    
    print("Fitting Scaler and PCA...")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    
    pca = PCA(n_components=13)
    X_train_pca = pca.fit_transform(X_train_norm)
    
    print("Training Decision Tree...")
    clf = DecisionTreeClassifier(criterion='gini', max_depth=None) 
    clf.fit(X_train_pca, y_train)
    
    saved_data = {
        'model': clf,
        'pca': pca,
        'scaler': scaler
    }
    
    with open(output_model_path, 'wb') as f:
        pickle.dump(saved_data, f)
        
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True, help='Path to DRIVE/training directory')
    parser.add_argument('--output', required=True, help='Path to save output .pkl model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.train_dir):
        print(f"Error: {args.train_dir} does not exist.")
        sys.exit(1)
        
    train_model(args.train_dir, args.output)
