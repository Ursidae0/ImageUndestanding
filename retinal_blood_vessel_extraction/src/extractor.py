
import cv2
import numpy as np
import pickle
import os
import sys

# Import core modules (assumes they are in the same package/directory)
try:
    from .preprocessing import preprocess_image
    from .features import extract_features
    from .segmentation import (
        flatten_features, apply_pca, perform_fcm, 
        apply_classification, reconstruct_mask, post_process
    )
except ImportError:
    # Fallback for direct execution
    from preprocessing import preprocess_image
    from features import extract_features
    from segmentation import (
        flatten_features, apply_pca, perform_fcm, 
        apply_classification, reconstruct_mask, post_process
    )

class RetinalVesselExtractor:
    """
    Implements the hybrid Retinal Blood Vessel Extraction algorithm 
    (Hashemzadeh et al., 2019) with corrected logic.
    """
    
    def __init__(self, model_path=None):
        self.pipeline_model = None
        self.classifier = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, path):
        print(f"Loading model from {path}...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            # Expecting {'pipeline': {'scaler':..., 'pca':...}, 'classifier': clf}
            # Or legacy format check
            if 'model' in data and 'pca' in data: # Legacy/Previous format
                 # Adaptation for new structure if possible, but better to retrain
                 print("Warning: Loading legacy model format. Standardization might be missing.")
                 self.pipeline_model = {'scaler': None, 'pca': data['pca']} # Scaler missing in legacy
                 self.classifier = data['model']
            else:
                 self.pipeline_model = data.get('pipeline_model')
                 self.classifier = data.get('classifier')

    def process_image(self, image, fov_mask):
        """
        Full pipeline execution on a single image.
        Returns the binary vessel mask.
        """
        # 1. Preprocessing
        crop_img, crop_mask, g, y, l = preprocess_image(image, fov_mask)
        
        # 2. Feature Extraction (13 Features)
        feature_maps = extract_features(g, y, l, crop_mask)
        
        # Flatten
        features_fov, indices, shape_2d = flatten_features(feature_maps, crop_mask)
        
        if features_fov.shape[0] == 0:
            return np.zeros(shape_2d, dtype=np.uint8)
            
        # 3. Standardization & PCA
        # If model exists, use it. If not, we can't properly run supervised part correctly without trained scaler.
        # But for unsupervised part (FCM), we could technically run on fresh PCA, but features 0-1 vs 0-255 need scaler.
        
        if self.pipeline_model is None:
            # Unsupervised mode / First run fallback? 
            # Prompt implies we need to RUN. We will fit on the fly if needed, 
            # but warn that supervised classifier needs training.
            print("Info: No pre-trained model. Fitting Scaler+PCA on single image (Unsupervised-only mostly valid).")
            pca_features, self.pipeline_model = apply_pca(features_fov, model_dict=None)
            # If we just fit it, we have no classifier trained on this space.
            clf_ready = False
        else:
            pca_features, _ = apply_pca(features_fov, model_dict=self.pipeline_model)
            clf_ready = True
            
        # 4. Hybrid Classification
        # 4a. Unsupervised (FCM)
        mask_vessel_fcm_indices, mask_bg_fcm_indices = perform_fcm(pca_features)
        
        # 4b. Supervised (Decision Tree)
        mask_vessel_supervised = np.zeros(shape_2d, dtype=np.uint8)
        
        if clf_ready and self.classifier:
            # Predict only on background candidates
            preds = apply_classification(self.classifier, pca_features, mask_bg_fcm_indices)
            
            # Map predictions
            fov_rows, fov_cols = indices
            bg_rows = fov_rows[mask_bg_fcm_indices]
            bg_cols = fov_cols[mask_bg_fcm_indices]
            
            # Identify vessels (class 1)
            is_vessel = (preds == 1)
            sup_vessel_rows = bg_rows[is_vessel]
            sup_vessel_cols = bg_cols[is_vessel]
            
            mask_vessel_supervised[sup_vessel_rows, sup_vessel_cols] = 255
            
        # Reconstruct FCM mask
        mask_vessel_fcm = reconstruct_mask(indices, mask_vessel_fcm_indices, shape_2d)
        
        # 5. Post-Processing & Combination
        final_output = post_process(mask_vessel_fcm, mask_vessel_supervised, crop_mask)
        
        return final_output

    def train(self, train_data_path, output_path):
        """
        Train the pipeline (Scaler, PCA, Classifier) on a dataset.
        train_data_path: Path to DRIVE/training folder
        """
        # Logic similar to previous train.py but using class structures
        from sklearn.tree import DecisionTreeClassifier
        
        print("Starting training...")
        images_dir = os.path.join(train_data_path, 'images')
        masks_dir = os.path.join(train_data_path, 'mask')
        manual_dir = os.path.join(train_data_path, '1st_manual')
        
        X_all = []
        y_all = []
        
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])
        
        for img_file in image_files:
            img_id = img_file.split('_')[0]
            print(f"Processing {img_id}...")
            
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, f"{img_id}_training_mask.gif")
            manual_path = os.path.join(manual_dir, f"{img_id}_manual1.gif")
            
            if not os.path.exists(mask_path): continue
            
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            fov_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            manual_label = cv2.imread(manual_path, cv2.IMREAD_GRAYSCALE)
            
            # Preprocess
            crop_img, crop_mask, g, y, l = preprocess_image(image, fov_mask)
            
            # Crop manual to FOV - Need helper
            from preprocessing import get_fov_bounding_box
            rmin, rmax, cmin, cmax = get_fov_bounding_box(fov_mask)
            manual_cropped = manual_label[rmin:rmax+1, cmin:cmax+1]
            
            # Features
            feature_maps = extract_features(g, y, l, crop_mask)
            features_fov, indices, _ = flatten_features(feature_maps, crop_mask)
            
            # Labels
            rows, cols = indices
            labels_fov = manual_cropped[rows, cols]
            labels_fov = (labels_fov > 127).astype(np.uint8)
            
            X_all.append(features_fov)
            y_all.append(labels_fov)
            
        X_train = np.vstack(X_all)
        y_train = np.concatenate(y_all)
        
        print("Fitting Scaler & PCA...")
        # Fit Scaler + PCA
        X_pca, self.pipeline_model = apply_pca(X_train, model_dict=None)
        
        print("Training Decision Tree...")
        self.classifier = DecisionTreeClassifier(criterion='gini', max_depth=None)
        self.classifier.fit(X_pca, y_train)
        
        # Save
        data = {
            'pipeline_model': self.pipeline_model,
            'classifier': self.classifier
        }
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {output_path}")

if __name__ == "__main__":
    # Simple CLI
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    parser.add_argument('--image', help='Input image path')
    parser.add_argument('--mask', help='Input FOV mask path')
    parser.add_argument('--output', help='Output image path/model path')
    parser.add_argument('--dataset', help='Dataset path for training')
    parser.add_argument('--model', help='Model path to load', default='retina_model_v2.pkl')
    
    args = parser.parse_args()
    
    extractor = RetinalVesselExtractor(model_path=args.model if args.mode == 'predict' else None)
    
    if args.mode == 'train':
        extractor.train(args.dataset, args.output)
    elif args.mode == 'predict':
        img = cv2.imread(args.image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fov = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        result = extractor.process_image(img, fov)
        cv2.imwrite(args.output, result)
        print(f"Saved to {args.output}")
