
import numpy as np
import cv2
from sklearn.decomposition import PCA
import skfuzzy as fuzz
from sklearn.tree import DecisionTreeClassifier

def flatten_features(feature_maps, fov_mask):
    """
    Flattens feature maps to (N_samples, N_features) restricting to FOV.
    Returns features and indices to reconstruct image.
    """
    row_indices, col_indices = np.where(fov_mask > 0)
    
    # Stack features: (H, W, 13)
    # feature_maps is a list of (H, W) arrays
    stacked = np.stack(feature_maps, axis=-1)
    
    # Extract pixels inside FOV
    features_fov = stacked[row_indices, col_indices, :]
    
    return features_fov, (row_indices, col_indices), stacked.shape[:2]

from sklearn.preprocessing import StandardScaler

def apply_pca(features_fov, model_dict=None):
    """
    Step 3: Standardization & PCA
    - Apply StandardScaler.
    - Apply PCA (n_components=13).
    """
    if model_dict is None:
        scaler = StandardScaler()
        norm_features = scaler.fit_transform(features_fov)
        
        pca = PCA(n_components=13)
        pca_features = pca.fit_transform(norm_features)
        
        return pca_features, {'scaler': scaler, 'pca': pca}
    else:
        scaler = model_dict['scaler']
        pca = model_dict['pca']
        
        norm_features = scaler.transform(features_fov)
        pca_features = pca.transform(norm_features)
        
        return pca_features, model_dict

def perform_fcm(pca_features):
    """
    Step 4: Unsupervised Extraction (Clustering)
    - Fuzzy C-Means (FCM).
    - Clusters (C) = 2, Fuzziness (m) = 2, Max Iter = 2000, Tolerance = 1e-6.
    """
    # skfuzzy.cmeans expects data shape (N_features, N_samples). 
    # Our pca_features is (N_samples, N_features). Transpose it.
    data = pca_features.T
    
    # cntr, u, u0, d, jm, p, fpc
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, c=2, m=2, error=1e-6, maxiter=2000, init=None
    )
    
    # u is (C, N_samples) membership matrix.
    # We assign cluster by max membership.
    cluster_labels = np.argmax(u, axis=0)
    
    # Logic: "Identify the Vessel Cluster as the one with fewer pixels".
    count_0 = np.sum(cluster_labels == 0)
    count_1 = np.sum(cluster_labels == 1)
    
    if count_0 < count_1:
        vessel_cluster = 0
    else:
        vessel_cluster = 1
        
    mask_vessel_fcm_indices = (cluster_labels == vessel_cluster)
    mask_background_fcm_indices = (cluster_labels != vessel_cluster)
    
    return mask_vessel_fcm_indices, mask_background_fcm_indices

def apply_classification(classifier, pca_features, mask_background_fcm_indices):
    """
    Step 5: Supervised Extraction (Classification)
    - Apply classifier ONLY to pixels belonging to mask_background_fcm.
    - Reject pixels are those NOT in vessel cluster (i.e. background candidates).
    """
    # Get features for background candidates
    bg_features = pca_features[mask_background_fcm_indices]
    
    if len(bg_features) == 0:
        return np.array([])
        
    # Predict
    # Assumes classifier is trained on similar PCA features.
    # User prompt implies we construct the architecture.
    # We need a trained classifier. If 'classifier' is None or untrained, we can't predict.
    # For now, we assume 'classifier' is a valid sk-learn model.
    if hasattr(classifier, "predict_proba"):
        # We can get probabilities
        # Check if we want probabilities or binary
        # For now, let's just return prediction unless we specifically want separate proba function.
        # But to avoid breaking existing callers, let's assume this function returns binary classes 
        # unless we modify the signature.
        pass
    
    predictions = classifier.predict(bg_features)
    
    return predictions

def apply_classification_proba(classifier, pca_features, mask_background_fcm_indices):
    """
    Returns probability of VESSEL (class 1) for background candidates.
    """
    bg_features = pca_features[mask_background_fcm_indices]
    
    if len(bg_features) == 0:
        return np.array([])
        
    # Predict Proba
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(bg_features)
        # Check shape. Usually (N, 2). We want column 1.
        if probs.shape[1] > 1:
            return probs[:, 1]
        else:
            return probs[:, 0] # Fallback
    else:
        # Fallback to binary
        return classifier.predict(bg_features).astype(float)


def train_dummy_classifier():
    """
    Creates a dummy classifier to satisfy pipeline if no weights provided.
    In real scenario, this should define the model structure.
    """
    clf = DecisionTreeClassifier(criterion='gini', max_depth=None) # Root Guided or Standard
    return clf

def reconstruct_mask(indices, mask_values, shape):
    """
    Reconstructs the 2D mask from sparse values.
    """
    rows, cols = indices
    mask = np.zeros(shape, dtype=np.uint8)
    # mask_values is boolean or 0/1
    mask[rows, cols] = mask_values.astype(np.uint8) * 255
    return mask

def post_process(mask_vessel_fcm, mask_vessel_supervised, fov_mask):
    """
    Step 6: Post-Processing & Combination
    - Combine: Final = FCM OR Supervised
    - Clean: Erode original FOV and apply.
    """
    # Combine
    # Note: mask_vessel_supervised is full size? 
    # The logic in 'apply_classification' returned predictions for a subset.
    # We need to map them back to the full image carefully in the main pipeline.
    # Here function assumes we have two full-size masks (or handles logic).
    
    # Let's assume input args are full images (0 or 255).
    final_vessel = cv2.bitwise_or(mask_vessel_fcm, mask_vessel_supervised)
    
    # Clean: Erode FOV mask slightly to remove border artifacts.
    # Using minimal erosion (3x3 kernel, 1 iteration) to preserve vessel details.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fov_eroded = cv2.erode(fov_mask.astype(np.uint8), kernel, iterations=9)

    
    # Apply clean FOV
    final_vessel = cv2.bitwise_and(final_vessel, final_vessel, mask=fov_eroded)
    
    return final_vessel
