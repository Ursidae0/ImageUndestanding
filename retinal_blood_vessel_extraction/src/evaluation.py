import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

class PerformanceEvaluator:
    """
    Evaluates retinal vessel segmentation performance based on Hashemzadeh et al. (2019).
    Metrics are calculated ONLY for pixels inside the Field of View (FOV).
    """
    
    def __init__(self):
        pass

    def _apply_mask(self, prediction, ground_truth, fov_mask):
        """
        Flattens the arrays and selects only the pixels inside the FOV mask.
        
        Args:
            prediction: processed image/map
            ground_truth: ground truth image
            fov_mask: FOV mask
            
        Returns:
            pred_flat, gt_flat: 1D arrays of pixels inside the mask
        """
        # Ensure boolean mask
        mask = (fov_mask > 0)
        
        # Check shapes
        if prediction.shape != mask.shape or ground_truth.shape != mask.shape:
            raise ValueError(f"Shape mismatch: Pred {prediction.shape}, GT {ground_truth.shape}, Mask {mask.shape}")

        # Flatten and select
        pred_flat = prediction[mask]
        gt_flat = ground_truth[mask]
        
        return pred_flat, gt_flat

    def calculate_metrics(self, binary_prediction, ground_truth, fov_mask):
        """
        Calculates TP, TN, FP, FN, Accuracy, Sensitivity, Specificity, Precision.
        Equations (2)-(5) from Hashemzadeh et al. (2019).
        
        Args:
            binary_prediction: Binary image (0, 1) or (0, 255)
            ground_truth: Binary image (0, 1) or (0, 255)
            fov_mask: Binary mask (0, 255)
            
        Returns:
            dict: Dictionary containing all metrics
        """
        # Normalize inputs to 0/1 if they are 255
        # Threshold just in case inputs are not strict binary
        pred_norm = (binary_prediction > 127).astype(np.uint8) if binary_prediction.max() > 1 else binary_prediction.astype(np.uint8)
        gt_norm = (ground_truth > 127).astype(np.uint8) if ground_truth.max() > 1 else ground_truth.astype(np.uint8)
        
        # Apply FOV mask
        y_pred, y_true = self._apply_mask(pred_norm, gt_norm, fov_mask)
        
        # Calculate confusion matrix components
        # confusion_matrix returns [[TN, FP], [FN, TP]]
        # We explicitly set labels=[0, 1] to ensure consistent unpacking even if a class is missing
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate Metrics
        # Sensitivity (Recall) = TP / (TP + FN) [Source 154]
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity = TN / (TN + FP) [Source 155]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Accuracy = (TP + TN) / N_total [Source 154]
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        # PPV (Precision) = TP / (TP + FP) [Source 155]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        return {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Accuracy": accuracy,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision
        }

    def calculate_auc(self, probability_map, ground_truth, fov_mask):
        """
        Calculates Area Under the ROC Curve (AUC).
        [Source 155]
        
        Args:
            probability_map: Float array (0.0 to 1.0)
            ground_truth: Binary image (0, 1) or (0, 255)
            fov_mask: Binary mask (0, 255)
            
        Returns:
            float: AUC score
        """
        gt_norm = (ground_truth > 127).astype(np.uint8) if ground_truth.max() > 1 else ground_truth.astype(np.uint8)
        
        # Apply FOV mask
        prob_flat, gt_flat = self._apply_mask(probability_map, gt_norm, fov_mask)
        
        # Check if we have both classes
        if len(np.unique(gt_flat)) < 2:
            print("Warning: Only one class present in ground truth within FOV. Cannot calculate AUC.")
            return 0.0
            
        auc = roc_auc_score(gt_flat, prob_flat)
        return auc
