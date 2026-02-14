
import cv2
import numpy as np
import os
import sys
import glob
import pickle

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import preprocess_image
from features import extract_features
import segmentation as seg
from utils import save_figure_grid
from evaluation import PerformanceEvaluator


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_normalized(path, img):
    # Normalize float to 0-255 uint8
    if img.dtype != np.uint8:
        img = np.nan_to_num(img)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    cv2.imwrite(path, img)

def run_reproduction(data_root="dataset/DRIVE/test", output_root="paper_figures_reproduction", model_path='retina_model.pkl'):
    
    images_dir = os.path.join(data_root, "images")

    masks_dir = os.path.join(data_root, "mask")
    
    ensure_dir(output_root)
    
    # Define subfolders per Figure
    fig_folders = {
        "Fig3": os.path.join(output_root, "Fig3_Channels"),
        "Fig4": os.path.join(output_root, "Fig4_CLAHE"),
        "Fig5": os.path.join(output_root, "Fig5_GaborResponse"),
        "Fig6": os.path.join(output_root, "Fig6_GaborBinary"),
        "Fig7": os.path.join(output_root, "Fig7_TopHat"),
        "Fig8": os.path.join(output_root, "Fig8_ShadeCorrected"),
        "Fig9": os.path.join(output_root, "Fig9_BPS"),
        "Fig10": os.path.join(output_root, "Fig10_FCM"),
        "Fig11": os.path.join(output_root, "Fig11_Hybrid"),
        "Fig12": os.path.join(output_root, "Fig12_Final")
    }
    for p in fig_folders.values(): ensure_dir(p)
    
    # Evaluation
    evaluator = PerformanceEvaluator()
    metrics_list = []
    
    # Ground Truth Dir
    # Try to find 1st_manual or 2nd_manual in data_root
    # For TEST set, we often have 2nd_manual. For TRAINING, 1st_manual.
    # User requested 2nd manual for evaluation.
    
    manual_dir = os.path.join(data_root, "2nd_manual")
    manual_suffix = "_manual2.gif"
    
    if not os.path.exists(manual_dir):
        # Fallback to 1st_manual if 2nd not found (e.g. if running on training set again)
        manual_dir = os.path.join(data_root, "1st_manual")
        manual_suffix = "_manual1.gif"
        
    if not os.path.exists(manual_dir):
        print(f"Warning: No Ground truth directory (1st_manual or 2nd_manual) found in {data_root}. Evaluation will be skipped.")


    
    # Load Model if exists
    loaded_model_data = None
    loaded_clf = None
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            loaded_clf = data['model']
            # We need the full dict for apply_pca if it has scaler
            if 'scaler' in data and 'pca' in data:
                loaded_model_data = data
            elif 'pca' in data:
                 # Legacy support if only pca
                 # But our apply_pca expects dict['pca']
                 loaded_model_data = {'pca': data['pca']}
    else:
        print(f"Model {model_path} not found. Running unsupervised/dummy mode.")
            
    # Process Images
    if not os.path.exists(images_dir):
        print(f"Error: Images dir {images_dir} does not exist.")
        return

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
    print(f"Found {len(image_paths)} images in {images_dir}.")
    
    for img_path in image_paths:
        fname = os.path.basename(img_path)
        img_id = fname.split('_')[0]
        # Handle different naming conventions in training vs test (e.g. 21_training.tif vs 01_test.tif)
        is_training = "_training" in fname
        print(f"Processing {img_id}...")
        
        mask_path = os.path.join(masks_dir, f"{img_id}_{'training' if is_training else 'test'}_mask.gif")
        if not os.path.exists(mask_path): 
            print(f"Mask not found: {mask_path}")
            continue
            
            continue
            
        # Ground Truth
        gt_path = os.path.join(manual_dir, f"{img_id}{manual_suffix}") if os.path.exists(manual_dir) else None
        
        # Load

        original_rgb = cv2.imread(img_path) # BGR

        if original_rgb is None: continue
        image_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)
        fov_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 1. Preprocessing
        # Fig 3, 4
        # Returns: crop_img, crop_mask, g_clahe, y_clahe, l_clahe, debug_data
        crop_img, crop_mask, g_clahe, y_clahe, l_clahe, deb_pre = preprocess_image(image_rgb, fov_mask, debug_mode=True)
        
        # Save Fig 3 (a, b, c) -> G, Y, L raw
        # f3_names = ['a_G', 'b_Y', 'c_L']
        # for name, img in zip(f3_names, deb_pre['fig3']):
        #     save_normalized(os.path.join(fig_folders['Fig3'], f"{img_id}_Fig3_{name}.png"), img)
        save_figure_grid(
            deb_pre['fig3'], 
            ['(a) Green Channel', '(b) Y Channel', '(c) L Channel'],
            os.path.join(fig_folders['Fig3'], f"{img_id}_Fig3_Combined.png"),
            (1, 3)
        )
            
        # Save Fig 4 (a, b, c) -> G, Y, L enhanced
        # for name, img in zip(f3_names, deb_pre['fig4']):
        #     save_normalized(os.path.join(fig_folders['Fig4'], f"{img_id}_Fig4_{name}.png"), img)
        save_figure_grid(
            deb_pre['fig4'], 
            ['(a) Enh Green', '(b) Enh Y', '(c) Enh L'],
            os.path.join(fig_folders['Fig4'], f"{img_id}_Fig4_Combined.png"),
            (1, 3)
        )

            
        # 2. Features
        # Returns: feature_maps, debug_data
        features, deb_feat = extract_features(g_clahe, y_clahe, l_clahe, crop_mask, debug_mode=True)
        
        # Save Fig 5 & 6
        # Structure: Rows G, Y, L. Cols 9, 10, 11 via lambda.
        if 'fig5' in deb_feat and len(deb_feat['fig5']) == 9:
            titles_5 = []
            for r in ['Green', 'Y', 'L']:
                for c in ['9', '10', '11']:
                    titles_5.append(f"{r} (λ={c})")
            save_figure_grid(
                deb_feat['fig5'],
                titles_5,
                os.path.join(fig_folders['Fig5'], f"{img_id}_Fig5_Combined.png"),
                (3, 3)
            )

        if 'fig6' in deb_feat and len(deb_feat['fig6']) == 9:
            titles_6 = []
            for r in ['Green', 'Y', 'L']:
                for c in ['9', '10', '11']:
                    titles_6.append(f"Binary {r} (λ={c})")
            save_figure_grid(
                deb_feat['fig6'],
                titles_6,
                os.path.join(fig_folders['Fig6'], f"{img_id}_Fig6_Combined.png"),
                (3, 3)
            )

                    
        # Save Fig 7 (TH) - Combined
        if 'fig7' in deb_feat:
            save_figure_grid(
                [g_clahe, deb_feat['fig7']],
                ['(a) Input', '(b) Top-Hat'],
                os.path.join(fig_folders['Fig7'], f"{img_id}_Fig7_Combined.png"),
                (1, 2)
            )
            
        # Save Fig 8 (SC) - Combined
        if 'fig8' in deb_feat:
            save_figure_grid(
                [g_clahe, deb_feat['fig8']],
                ['(a) Input', '(b) Shade Corrected'],
                os.path.join(fig_folders['Fig8'], f"{img_id}_Fig8_Combined.png"),
                (1, 2)
            )
            
        # Save Fig 9 (BPS)
        # (a) Input, (b)-(i) Planes 1-8, (j) Sum of last two
        if 'fig9_planes' in deb_feat and 'fig9' in deb_feat:
            # (a) Input
            imgs_9 = [g_clahe]
            titles_9 = ['(a) Input']
            
            # (b)-(i) Planes 
            plane_names = ['(b) 1st', '(c) 2nd', '(d) 3rd', '(e) 4th', '(f) 5th', '(g) 6th', '(h) 7th', '(i) 8th']
            for i, p in enumerate(deb_feat['fig9_planes']):
                if i < len(plane_names):
                    imgs_9.append(p)
                    titles_9.append(plane_names[i])
            
            # (j) Sum
            imgs_9.append(deb_feat['fig9'].astype(np.uint8))
            titles_9.append('(j) Final BPS')
            
            save_figure_grid(
                imgs_9,
                titles_9,
                os.path.join(fig_folders['Fig9'], f"{img_id}_Fig9_Combined.png"),
                (2, 5) # 10 images total
            )

            
        # 3. Segmentation
        features_fov, indices, shape_2d = seg.flatten_features(features, crop_mask) 
        if features_fov.size == 0: continue
        
        # PCA
        if loaded_model_data: pca_feat, _ = seg.apply_pca(features_fov, loaded_model_data)
        else: pca_feat, pca_mod = seg.apply_pca(features_fov)
        
        # FCM
        # mask_vessel_fcm_indices, mask_bg_fcm_indices
        v_idx, bg_idx = seg.perform_fcm(pca_feat)
        mask_fcm = seg.reconstruct_mask(indices, v_idx, shape_2d)
        
        # Fig 10 - Combined
        save_figure_grid(
            [cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR), mask_fcm],
            ['(a) Input', '(b) FCM'],
            os.path.join(fig_folders['Fig10'], f"{img_id}_Fig10_Combined.png"),
            (1, 2)
        )
        
        # Classification
        clf = loaded_clf
        if clf is None:
            # Dummy
            clf = seg.train_dummy_classifier()
            clf.fit(np.random.rand(10, 13), np.random.randint(0,2,10))
            
        preds = seg.apply_classification(clf, pca_feat, bg_idx)
        
        mask_sup = np.zeros(shape_2d, dtype=np.uint8)
        r, c = indices
        if len(preds) > 0:
            mask_sup[r[bg_idx][preds==1], c[bg_idx][preds==1]] = 255
            
        # Hybrid
        mask_hybrid = cv2.bitwise_or(mask_fcm, mask_sup)
        
        # PROBABILITY MAP for AUC
        # Initialize with FCM background probability (approx 0 for bg, but we want prob of VESSEL)
        # FCM gives hard clusters. 
        # If in Vessel Cluster -> Confident Vessel (Prob = 1.0)
        # If in Background Cluster -> Use Classifier Probabilities
        
        prob_map = np.zeros(shape_2d, dtype=float)
        
        # Identify FCM Vessel pixels
        r, c = indices
        prob_map[r[v_idx], c[v_idx]] = 1.0
        
        # Identify FCM Background pixels (Candidates for Classification)
        # We need probabilities for these
        probs_bg = seg.apply_classification_proba(clf, pca_feat, bg_idx)
        
        if len(probs_bg) > 0:
            prob_map[r[bg_idx], c[bg_idx]] = probs_bg
        
        # Reconstruct full image probability map? 
        # Actually `prob_map` is just for the FOV pixels (using indices).
        # But `indices` maps to (rows, cols) of the full image.
        # So mask construction above is correct.
        
        # Fig 11 - Combined
        save_figure_grid(
            [cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR), mask_fcm, mask_hybrid],
            ['(a) Input', '(b) Clustering', '(c) Hybrid'],
            os.path.join(fig_folders['Fig11'], f"{img_id}_Fig11_Combined.png"),
            (1, 3)
        )
        
        # Post Process
        final = seg.post_process(mask_fcm, mask_sup, crop_mask)
        
        # Fig 12
        # (a) Input, (b) Before, (c) Final
        # cv2.imwrite(os.path.join(fig_folders['Fig12'], f"{img_id}_Fig12_a_Input.png"), cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(os.path.join(fig_folders['Fig12'], f"{img_id}_Fig12_b_BeforePost.png"), mask_hybrid)
        # cv2.imwrite(os.path.join(fig_folders['Fig12'], f"{img_id}_Fig12_c_Final.png"), final)
        
        save_figure_grid(
            [cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR), mask_hybrid, final],
            ['(a) Input', '(b) Before Post-Proc', '(c) Final'],
            os.path.join(fig_folders['Fig12'], f"{img_id}_Fig12_Combined.png"),
            (1, 3)
        )
        
        # Calculate Metrics
        if gt_path and os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_img is not None:
                # Need to crop GT to match crop_img if necessary? 
                # preprocees_image crops the input based on mask bbox.
                # Currently we don't have the crop coordinates exposed easily from preprocess_image return...
                # BUT, preprocess_image returns `crop_mask` which is the cropped mask.
                # We need the SAME crop for GT.
                # Actually, preprocess_image returns `debug_data` which might help, or we should modify preprocess logic.
                # For now, let's assume we can get the crop coords or re-crop.
                # Wait, `preprocess_image` implementation in `preprocessing.py` creates the bounding box.
                # We should probably pass the full GT to `calculate_metrics` and let it handle... NO.
                # We need to crop GT exactly as we cropped the image.
                
                # Re-calculate bbox from fov_mask
                contours, _ = cv2.findContours(fov_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    max_cnt = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(max_cnt)
                    crop_gt = gt_img[y:y+h, x:x+w]
                    
                    # Ensure shapes match
                    if crop_gt.shape != final.shape:
                        crop_gt = cv2.resize(crop_gt, (final.shape[1], final.shape[0]), interpolation=cv2.INTER_NEAREST)
                        
                    met = evaluator.calculate_metrics(final, crop_gt, crop_mask)
                    
                    # Calculate AUC
                    # Need probability map of same shape as final/crop_gt
                    # Current `prob_map` is full size (shape_2d). Need to crop/adjust?
                    # `post_process` uses `mask_sup` which uses `shape_2d`.
                    # So `prob_map` is 565x584 (example).
                    # `final` comes from `post_process` which handles `fov_mask`.
                    # `crop_img` comes from `preprocess_image` which crops to bbox.
                    # Wait, `flatten_features` uses `crop_mask` (which is small).
                    # So `shape_2d` returned by `flatten_features` is the CROPPED shape (h, w).
                    # Let's verify: 
                    # `features, deb_feat = extract_features(..., crop_mask, ...)` -> returns maps of size crop_mask.
                    # `features_fov, indices, shape_2d = seg.flatten_features(features, crop_mask)` -> shape_2d is crop_mask.shape.
                    # So `prob_map` isALREADY cropped size.
                    
                    auc = evaluator.calculate_auc(prob_map, crop_gt, crop_mask)
                    met['AUC'] = auc
                    
                    met['Image'] = img_id
                    metrics_list.append(met)
        
        print(f"Done {img_id}")
        
    # Print Table 1
    if metrics_list:
        print("\nTable 1: Performance Evaluation")
        print(f"{'Image':<10} | {'Acc':<8} | {'Sens':<8} | {'Spec':<8} | {'Prec':<8} | {'AUC':<8}")
        print("-" * 66)
        
        acc_scores = []
        sens_scores = []
        spec_scores = []
        prec_scores = []
        auc_scores = []
        
        for m in metrics_list:
            # print(f"{m['Image']:<10} | {m['Accuracy']:.4f}   | {m['Sensitivity']:.4f}   | {m['Specificity']:.4f}   | {m['Precision']:.4f}   | {m['AUC']:.4f}")
            acc_scores.append(m['Accuracy'])
            sens_scores.append(m['Sensitivity'])
            spec_scores.append(m['Specificity'])
            prec_scores.append(m['Precision'])
            auc_scores.append(m['AUC'])
            
        print("-" * 66)
        print(f"{'MEAN':<10} | {np.mean(acc_scores):.4f}   | {np.mean(sens_scores):.4f}   | {np.mean(spec_scores):.4f}   | {np.mean(prec_scores):.4f}   | {np.mean(auc_scores):.4f}")
        print("-" * 66)
        print("Paper Results (DRIVE):")
        print(f"{'2nd Human':<10} | {'0.9464':<8} | {'0.7796':<8} | {'0.9717':<8} | {'0.8072':<8} | {'-':<8}")
        print(f"{'Proposed':<10}  | {'0.9531':<8} | {'0.7830':<8} | {'0.9800':<8} | {'0.8594':<8} | {'0.9752':<8}")




if __name__ == "__main__":
    run_reproduction()
