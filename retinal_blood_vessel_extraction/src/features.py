
import numpy as np
import cv2
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

def kittler_illingworth_threshold(image, mask):
    """
    Implements Minimum Error Thresholding (Kittler & Illingworth, 1986).
    Source: Reference [5] in Hashemzadeh et al. (2019).
    """
    # Get valid pixels within the FOV mask
    valid_pixels = image[mask > 0]
    if valid_pixels.size == 0:
        return 0

    # 1. Histogram Calculation (normalized to integer 0-255)
    valid_pixels = cv2.normalize(valid_pixels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hist, _ = np.histogram(valid_pixels, bins=256, range=(0, 256))
    hist = hist.astype(float)
    total_pixels = valid_pixels.size
    
    # 2. Iterate through all possible thresholds T
    # J(T) is the criterion to minimize
    min_J = np.inf
    best_T = 0
    
    # Optimization: Calculate cumulative sums (P1) and cumulative means
    # to avoid re-looping inside the loop
    pdf = hist / total_pixels
    P1_cum = np.cumsum(pdf)
    mean_cum = np.cumsum(np.arange(256) * pdf)
    
    # Total mean of the whole image
    total_mean = mean_cum[-1]

    for T in range(10, 250): # Avoid extreme ends to prevent log(0)
        # Class 1 (Background): 0 to T
        P1 = P1_cum[T]
        # Class 2 (Vessel): T+1 to 255
        P2 = 1.0 - P1
        
        if P1 < 1e-6 or P2 < 1e-6: continue
            
        # Means
        mu1 = mean_cum[T] / P1
        mu2 = (total_mean - mean_cum[T]) / P2
        
        # Variances (sigma^2)
        # Var1 = E[x^2]_1 - mu1^2
        # Simple iterative variance calc using vectorized operations for speed
        # However, for speed in Python loop, we can use precomputed E[x^2] similarly to mean, 
        # but the prompt provided a loop structure. Let's stick to the prompt's loop approach for clarity/faithfulness.
        # sigma1_sq = sum(((i - mu1)**2) * pdf[i]) / P1
        
        # We can implement it efficiently inside loop:
        # Variance calculation within the loop
        
        # Class 1 variance
        indices1 = np.arange(T+1)
        sigma1_sq = np.sum(((indices1 - mu1)**2) * pdf[:T+1]) / P1
        
        # Class 2 variance
        indices2 = np.arange(T+1, 256)
        sigma2_sq = np.sum(((indices2 - mu2)**2) * pdf[T+1:]) / P2
        
        if sigma1_sq < 1e-9: sigma1_sq = 1e-9
        if sigma2_sq < 1e-9: sigma2_sq = 1e-9
        
        # Kittler-Illingworth Criterion J(T)
        sigma1 = np.sqrt(sigma1_sq)
        sigma2 = np.sqrt(sigma2_sq)
        
        J = 1 + 2 * (P1 * np.log(sigma1) + P2 * np.log(sigma2)) \
            - 2 * (P1 * np.log(P1) + P2 * np.log(P2))
            
        if J < min_J:
            min_J = J
            best_T = T
            
    # Scale best_T back to image range if needed, but we normalized image to 0-255. 
    # The comparison later (img_new > T) needs T in same domain as img_new.
    # img_new is float. We normalized valid_pixels to 0-255 to find threshold.
    # So we must map best_T back to original range or map image to 0-255 for thresholding.
    # Mapping image to 0-255 is safer.
    
    # We return the threshold in 0-255 scale AND applies to normalized image.
    # But extract_gabor_features applies threshold to `img_new`.
    # Let's return the threshold in 0-255.
    return best_T

def extract_gabor_features(channel, fov_mask, debug_mode=False):
    """
    Step 2A: The 9 Gabor Features (Binary)
    Target:
    - Wavelengths [9, 10, 11] (Paper Section 3.3.1.1)
    - Orientations 0 to 360 step 15.
    - Max response of filter bank -> Laplace -> Add to original -> Threshold -> Binary.
    """
    gabor_features = []
    fig5_imgs = [] # Pre-threshold (Response)
    fig6_imgs = [] # Post-threshold (Binary)
    
    # Invert the channel: Vessels are dark in G/Y/L, but Gabor (psi=0) detects bright ridges.
    # Inverting makes vessels bright ridges.
    if channel.max() <= 1.0:
        channel_inv = 1.0 - channel
    else:
        channel_inv = 255.0 - channel

    # Wavelengths lambda in [9, 10, 11]
    wavelengths = [9, 10, 11] # Pixels
    
    # Orientations: 0 to 360 step 15 (24 orientations)
    # Paper: "starts from 0 and occurs at every 15 angle" + "bank of 72 Gabor filter"
    # 72 filters / 3 wavelengths = 24 orientations.
    orientations = np.arange(0, 360, 15) * np.pi / 180
    
    for l in wavelengths:
        # Create kernels for this wavelength
        
        responses_for_lambda = []
        
        for theta in orientations:
            # FIX PARAMETERS: Bandwidth = 1
            # Relationship: sigma = 0.56 * lambda * (bandwidth related factor?)
            # For bandwidth b=1, sigma = lambda * (1/pi) * sqrt(ln2/2) * ... 
            # Approx sigma = 0.56 * lambda
            sigma = 0.56 * l
            
            # Aspect Ratio (gamma): 0.5
            gamma = 0.5
            
            ksize = (int(4*sigma) | 1, int(4*sigma) | 1) # Make sure it's odd
            if ksize[0] < 3: ksize = (3,3)
            
            kernel = cv2.getGaborKernel(ksize, sigma, theta, l, gamma, 0, ktype=cv2.CV_32F)
            fimg = cv2.filter2D(channel_inv, cv2.CV_32F, kernel)
            responses_for_lambda.append(fimg)
            
        # 1. Take maximum response across orientations
        if not responses_for_lambda:
             max_response = np.zeros_like(channel, dtype=np.float32)
        else:
             max_response = np.max(np.array(responses_for_lambda), axis=0)
             
        # 2. Apply Laplace Filter to the Gabor response.
        # Laplace emphasizes edges. 
        laplace = cv2.Laplacian(max_response, cv2.CV_32F)
        
        # 3. Add Laplace result to the original Gabor response
        img_new = max_response + laplace
        
        # 4. THRESHOLDING (Kittler-Illingworth)
        # Normalize for consistent thresholding domain (0-255)
        # This image corresponds to Figure 5 (Response Image)
        img_min, img_max = img_new.min(), img_new.max()
        if img_max - img_min > 0:
            img_norm = (img_new - img_min) / (img_max - img_min) * 255
        else:
            img_norm = np.zeros_like(img_new)
            
        T_val = kittler_illingworth_threshold(img_new, fov_mask)
        
        # 5. Binarize
        # This image corresponds to Figure 6 (Binary Part)
        binary_feature = (img_norm > T_val).astype(np.float32)
        
        gabor_features.append(binary_feature)
        
        if debug_mode:
            fig5_imgs.append(img_norm)
            fig6_imgs.append(binary_feature * 255) # Scale to 0-255 for visibility

    if debug_mode:
        return gabor_features, fig5_imgs, fig6_imgs
        
    return gabor_features

def get_strong_features(g_img, fov_mask, debug_mode=False):
    """
    Extracts explicit strong features from G-channel.
    Features 10-13.
    """
    strong_features = []
    debug_imgs = {}
    
    # 10. G-Channel (CLAHE enhanced already passed in)
    strong_features.append(g_img.astype(np.float32))
    
    # 11. Top-Hat (TH)
    # Invert G-channel
    g_inv = 255 - g_img
    
    # Structuring Element: Linear, Length 21. Rotations 0..180.
    length = 21
    angles = np.arange(0, 180, 22.5)
    
    th_responses = []
    
    for ang in angles:
        kernel = np.zeros((length, length), dtype=np.uint8)
        ang_rad = np.deg2rad(ang)
        c = length // 2
        pt1 = (int(c + c * np.cos(ang_rad)), int(c - c * np.sin(ang_rad)))
        pt2 = (int(c - c * np.cos(ang_rad)), int(c + c * np.sin(ang_rad)))
        cv2.line(kernel, pt1, pt2, 1, thickness=1)
        
        th = cv2.morphologyEx(g_inv, cv2.MORPH_TOPHAT, kernel)
        th_responses.append(th)
        
    th_max = np.max(np.array(th_responses), axis=0)
    strong_features.append(th_max.astype(np.float32))
    
    if debug_mode:
        debug_imgs['fig7'] = th_max # Top-Hat Result
    
    # 12. Shade Corrected (SC)
    # Feature = Background - G_Channel (So vessels become bright positive differences)
    # G_channel has dark vessels.
    bg_est = cv2.medianBlur(g_img, 25)
    # Invert subtraction order to get bright vessels
    sc = bg_est.astype(np.float32) - g_img.astype(np.float32)
    strong_features.append(sc)
    
    if debug_mode:
        debug_imgs['fig8'] = sc # Shade Corrected
    
    # 13. Bit Plane Slicing (BPS)
    # Source: Top-Hat feature (th_max).
    # Fraz et al.: BPS is applied to "vessel enhanced image" (Top-Hat).
    # Vessels are bright in Top-Hat.
    if th_max.max() > th_max.min():
        th_norm = ((th_max - th_max.min()) / (th_max.max() - th_max.min()) * 255).astype(np.uint8)
    else:
        th_norm = th_max.astype(np.uint8)
        
    # User requested Last Two Planes (Bit 7 and Bit 8)
    # Bit 8 (MSB) = 128
    # Bit 7 = 64
    bps = (th_norm & 64) + (th_norm & 32)
    strong_features.append(bps.astype(np.float32))
    
    if debug_mode:
        debug_imgs['fig9'] = bps # BPS Result (Sum)
        # Extract all 8 planes for Fig 9 from the TOP-HAT image
        planes = []
        for i in range(8):
            # Bit i: (val >> i) & 1
            # Scale to 255 for visualization
            plane = ((th_norm >> i) & 1) * 255
            planes.append(plane.astype(np.uint8))
        debug_imgs['fig9_planes'] = planes
    
    if debug_mode:
        return strong_features, debug_imgs
    
    return strong_features

def extract_features(g_img, y_img, l_img, fov_mask, debug_mode=False):
    """
    Main feature extraction function.
    debug_mode: If True, returns (feature_maps, debug_data)
                where debug_data contains {'fig5': [9 images], 'fig6': [9 images]}
    """
    # 1. Gabor Features (9 total: 3 channels * 3 wavelengths)
    debug_data = {}
    
    if debug_mode:
        gabor_g, fig5_g, fig6_g = extract_gabor_features(g_img, fov_mask, debug_mode=True)
        gabor_y, fig5_y, fig6_y = extract_gabor_features(y_img, fov_mask, debug_mode=True)
        gabor_l, fig5_l, fig6_l = extract_gabor_features(l_img, fov_mask, debug_mode=True)
        
        # Combine debug images (Order: G1, G2, G3, Y1, Y2, Y3, L1, L2, L3 or per channel?)
        # Let's concatenate them
        debug_data['fig5'] = fig5_g + fig5_y + fig5_l
        debug_data['fig6'] = fig6_g + fig6_y + fig6_l
    else:
        gabor_g = extract_gabor_features(g_img, fov_mask) # 3 features
        gabor_y = extract_gabor_features(y_img, fov_mask) # 3 features
        gabor_l = extract_gabor_features(l_img, fov_mask) # 3 features
    
    # Total 9
    all_gabor = gabor_g + gabor_y + gabor_l
    
    # 2. Strong Features (4 total, only from G)
    if debug_mode:
        strong, debug_strong = get_strong_features(g_img, fov_mask, debug_mode=True)
        debug_data.update(debug_strong)
    else:
        strong = get_strong_features(g_img, fov_mask)
    
    # Combine all (13 feature maps)
    feature_maps = all_gabor + strong
    
    if debug_mode:
        return feature_maps, debug_data
        
    return feature_maps
