
import cv2
import numpy as np
from skimage import exposure, color

def get_fov_bounding_box(fov_mask):
    """
    Finds the bounding box of the FOV mask.
    """
    rows = np.any(fov_mask, axis=1)
    cols = np.any(fov_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_to_fov(image, fov_mask):
    """
    Crops the image and mask to the bounding box of the FOV.
    """
    rmin, rmax, cmin, cmax = get_fov_bounding_box(fov_mask)
    # Add 1 to max indices for slicing
    image_cropped = image[rmin:rmax+1, cmin:cmax+1]
    mask_cropped = fov_mask[rmin:rmax+1, cmin:cmax+1]
    return image_cropped, mask_cropped

def apply_clahe(channel, clip_limit=0.01):
    """
    Applies CLAHE enhancement to a single channel.
    Target function parameters: Clip Limit = 0.01, Tile Grid Size = (8, 8).
    Dimensions are preserved.
    """
    # skimage expects float in [0, 1] or preserves range. 
    # normalize to 0-1 for stability with skimage
    
    # Ensure standard normalization
    channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
    
    # skimage's equalize_adapthist default kernel_size results in 1/8 of image if not specified? 
    # No, it uses kernel_size. But prompt says "Tile Grid Size = (8, 8)" which is OpenCV terminology.
    # IN skimage, we use 'kernel_size'. If we want 8x8 tiles (meaning grid), the kernel size is ImageSize / 8.
    # However, 'nbins=256' is also specified.
    
    # Let's try to infer if 'Tile Grid Size = (8, 8)' means the Grid is 8x8 (OpenCV default) or the kernel size is 8x8 pixels.
    # OpenCV createCLAHE(tileGridSize=(8,8)) means the image is divided into 8x8 blocks. This is the standard interpretation.
    # The prompt explicitly asks for "Clip Limit = 0.01". This is an skimage parameter style (0-1). OpenCV uses larger numbers (default 40.0?).
    # So we have a mixed requirement. I will use skimage as it fits the "0.01" better and is more robust for "scientific" pipelines.
    # For 'Tile Grid Size = (8, 8)' in skimage, we need to compute kernel_size based on image shape.
    
    h, w = channel.shape
    kernel_size = (h // 8, w // 8) # Equivalent to 8x8 grid
    
    enhanced = exposure.equalize_adapthist(channel_norm, kernel_size=kernel_size, clip_limit=clip_limit, nbins=256)
    
    # Return as 0-255 uint8
    return (enhanced * 255).astype(np.uint8)

def preprocess_image(image, fov_mask, debug_mode=False):
    """
    1. Crop Image to FOV
    2. Extract G, Y (YCbCr), L (CIELab)
    3. Apply CLAHE
    
    If debug_mode=True, returns:
        crop_img, crop_mask, g_clahe, y_clahe, l_clahe, debug_data
        where debug_data = {'fig3': [g_img, y_img, l_img], 'fig4': [g_clahe, y_clahe, l_clahe]}
    """
    # Ensure binary mask
    if fov_mask.max() > 1:
        fov_mask = fov_mask > 128
        
    # 1. Crop
    crop_img, crop_mask = crop_to_fov(image, fov_mask)
    
    # 2. Channel Selection
    # G proper from RGB
    g_img = crop_img[:, :, 1]
    
    # Y from YCbCr (Using OpenCV)
    # Note: OpenCV RGB2YCrCb returns Y, Cr, Cb. Y is channel 0.
    y_img_full = cv2.cvtColor(crop_img, cv2.COLOR_RGB2YCrCb)
    y_img = y_img_full[:, :, 0]
    
    # L from CIELab (Using OpenCV)
    # Note: OpenCV RGB2Lab returns L, a, b. L is channel 0.
    l_img_full = cv2.cvtColor(crop_img, cv2.COLOR_RGB2Lab)
    l_img = l_img_full[:, :, 0]
    
    # 3. CLAHE
    # Apply to each channel
    g_clahe = apply_clahe(g_img)
    y_clahe = apply_clahe(y_img)
    l_clahe = apply_clahe(l_img)
    
    if debug_mode:
        debug_data = {
            'fig3': [g_img, y_img, l_img],
            'fig4': [g_clahe, y_clahe, l_clahe]
        }
        return crop_img, crop_mask, g_clahe, y_clahe, l_clahe, debug_data
    
    return crop_img, crop_mask, g_clahe, y_clahe, l_clahe
