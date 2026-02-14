import matplotlib.pyplot as plt
import numpy as np
import cv2

def save_figure_grid(images, titles, filename, grid_shape, figsize=None):
    """
    Saves a grid of images with titles.
    
    Args:
        images: List of numpy arrays (BGR or Grayscale).
        titles: List of strings.
        filename: Output filename.
        grid_shape: Tuple (rows, cols).
        figsize: Optional tuple (width, height).
    """
    rows, cols = grid_shape
    if figsize is None:
        figsize = (5 * cols, 5 * rows)
        
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for i, ax in enumerate(axes):
        if i < len(images):
            img = images[i]
            if len(img.shape) == 2: # Grayscale
                ax.imshow(img, cmap='gray')
            else: # BGR
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if i < len(titles):
                ax.set_title(titles[i], fontsize=12, y=-0.15)
                
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
