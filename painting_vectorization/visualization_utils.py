"""
Shared visualization utilities for consistent scaling across pipeline steps
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def get_consistent_figure_layout(figsize=(18, 12)):
    """
    Create consistent figure layout for all pipeline visualizations
    
    Returns:
        tuple: (fig, axes) with standardized layout
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    return fig, axes


def normalize_coordinates_to_display(coords, original_shape, target_size=512):
    """
    Normalize coordinates to consistent display size
    
    Args:
        coords: Array of coordinates (N, 2) in (y, x) or (x, y) format
        original_shape: Original image shape (height, width)
        target_size: Target display size (square)
    
    Returns:
        Normalized coordinates scaled to target_size
    """
    if len(coords) == 0:
        return coords
    
    coords = np.array(coords)
    height, width = original_shape[:2]
    
    # Scale to target size while maintaining aspect ratio
    scale_factor = target_size / max(height, width)
    
    # Apply scaling
    scaled_coords = coords * scale_factor
    
    return scaled_coords


def set_consistent_axis_properties(ax, title, target_size=512):
    """
    Apply consistent axis properties across all visualizations
    
    Args:
        ax: Matplotlib axis
        title: Title for the subplot
        target_size: Display size for consistent scaling
    """
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    ax.set_xlim(0, target_size)
    ax.set_ylim(0, target_size)
    ax.set_aspect('equal')
    # Invert y-axis to match image coordinates
    ax.invert_yaxis()


def create_consistent_mask_overlay(mask, original_shape, target_size=512, color=[0, 1, 0, 0.5]):
    """
    Create consistent mask overlay for visualization
    
    Args:
        mask: Boolean mask array
        original_shape: Original image shape
        target_size: Target display size
        color: RGBA color for overlay
    
    Returns:
        Resized mask overlay for consistent display
    """
    import cv2
    
    # Resize mask to target size
    height, width = original_shape[:2]
    scale_factor = target_size / max(height, width)
    
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    # Resize mask
    resized_mask = cv2.resize(mask.astype(np.uint8), (new_width, new_height), 
                             interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # Create RGBA overlay
    mask_overlay = np.zeros((target_size, target_size, 4))
    
    # Center the resized mask
    y_offset = (target_size - new_height) // 2
    x_offset = (target_size - new_width) // 2
    
    mask_overlay[y_offset:y_offset+new_height, x_offset:x_offset+new_width][resized_mask] = color
    
    return mask_overlay


def create_consistent_background_image(image, target_size=512):
    """
    Create consistent background image for visualization
    
    Args:
        image: Input image array
        target_size: Target display size
    
    Returns:
        Resized and padded image for consistent display
    """
    import cv2
    
    height, width = image.shape[:2]
    scale_factor = target_size / max(height, width)
    
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    # Resize image
    if len(image.shape) == 3:
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    else:
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    if len(image.shape) == 3:
        padded_image = np.ones((target_size, target_size, image.shape[2]), dtype=image.dtype)
        if image.dtype == np.float32 or image.dtype == np.float64:
            padded_image *= 1.0  # White background for float images
        else:
            padded_image *= 255  # White background for uint8 images
    else:
        padded_image = np.ones((target_size, target_size), dtype=image.dtype)
        if image.dtype == np.float32 or image.dtype == np.float64:
            padded_image *= 1.0
        else:
            padded_image *= 255
    
    # Center the resized image
    y_offset = (target_size - new_height) // 2
    x_offset = (target_size - new_width) // 2
    
    if len(image.shape) == 3:
        padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
    else:
        padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
    
    return padded_image


def save_visualization_with_timestamp(fig, output_dir, filename_prefix, dpi=150):
    """
    Save visualization with consistent timestamp and path handling
    
    Args:
        fig: Matplotlib figure
        output_dir: Output directory
        filename_prefix: Prefix for filename (e.g., 'segmentation_results')
        dpi: DPI for saved image
    
    Returns:
        str: Path to saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine output path with fallbacks
    results_dir = Path(output_dir)
    if not results_dir.exists():
        results_dir = Path(f"results/{output_dir.split('/')[-1]}")
    if not results_dir.exists():
        results_dir = Path(f"../{output_dir}")
    if not results_dir.exists():
        results_dir = Path(".")  # Fallback to current directory
    
    # Create directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = results_dir / f'{filename_prefix}_{timestamp}.png'
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"   💾 Saved visualization: {output_path}")
    return str(output_path)


def hide_unused_subplots(axes, num_used):
    """
    Hide unused subplots in a consistent manner
    
    Args:
        axes: Flattened axes array
        num_used: Number of used subplots
    """
    for i in range(num_used, len(axes)):
        axes[i].axis('off')


# Standard visualization parameters
STANDARD_TARGET_SIZE = 512
STANDARD_FIGSIZE = (18, 12)
STANDARD_DPI = 150

# Standard color schemes
NEON_GREEN_OVERLAY = [0, 1, 0, 0.5]
SEMANTIC_COLORS = {
    'boundary': 'red',
    'internal': 'blue', 
    'detail': 'green'
}
SEMANTIC_ALPHAS = {
    'boundary': 0.8,
    'internal': 0.6,
    'detail': 0.4
}
SEMANTIC_WIDTHS = {
    'boundary': 3,
    'internal': 2,
    'detail': 1
}
