import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def blend_colors_kmeans(image, n_colors=8):
    """
    Legacy K-means clustering (kept for compatibility)
    """
    # Reshape image to list of pixels
    data = image.reshape((-1, 3))
    data = np.float32(data)
    
    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8 and reshape to original image shape
    centers = np.uint8(centers)
    blended_data = centers[labels.flatten()]
    blended_image = blended_data.reshape(image.shape)
    
    return blended_image


def enhanced_color_clustering(image, method='adaptive_spatial', **kwargs):
    """
    Enhanced color clustering that preserves distinct regions
    
    Args:
        image: Input RGB image (uint8)
        method: Clustering method ('adaptive_spatial', 'region_aware', 'basic_kmeans')
        **kwargs: Method-specific parameters
    
    Returns:
        numpy.ndarray: Enhanced clustered image
    """
    try:
        from advanced_color_clustering import enhanced_color_clustering as advanced_clustering
        return advanced_clustering(image, method=method, **kwargs)
    except ImportError:
        # Fallback to basic K-means if advanced clustering not available
        print("   ⚠️  Advanced clustering not available, using basic K-means")
        return blend_colors_kmeans(image, n_colors=kwargs.get('n_colors', 8))


def save_clustering_results(original_img, clustered_img, method, image_path):
    """
    Save K-means clustering results for visual inspection
    
    Args:
        original_img: Original resized image (before clustering)
        clustered_img: Image after color clustering
        method: Clustering method used
        image_path: Path to original image file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = Path("results/step1_preprocessing")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original (Resized)', fontsize=12)
    axes[0].axis('off')
    
    # Clustered image
    axes[1].imshow(clustered_img)
    axes[1].set_title(f'After {method.replace("_", " ").title()}', fontsize=12)
    axes[1].axis('off')
    
    # Color palette extraction
    unique_colors = extract_color_palette(clustered_img)
    axes[2] = show_color_palette(axes[2], unique_colors)
    axes[2].set_title(f'Color Palette ({len(unique_colors)} colors)', fontsize=12)
    
    # Overall title
    image_name = Path(image_path).stem
    plt.suptitle(f'Step 1: Color Clustering Results - {image_name}', fontsize=16)
    plt.tight_layout()
    
    # Save the comparison
    output_path = results_dir / f"clustering_results_{image_name}_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved clustering visualization: {output_path}")
    
    # Also save individual images for detailed inspection
    individual_dir = results_dir / "individual_results"
    individual_dir.mkdir(exist_ok=True)
    
    # Save original
    original_path = individual_dir / f"original_{image_name}_{timestamp}.png"
    plt.figure(figsize=(8, 6))
    plt.imshow(original_img)
    plt.title('Original (Resized)')
    plt.axis('off')
    plt.savefig(original_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save clustered
    clustered_path = individual_dir / f"clustered_{image_name}_{timestamp}.png"
    plt.figure(figsize=(8, 6))
    plt.imshow(clustered_img)
    plt.title(f'After {method.replace("_", " ").title()}')
    plt.axis('off')
    plt.savefig(clustered_path, dpi=150, bbox_inches='tight')
    plt.close()


def extract_color_palette(image):
    """
    Extract unique colors from clustered image
    
    Args:
        image: Clustered image (uint8)
    
    Returns:
        numpy.ndarray: Array of unique RGB colors
    """
    # Reshape to list of pixels
    pixels = image.reshape(-1, 3)
    
    # Find unique colors
    unique_colors = np.unique(pixels, axis=0)
    
    # Sort by brightness for better visualization
    brightness = np.sum(unique_colors, axis=1)
    sorted_indices = np.argsort(brightness)
    
    return unique_colors[sorted_indices]


def show_color_palette(ax, colors):
    """
    Display color palette as colored rectangles
    
    Args:
        ax: Matplotlib axis
        colors: Array of RGB colors
    
    Returns:
        Modified axis
    """
    n_colors = len(colors)
    
    if n_colors == 0:
        ax.text(0.5, 0.5, 'No colors found', ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # Create color swatches
        for i, color in enumerate(colors):
            rect_height = 1.0 / n_colors
            rect = plt.Rectangle((0, i * rect_height), 1, rect_height, 
                               facecolor=color/255.0, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # Add color values as text
            text_color = 'white' if np.sum(color) < 384 else 'black'  # Use white text on dark colors
            ax.text(0.5, i * rect_height + rect_height/2, 
                   f'RGB({color[0]}, {color[1]}, {color[2]})',
                   ha='center', va='center', fontsize=8, color=text_color, weight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    ax.axis('off')
    return ax


def apply_structural_segmentation(image, image_path, **params):
    """
    Apply structural segmentation and save results for tactile drawing
    
    Args:
        image: Input RGB image (uint8)
        image_path: Original image path for saving results
        **params: Segmentation parameters
    
    Returns:
        numpy.ndarray: Processed image optimized for tactile drawing
    """
    try:
        from structural_segmentation import create_tactile_friendly_segmentation, visualize_structural_segmentation
        
        # Extract method from params
        method = params.pop('method', 'contour_hierarchy')
        
        # Create structural segmentation
        segmentation, region_info = create_tactile_friendly_segmentation(
            image, method=method, **params
        )
        
        # Save structural segmentation results
        save_structural_results(image, segmentation, region_info, image_path, method)
        
        # Create simplified image based on structural regions
        simplified_image = create_simplified_structural_image(image, segmentation, region_info)
        
        return simplified_image
        
    except ImportError:
        print("   ⚠️  Structural segmentation not available, using basic smoothing")
        # Fallback to simple smoothing
        return cv2.bilateralFilter(image, 9, 75, 75)


def save_structural_results(original_img, segmentation, region_info, image_path, method):
    """
    Save structural segmentation results for inspection
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = Path(image_path).stem
    
    # Create results directory
    results_dir = Path("results/step1_preprocessing")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed visualization
    save_path = results_dir / f"structural_segmentation_{image_name}_{timestamp}.png"
    
    try:
        from structural_segmentation import visualize_structural_segmentation
        visualize_structural_segmentation(original_img, segmentation, region_info, save_path)
    except ImportError:
        # Simple fallback visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(segmentation, cmap='tab10')
        axes[1].set_title(f'Structural Regions ({method})')
        axes[1].axis('off')
        
        plt.suptitle(f'Structural Segmentation - {image_name}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"   💾 Saved structural segmentation: {save_path}")
    
    # Print region analysis
    print(f"   🏗️  Found {len(region_info)} major structural regions:")
    for label, info in sorted(region_info.items(), key=lambda x: x[1]['drawing_priority'], reverse=True):
        area_pct = info['area'] / (original_img.shape[0] * original_img.shape[1]) * 100
        print(f"      Region {label}: {area_pct:.1f}% area, priority={info['drawing_priority']:.3f}")


def create_simplified_structural_image(image, segmentation, region_info):
    """
    Create simplified image based on structural regions for better SAM segmentation
    
    Args:
        image: Original image
        segmentation: Structural segmentation mask
        region_info: Region analysis information
    
    Returns:
        numpy.ndarray: Simplified image with clear structural boundaries
    """
    h, w = image.shape[:2]
    simplified = np.zeros_like(image)
    
    # Assign average color to each structural region
    for label in np.unique(segmentation):
        if label == 0:
            continue
            
        mask = segmentation == label
        if np.any(mask):
            # Calculate average color for this region
            region_pixels = image[mask]
            avg_color = np.mean(region_pixels, axis=0).astype(np.uint8)
            
            # Apply average color to entire region
            simplified[mask] = avg_color
    
    # Enhance boundaries between regions
    simplified = enhance_structural_boundaries(simplified, segmentation)
    
    return simplified


def enhance_structural_boundaries(image, segmentation):
    """
    Enhance boundaries between structural regions for clearer segmentation
    """
    # Find boundaries between regions
    kernel = np.ones((3, 3), np.uint8)
    boundaries = cv2.morphologyEx(segmentation.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    
    # Darken boundary pixels to create clear separation
    enhanced = image.copy()
    boundary_mask = boundaries > 0
    enhanced[boundary_mask] = enhanced[boundary_mask] * 0.3  # Darken boundaries
    
    return enhanced


def apply_outline_focused_segmentation(image, image_path, **params):
    """
    Apply outline-focused segmentation for contour tracing
    
    Args:
        image: Input RGB image (uint8)
        image_path: Original image path for saving results
        **params: Segmentation parameters
    
    Returns:
        numpy.ndarray: Processed image optimized for outline tracing
    """
    try:
        from .outline_focused_segmentation import create_outline_focused_segmentation, visualize_outline_segmentation, create_outline_image
        
        # Create outline-focused segmentation
        contours, contour_info, outline_count = create_outline_focused_segmentation(
            image, **params
        )
        
        # Save outline segmentation results
        save_outline_results(image, contours, contour_info, outline_count, image_path)
        
        # Create simplified image with only outlines
        outline_image = create_outline_image(image, contours, contour_info)
        
        return outline_image
        
    except ImportError:
        print("   ⚠️  Outline-focused segmentation not available, using edge detection")
        # Fallback to simple edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Create 3-channel edge image
        edge_image = np.zeros_like(image)
        edge_image[:, :, 0] = edges
        edge_image[:, :, 1] = edges  
        edge_image[:, :, 2] = edges
        
        return edge_image


def save_outline_results(original_img, contours, contour_info, outline_count, image_path):
    """
    Save outline-focused segmentation results for inspection
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = Path(image_path).stem
    
    # Create results directory
    results_dir = Path("results/step1_preprocessing")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed visualization
    save_path = results_dir / f"outline_segmentation_{image_name}_{timestamp}.png"
    
    try:
        from .outline_focused_segmentation import visualize_outline_segmentation
        visualize_outline_segmentation(original_img, contours, contour_info, save_path)
    except ImportError:
        # Simple fallback visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Show detected outlines
        axes[1].imshow(original_img, alpha=0.7)
        colors = plt.cm.Set1(np.linspace(0, 1, len(contours)))
        
        for i, contour_data in enumerate(contours):
            contour = contour_data['contour']
            contour_points = contour.reshape(-1, 2)
            axes[1].plot(contour_points[:, 0], contour_points[:, 1], 
                        color=colors[i], linewidth=3, alpha=0.8)
        
        axes[1].set_title(f'Outlines ({len(contours)} pen lifts)')
        axes[1].axis('off')
        
        plt.suptitle(f'Outline-Focused Segmentation - {image_name}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"   💾 Saved outline segmentation: {save_path}")
    
    # Print outline analysis with pen lift count
    print(f"   ✏️  Found {len(contours)} object outlines = {outline_count} pen lifts for tracing")
    print(f"   🖊️  Pure outline drawing: No filled regions, only meaningful object shapes")
    
    for obj_id, info in sorted(contour_info.items(), key=lambda x: x[1]['drawing_priority'], reverse=True):
        area_pct = info['area'] / (original_img.shape[0] * original_img.shape[1]) * 100
        method = info['detection_method']
        perimeter = info['stroke_plan']['perimeter']
        print(f"      Outline {obj_id}: {area_pct:.1f}% area, {perimeter:.0f}px perimeter, {method}, priority={info['drawing_priority']:.3f}")


def apply_object_focused_segmentation(image, image_path, **params):
    """
    Apply object-focused segmentation for minimal pen lifts
    
    Args:
        image: Input RGB image (uint8)
        image_path: Original image path for saving results
        **params: Segmentation parameters
    
    Returns:
        numpy.ndarray: Processed image optimized for minimal pen lifts
    """
    try:
        from object_focused_segmentation import create_tactile_object_segmentation, visualize_object_segmentation
        
        # Create object-focused segmentation
        object_masks, object_info, stroke_count = create_tactile_object_segmentation(
            image, **params
        )
        
        # Save object segmentation results
        save_object_results(image, object_masks, object_info, stroke_count, image_path)
        
        # Create simplified image based on objects
        simplified_image = create_simplified_object_image(image, object_masks, object_info)
        
        return simplified_image
        
    except ImportError:
        print("   ⚠️  Object-focused segmentation not available, using basic smoothing")
        # Fallback to simple smoothing
        return cv2.bilateralFilter(image, 9, 75, 75)


def save_object_results(original_img, object_masks, object_info, stroke_count, image_path):
    """
    Save object-focused segmentation results for inspection
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = Path(image_path).stem
    
    # Create results directory
    results_dir = Path("results/step1_preprocessing")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed visualization
    save_path = results_dir / f"object_segmentation_{image_name}_{timestamp}.png"
    
    try:
        from object_focused_segmentation import visualize_object_segmentation
        visualize_object_segmentation(original_img, object_masks, object_info, save_path)
    except ImportError:
        # Simple fallback visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Show object masks
        axes[1].imshow(original_img, alpha=0.7)
        colors = plt.cm.Set1(np.linspace(0, 1, len(object_masks)))
        for i, mask in enumerate(object_masks):
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = [*colors[i][:3], 0.6]
            axes[1].imshow(colored_mask)
        
        axes[1].set_title(f'Objects ({len(object_masks)} pen lifts)')
        axes[1].axis('off')
        
        plt.suptitle(f'Object-Focused Segmentation - {image_name}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"   💾 Saved object segmentation: {save_path}")
    
    # Print object analysis with pen lift count
    print(f"   🎯 Found {len(object_masks)} objects = {stroke_count} pen lifts total")
    print(f"   🖊️  Pen lift reduction: ~{100 - (stroke_count/20)*100:.0f}% fewer lifts than typical")
    
    for obj_id, info in sorted(object_info.items(), key=lambda x: x[1]['drawing_priority'], reverse=True):
        area_pct = info['area'] / (original_img.shape[0] * original_img.shape[1]) * 100
        stroke_type = info['stroke_plan']['type']
        print(f"      Object {obj_id}: {area_pct:.1f}% area, {stroke_type}, priority={info['drawing_priority']:.3f}")


def create_simplified_object_image(image, object_masks, object_info):
    """
    Create simplified image based on individual objects for better SAM segmentation
    
    Args:
        image: Original image
        object_masks: List of object masks
        object_info: Object analysis information
    
    Returns:
        numpy.ndarray: Simplified image with clear object boundaries
    """
    h, w = image.shape[:2]
    simplified = np.zeros_like(image)
    
    # Assign distinct colors to each object
    object_colors = generate_distinct_colors(len(object_masks))
    
    for i, mask in enumerate(object_masks):
        if np.any(mask):
            # Use distinct color for each object to ensure SAM separates them
            simplified[mask] = object_colors[i]
    
    # Fill background with neutral color
    background_mask = np.ones((h, w), dtype=bool)
    for mask in object_masks:
        background_mask &= ~mask
    
    if np.any(background_mask):
        # Use average background color
        background_pixels = image[background_mask]
        if len(background_pixels) > 0:
            avg_bg_color = np.mean(background_pixels, axis=0).astype(np.uint8)
            simplified[background_mask] = avg_bg_color
    
    # Enhance boundaries between objects
    simplified = enhance_object_boundaries(simplified, object_masks)
    
    return simplified


def generate_distinct_colors(n_colors):
    """
    Generate visually distinct colors for objects
    """
    if n_colors <= 0:
        return []
    
    # Use HSV to generate evenly spaced hues
    colors = []
    for i in range(n_colors):
        # OpenCV HSV: H=0-179, S=0-255, V=0-255
        hue = int((i * 179 / n_colors) % 179)  # Scale to 0-179 for OpenCV
        # Convert HSV to RGB
        hsv_color = np.array([[[hue, 255, 200]]], dtype=np.uint8)
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0, 0]
        colors.append(rgb_color)
    
    return colors


def enhance_object_boundaries(image, object_masks):
    """
    Enhance boundaries between objects for clearer segmentation
    """
    h, w = image.shape[:2]
    enhanced = image.copy()
    
    # Create boundary mask between objects
    boundary_mask = np.zeros((h, w), dtype=bool)
    
    for i, mask1 in enumerate(object_masks):
        for j, mask2 in enumerate(object_masks[i+1:], i+1):
            # Find boundary between these two objects
            kernel = np.ones((3, 3), np.uint8)
            
            # Dilate both masks
            dilated1 = cv2.dilate(mask1.astype(np.uint8), kernel, iterations=1)
            dilated2 = cv2.dilate(mask2.astype(np.uint8), kernel, iterations=1)
            
            # Find overlap (boundary region)
            overlap = np.logical_and(dilated1, dilated2)
            boundary_mask |= overlap
    
    # Darken boundary pixels to create clear separation
    if np.any(boundary_mask):
        enhanced[boundary_mask] = enhanced[boundary_mask] * 0.2  # Very dark boundaries
    
    return enhanced


def preprocess_image(image_path, segmentation_method='none', segmentation_params=None):
    """
    Image preprocessing pipeline for painting vectorization.

    Args:
        image_path (str): Path to the input image
        segmentation_method (str): Segmentation method ('outline_focused', 'object_focused', 'structural', 'color_clustering')
        segmentation_params (dict): Parameters for the segmentation method

    Returns:
        tuple: (processed_image, scale_factor, original_dimensions)
            processed_image: numpy.ndarray with shape (height, width, 3),
                           dtype float32, values in range [0.0, 1.0], RGB color order
            scale_factor: float, resize ratio applied to original image
            original_dimensions: tuple (original_height, original_width) in pixels
    """

    # 1.1 Image Loading and Format Conversion
    img = cv2.imread(image_path)[:, :, ::-1]  # BGR -> RGB
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Handle RGBA to RGB conversion
    if img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]  # Remove alpha channel

    # Store original dimensions
    original_h, original_w = img.shape[:2]

    # 1.2 Resizing Strategy
    h, w = img.shape[:2]
    scale = 1024 / max(h, w) if max(h, w) > 1024 else 1.0
    img_small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # 1.3 Skip complex preprocessing - use original image for clean SAM segmentation
    if segmentation_method == 'none':
        print("   🎯 Using original image for clean SAM segmentation (no preprocessing)")
        img_processed = img_small
    elif segmentation_method == 'color_clustering':
        # Only keep color clustering as an option if specifically requested
        if segmentation_params is None:
            segmentation_params = {}
        default_params = {'spatial_weight': 0.3, 'preserve_boundaries': True}
        final_params = {**default_params, **segmentation_params}
        
        print("   🎨 Applying color clustering...")
        img_processed = enhanced_color_clustering(
            img_small, 
            method='adaptive_spatial',
            **final_params
        )
        save_clustering_results(img_small, img_processed, 'color_clustering', image_path)
    else:
        print(f"   🎯 Using original image (segmentation_method='{segmentation_method}' not recognized)")
        img_processed = img_small
    
    # Use processed image for further steps
    img_blended = img_processed

    # 1.4 Contrast Enhancement
    lab = cv2.cvtColor(img_blended, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img_enhanced = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)

    # 1.5 Data Type Conversion
    img_processed = img_enhanced.astype(np.float32) / 255.0

    return img_processed, scale, (original_h, original_w)

if __name__ == "__main__":
    # Process the monet.jpeg image
    processed_img, scale_factor, original_dims = preprocess_image("monet.jpeg")

    print(f"Original dimensions: {original_dims}")
    print(f"Scale factor: {scale_factor}")
    print(f"Processed image shape: {processed_img.shape}")
    print(f"Processed image dtype: {processed_img.dtype}")
    print(f"Processed image range: [{processed_img.min():.3f}, {processed_img.max():.3f}]")