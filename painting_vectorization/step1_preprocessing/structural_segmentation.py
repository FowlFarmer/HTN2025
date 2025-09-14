"""
Structural segmentation for tactile drawing - focuses on major shapes and regions
rather than color similarity to minimize pen lifts and create smooth, continuous strokes.
"""
import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from scipy import ndimage
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def extract_major_structures(image, method='contour_hierarchy', **kwargs):
    """
    Extract major structural elements instead of color-based segments
    
    Args:
        image: Input RGB image (uint8)
        method: Segmentation method ('contour_hierarchy', 'watershed_regions', 'gradient_regions', 'texture_regions')
        **kwargs: Method-specific parameters
    
    Returns:
        numpy.ndarray: Segmentation mask with major structural regions
    """
    if method == 'contour_hierarchy':
        return segment_by_contour_hierarchy(image, **kwargs)
    elif method == 'watershed_regions':
        return segment_by_watershed(image, **kwargs)
    elif method == 'gradient_regions':
        return segment_by_gradient_regions(image, **kwargs)
    elif method == 'texture_regions':
        return segment_by_texture_regions(image, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def segment_by_contour_hierarchy(image, min_area_ratio=0.02, max_regions=6, smoothing=5):
    """
    Segment based on contour hierarchy - focuses on major shapes and objects
    
    Args:
        image: Input RGB image
        min_area_ratio: Minimum area as ratio of total image (0.01 = 1%)
        max_regions: Maximum number of regions to extract
        smoothing: Gaussian blur kernel size for smoothing
    
    Returns:
        numpy.ndarray: Segmentation mask
    """
    h, w = image.shape[:2]
    total_area = h * w
    
    # Convert to grayscale and apply smoothing
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if smoothing > 0:
        gray = cv2.GaussianBlur(gray, (smoothing, smoothing), 0)
    
    # Multi-scale edge detection for better structure detection
    edges = detect_structural_edges(gray)
    
    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and hierarchy
    major_contours = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > total_area * min_area_ratio:
            # Check if this is a major structural element (not nested too deep)
            if hierarchy[0][i][3] == -1 or hierarchy[0][i][2] == -1:  # Top level or has children
                major_contours.append((contour, area))
    
    # Sort by area (largest first) and take top regions
    major_contours.sort(key=lambda x: x[1], reverse=True)
    major_contours = major_contours[:max_regions]
    
    # Create segmentation mask
    segmentation = np.zeros((h, w), dtype=np.uint8)
    
    for i, (contour, _) in enumerate(major_contours):
        # Fill contour region
        cv2.fillPoly(segmentation, [contour], i + 1)
    
    # Post-process to ensure no overlaps and fill gaps
    segmentation = refine_structural_segments(segmentation, image)
    
    return segmentation


def detect_structural_edges(gray_image, scale_factors=[1.0, 0.5, 2.0]):
    """
    Multi-scale edge detection to capture both fine and coarse structures
    
    Args:
        gray_image: Grayscale image
        scale_factors: Different scales for edge detection
    
    Returns:
        numpy.ndarray: Combined edge map
    """
    h, w = gray_image.shape
    combined_edges = np.zeros((h, w), dtype=np.float32)
    
    for scale in scale_factors:
        # Resize for different scales
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(gray_image, (new_w, new_h))
        else:
            scaled = gray_image.copy()
        
        # Adaptive threshold for structure detection
        adaptive_thresh = cv2.adaptiveThreshold(
            scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to connect nearby structures
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Resize back if needed
        if scale != 1.0:
            adaptive_thresh = cv2.resize(adaptive_thresh, (w, h))
        
        # Combine with weight based on scale
        weight = 1.0 if scale == 1.0 else 0.5
        combined_edges += (adaptive_thresh.astype(np.float32) / 255.0) * weight
    
    # Normalize and convert to binary
    combined_edges = (combined_edges / np.max(combined_edges) * 255).astype(np.uint8)
    
    # Final morphological operations to create clean structural boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
    
    return combined_edges


def segment_by_watershed(image, num_markers=6, compactness=0.1):
    """
    Watershed segmentation focusing on major regions
    
    Args:
        image: Input RGB image
        num_markers: Number of watershed markers (regions)
        compactness: Compactness parameter for watershed
    
    Returns:
        numpy.ndarray: Segmentation mask
    """
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_maxima
    from scipy import ndimage as ndi
    
    # Convert to LAB for better perceptual segmentation
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Create distance transform from edges
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Distance transform
    distance = ndi.distance_transform_edt(~edges)
    
    # Find peaks as markers
    coords = peak_local_maxima(distance, min_distance=min(image.shape[:2]) // 10, 
                              threshold_abs=0.3 * distance.max(), num_peaks=num_markers)
    
    # Create markers
    markers = np.zeros(distance.shape, dtype=int)
    for i, coord in enumerate(zip(coords[0], coords[1])):
        markers[coord] = i + 1
    
    # Apply watershed
    labels = watershed(-distance, markers, mask=distance > 0, compactness=compactness)
    
    return labels.astype(np.uint8)


def segment_by_gradient_regions(image, gradient_threshold=30, min_region_size=0.03):
    """
    Segment based on gradient magnitude - separates regions with different textures/patterns
    
    Args:
        image: Input RGB image
        gradient_threshold: Threshold for gradient magnitude
        min_region_size: Minimum region size as ratio of image
    
    Returns:
        numpy.ndarray: Segmentation mask
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calculate gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Threshold gradient to find region boundaries
    boundaries = gradient_mag > gradient_threshold
    
    # Fill regions between boundaries
    filled_regions = ~boundaries
    
    # Label connected components
    num_labels, labels = cv2.connectedComponents(filled_regions.astype(np.uint8))
    
    # Filter by size
    h, w = image.shape[:2]
    min_area = h * w * min_region_size
    
    filtered_labels = np.zeros_like(labels)
    label_counter = 1
    
    for label in range(1, num_labels):
        if np.sum(labels == label) > min_area:
            filtered_labels[labels == label] = label_counter
            label_counter += 1
    
    return filtered_labels.astype(np.uint8)


def segment_by_texture_regions(image, window_size=15, n_clusters=6):
    """
    Segment based on texture patterns using Local Binary Patterns
    
    Args:
        image: Input RGB image
        window_size: Size of texture analysis window
        n_clusters: Number of texture clusters
    
    Returns:
        numpy.ndarray: Segmentation mask
    """
    from skimage.feature import local_binary_pattern
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calculate LBP
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Calculate texture features in sliding windows
    h, w = gray.shape
    texture_features = []
    coordinates = []
    
    step = window_size // 2
    for y in range(step, h - step, step):
        for x in range(step, w - step, step):
            # Extract window
            window = lbp[y-step:y+step, x-step:x+step]
            
            # Calculate histogram of LBP
            hist, _ = np.histogram(window.flatten(), bins=n_points + 2, range=(0, n_points + 2))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)  # Normalize
            
            texture_features.append(hist)
            coordinates.append((y, x))
    
    # Cluster texture features
    texture_features = np.array(texture_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    texture_labels = kmeans.fit_predict(texture_features)
    
    # Create segmentation map
    segmentation = np.zeros((h, w), dtype=np.uint8)
    
    for (y, x), label in zip(coordinates, texture_labels):
        segmentation[y-step:y+step, x-step:x+step] = label + 1
    
    # Fill gaps using nearest neighbor
    segmentation = fill_segmentation_gaps(segmentation)
    
    return segmentation


def refine_structural_segments(segmentation, original_image):
    """
    Refine segmentation to ensure smooth, continuous regions
    
    Args:
        segmentation: Initial segmentation mask
        original_image: Original RGB image
    
    Returns:
        numpy.ndarray: Refined segmentation
    """
    # Fill small holes within regions
    refined = segmentation.copy()
    
    for label in np.unique(segmentation):
        if label == 0:
            continue
        
        # Extract region
        region_mask = (segmentation == label)
        
        # Fill holes
        filled = ndimage.binary_fill_holes(region_mask)
        
        # Smooth boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        filled = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)
        
        refined[filled > 0] = label
    
    return refined


def fill_segmentation_gaps(segmentation):
    """
    Fill gaps in segmentation using nearest neighbor interpolation
    """
    # Find unlabeled pixels
    unlabeled = segmentation == 0
    
    if not np.any(unlabeled):
        return segmentation
    
    # Get coordinates of labeled and unlabeled pixels
    labeled_coords = np.column_stack(np.where(segmentation > 0))
    unlabeled_coords = np.column_stack(np.where(unlabeled))
    
    if len(labeled_coords) == 0:
        return segmentation
    
    # Find nearest labeled pixel for each unlabeled pixel
    distances = cdist(unlabeled_coords, labeled_coords)
    nearest_indices = np.argmin(distances, axis=1)
    
    # Assign labels
    result = segmentation.copy()
    for i, coord in enumerate(unlabeled_coords):
        nearest_coord = labeled_coords[nearest_indices[i]]
        result[coord[0], coord[1]] = segmentation[nearest_coord[0], nearest_coord[1]]
    
    return result


def create_tactile_friendly_segmentation(image, method='contour_hierarchy', **kwargs):
    """
    Create segmentation optimized for tactile drawing with minimal pen lifts
    
    Args:
        image: Input RGB image
        method: Segmentation method
        **kwargs: Method parameters
    
    Returns:
        tuple: (segmentation_mask, region_info)
    """
    # Get structural segmentation
    segmentation = extract_major_structures(image, method=method, **kwargs)
    
    # Analyze regions for tactile drawing optimization
    region_info = analyze_regions_for_tactile_drawing(segmentation, image)
    
    # Optimize region ordering for minimal pen lifts
    optimized_segmentation = optimize_for_pen_lifts(segmentation, region_info)
    
    return optimized_segmentation, region_info


def analyze_regions_for_tactile_drawing(segmentation, image):
    """
    Analyze regions to determine drawing order and characteristics
    
    Args:
        segmentation: Segmentation mask
        image: Original image
    
    Returns:
        dict: Region analysis information
    """
    unique_labels = np.unique(segmentation)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
    
    region_info = {}
    
    for label in unique_labels:
        mask = segmentation == label
        
        # Calculate region properties
        area = np.sum(mask)
        
        # Find bounding box
        coords = np.column_stack(np.where(mask))
        if len(coords) > 0:
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            
            # Calculate centroid
            centroid_y = np.mean(coords[:, 0])
            centroid_x = np.mean(coords[:, 1])
            
            # Calculate compactness (how circular/smooth the region is)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            else:
                compactness = 0
            
            region_info[label] = {
                'area': area,
                'centroid': (centroid_x, centroid_y),
                'bbox': (min_x, min_y, max_x, max_y),
                'compactness': compactness,
                'drawing_priority': calculate_drawing_priority(area, centroid_y, compactness)
            }
    
    return region_info


def calculate_drawing_priority(area, centroid_y, compactness):
    """
    Calculate drawing priority for tactile drawing (larger, higher, more compact regions first)
    """
    # Normalize factors
    area_score = area / 10000  # Larger regions first
    height_score = (1000 - centroid_y) / 1000  # Higher regions first (sky, etc.)
    compactness_score = compactness  # More compact regions are easier to draw
    
    # Weighted combination
    priority = 0.5 * area_score + 0.3 * height_score + 0.2 * compactness_score
    
    return priority


def optimize_for_pen_lifts(segmentation, region_info):
    """
    Reorder region labels to minimize pen lifts during drawing
    
    Args:
        segmentation: Original segmentation
        region_info: Region analysis information
    
    Returns:
        numpy.ndarray: Optimized segmentation with reordered labels
    """
    # Sort regions by drawing priority
    sorted_regions = sorted(region_info.items(), key=lambda x: x[1]['drawing_priority'], reverse=True)
    
    # Create new segmentation with optimized ordering
    optimized = np.zeros_like(segmentation)
    
    for new_label, (old_label, info) in enumerate(sorted_regions, 1):
        optimized[segmentation == old_label] = new_label
    
    return optimized


def visualize_structural_segmentation(image, segmentation, region_info, save_path=None):
    """
    Visualize structural segmentation results
    
    Args:
        image: Original image
        segmentation: Segmentation mask
        region_info: Region information
        save_path: Optional path to save visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Segmentation overlay
    axes[0, 1].imshow(image)
    
    # Create colored overlay
    colored_segmentation = np.zeros((*segmentation.shape, 3))
    colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(segmentation))))
    
    for i, label in enumerate(np.unique(segmentation)):
        if label > 0:
            colored_segmentation[segmentation == label] = colors[i][:3]
    
    axes[0, 1].imshow(colored_segmentation, alpha=0.6)
    axes[0, 1].set_title('Structural Segmentation Overlay', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Pure segmentation
    axes[1, 0].imshow(segmentation, cmap='tab10')
    axes[1, 0].set_title('Segmentation Regions', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Region analysis
    axes[1, 1].imshow(image, alpha=0.7)
    
    # Add region information
    for label, info in region_info.items():
        centroid = info['centroid']
        axes[1, 1].plot(centroid[0], centroid[1], 'ro', markersize=8)
        axes[1, 1].text(centroid[0], centroid[1], f'{label}', 
                        ha='center', va='center', fontweight='bold', color='white')
        
        # Add bounding box
        bbox = info['bbox']
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                           fill=False, edgecolor='red', linewidth=2)
        axes[1, 1].add_patch(rect)
    
    axes[1, 1].set_title('Region Analysis (Drawing Order)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.suptitle('Structural Segmentation for Tactile Drawing', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved structural segmentation: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test structural segmentation
    test_image_path = "../examples/tower.jpg"
    
    if Path(test_image_path).exists():
        # Load image
        image = cv2.imread(test_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Test different methods
        methods = [
            ('contour_hierarchy', {'min_area_ratio': 0.03, 'max_regions': 5}),
            ('watershed_regions', {'num_markers': 5}),
            ('gradient_regions', {'gradient_threshold': 25}),
        ]
        
        for method, params in methods:
            print(f"Testing {method}...")
            
            segmentation, region_info = create_tactile_friendly_segmentation(
                image, method=method, **params
            )
            
            print(f"Found {len(region_info)} major regions")
            for label, info in region_info.items():
                print(f"  Region {label}: area={info['area']}, priority={info['drawing_priority']:.3f}")
            
            # Visualize
            save_path = f"structural_segmentation_{method}.png"
            visualize_structural_segmentation(image, segmentation, region_info, save_path)
    
    else:
        print(f"Test image not found: {test_image_path}")
