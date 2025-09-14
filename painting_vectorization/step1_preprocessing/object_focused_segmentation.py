"""
Object-focused segmentation for tactile drawing - identifies individual objects
and creates single continuous strokes for each, minimizing pen lifts to under 15 total.
"""
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy import ndimage
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def extract_individual_objects(image, max_objects=8, min_object_size=0.05):
    """
    Extract individual objects from image for single-stroke drawing
    
    Args:
        image: Input RGB image (uint8)
        max_objects: Maximum number of objects to extract
        min_object_size: Minimum object size as ratio of image area
    
    Returns:
        tuple: (object_masks, object_info)
    """
    # Multi-scale object detection
    objects = detect_objects_multiscale(image, min_object_size)
    
    # Filter and rank objects by drawing importance
    important_objects = rank_objects_for_drawing(objects, image, max_objects)
    
    # Create clean object masks
    object_masks = create_clean_object_masks(important_objects, image.shape[:2])
    
    # Analyze objects for stroke planning
    object_info = analyze_objects_for_strokes(object_masks, image)
    
    return object_masks, object_info


def detect_objects_multiscale(image, min_size_ratio=0.05):
    """
    Detect objects using multiple scales and methods
    
    Args:
        image: Input RGB image
        min_size_ratio: Minimum object size ratio
    
    Returns:
        list: Detected object contours and properties
    """
    h, w = image.shape[:2]
    min_area = h * w * min_size_ratio
    
    # Convert to different color spaces for better object detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    detected_objects = []
    
    # Method 1: Adaptive thresholding for clear objects
    objects_adaptive = detect_objects_adaptive_threshold(gray, min_area)
    detected_objects.extend(objects_adaptive)
    
    # Method 2: Edge-based object detection
    objects_edges = detect_objects_by_edges(gray, min_area)
    detected_objects.extend(objects_edges)
    
    # Method 3: Color-based object detection in HSV
    objects_color = detect_objects_by_color_regions(hsv, min_area)
    detected_objects.extend(objects_color)
    
    # Remove duplicates and merge overlapping objects
    unique_objects = merge_overlapping_objects(detected_objects, overlap_threshold=0.3)
    
    return unique_objects


def detect_objects_adaptive_threshold(gray, min_area):
    """
    Detect objects using adaptive thresholding - good for distinct objects
    """
    objects = []
    
    # Multiple adaptive threshold parameters
    threshold_params = [
        (11, 2), (15, 5), (21, 10)
    ]
    
    for block_size, C in threshold_params:
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
        )
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Calculate object properties
                obj_info = calculate_object_properties(contour, area, 'adaptive')
                objects.append(obj_info)
    
    return objects


def detect_objects_by_edges(gray, min_area):
    """
    Detect objects by finding closed edge regions
    """
    objects = []
    
    # Multi-scale edge detection
    for sigma in [1.0, 2.0, 3.0]:
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Fill enclosed regions
        filled = ndimage.binary_fill_holes(closed_edges)
        
        # Find contours of filled regions
        contours, _ = cv2.findContours(
            filled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                obj_info = calculate_object_properties(contour, area, 'edges')
                objects.append(obj_info)
    
    return objects


def detect_objects_by_color_regions(hsv, min_area):
    """
    Detect objects by grouping similar color regions
    """
    objects = []
    h, s, v = cv2.split(hsv)
    
    # Focus on saturation and value for object detection
    # High saturation often indicates distinct objects
    high_sat_mask = s > 50
    
    # Use watershed on saturation channel
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_maxima
    
    # Distance transform on high saturation areas
    distance = ndimage.distance_transform_edt(high_sat_mask)
    
    # Find peaks as object centers
    coords = peak_local_maxima(
        distance, 
        min_distance=min(hsv.shape[:2]) // 8,
        threshold_abs=0.3 * distance.max(),
        num_peaks=15
    )
    
    # Create markers
    markers = np.zeros(distance.shape, dtype=int)
    for i, coord in enumerate(zip(coords[0], coords[1])):
        markers[coord] = i + 1
    
    # Apply watershed
    if len(coords[0]) > 0:
        labels = watershed(-distance, markers, mask=high_sat_mask)
        
        # Extract objects from watershed regions
        for label in np.unique(labels):
            if label == 0:
                continue
                
            mask = labels == label
            area = np.sum(mask)
            
            if area > min_area:
                # Find contour of this region
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    obj_info = calculate_object_properties(largest_contour, area, 'color')
                    objects.append(obj_info)
    
    return objects


def calculate_object_properties(contour, area, detection_method):
    """
    Calculate properties of detected object for ranking and stroke planning
    """
    # Basic properties
    perimeter = cv2.arcLength(contour, True)
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w//2, y + h//2
    
    # Shape analysis
    aspect_ratio = w / h if h > 0 else 1
    extent = area / (w * h) if w * h > 0 else 0
    compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Convex hull for complexity analysis
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    return {
        'contour': contour,
        'area': area,
        'perimeter': perimeter,
        'centroid': (cx, cy),
        'bbox': (x, y, w, h),
        'aspect_ratio': aspect_ratio,
        'extent': extent,
        'compactness': compactness,
        'solidity': solidity,
        'detection_method': detection_method
    }


def merge_overlapping_objects(objects, overlap_threshold=0.3):
    """
    Merge objects that significantly overlap
    """
    if len(objects) <= 1:
        return objects
    
    # Calculate overlap matrix
    n_objects = len(objects)
    overlap_matrix = np.zeros((n_objects, n_objects))
    
    for i in range(n_objects):
        for j in range(i+1, n_objects):
            overlap = calculate_contour_overlap(objects[i]['contour'], objects[j]['contour'])
            overlap_matrix[i, j] = overlap
            overlap_matrix[j, i] = overlap
    
    # Group overlapping objects
    merged_objects = []
    used = set()
    
    for i in range(n_objects):
        if i in used:
            continue
            
        # Find all objects that overlap with object i
        overlapping = [i]
        for j in range(n_objects):
            if j != i and j not in used and overlap_matrix[i, j] > overlap_threshold:
                overlapping.append(j)
                used.add(j)
        
        # Merge overlapping objects
        if len(overlapping) == 1:
            merged_objects.append(objects[i])
        else:
            merged_obj = merge_object_group([objects[k] for k in overlapping])
            merged_objects.append(merged_obj)
        
        used.add(i)
    
    return merged_objects


def calculate_contour_overlap(contour1, contour2):
    """
    Calculate overlap ratio between two contours
    """
    # Create masks for both contours
    # Find bounding box that contains both
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    
    min_x = min(x1, x2)
    min_y = min(y1, y2)
    max_x = max(x1 + w1, x2 + w2)
    max_y = max(y1 + h1, y2 + h2)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Create masks
    mask1 = np.zeros((height, width), dtype=np.uint8)
    mask2 = np.zeros((height, width), dtype=np.uint8)
    
    # Adjust contours to new coordinate system
    adjusted_contour1 = contour1.copy()
    adjusted_contour1[:, :, 0] -= min_x
    adjusted_contour1[:, :, 1] -= min_y
    
    adjusted_contour2 = contour2.copy()
    adjusted_contour2[:, :, 0] -= min_x
    adjusted_contour2[:, :, 1] -= min_y
    
    # Fill contours
    cv2.fillPoly(mask1, [adjusted_contour1], 255)
    cv2.fillPoly(mask2, [adjusted_contour2], 255)
    
    # Calculate overlap
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    if np.sum(union) == 0:
        return 0
    
    return np.sum(intersection) / np.sum(union)


def merge_object_group(object_group):
    """
    Merge a group of overlapping objects into one
    """
    # Combine all contours
    all_points = []
    total_area = 0
    
    for obj in object_group:
        all_points.extend(obj['contour'].reshape(-1, 2))
        total_area += obj['area']
    
    # Create convex hull of all points
    all_points = np.array(all_points)
    hull = cv2.convexHull(all_points.reshape(-1, 1, 2))
    
    # Calculate merged properties
    merged_area = cv2.contourArea(hull)
    merged_obj = calculate_object_properties(hull, merged_area, 'merged')
    
    return merged_obj


def rank_objects_for_drawing(objects, image, max_objects):
    """
    Rank objects by importance for tactile drawing
    """
    if len(objects) <= max_objects:
        return objects
    
    h, w = image.shape[:2]
    image_area = h * w
    
    # Calculate drawing importance score for each object
    for obj in objects:
        score = calculate_drawing_importance(obj, image_area, h, w)
        obj['drawing_importance'] = score
    
    # Sort by importance and take top objects
    objects.sort(key=lambda x: x['drawing_importance'], reverse=True)
    
    return objects[:max_objects]


def calculate_drawing_importance(obj, image_area, image_height, image_width):
    """
    Calculate how important an object is for tactile drawing
    """
    # Size importance (larger objects are more important)
    size_score = obj['area'] / image_area
    
    # Position importance (objects in center and upper areas are more important)
    cx, cy = obj['centroid']
    center_distance = np.sqrt((cx - image_width/2)**2 + (cy - image_height/2)**2)
    max_distance = np.sqrt((image_width/2)**2 + (image_height/2)**2)
    position_score = 1 - (center_distance / max_distance)
    
    # Height importance (objects higher in image often more important)
    height_score = (image_height - cy) / image_height
    
    # Shape complexity (simpler shapes are better for tactile)
    complexity_score = obj['compactness'] * obj['solidity']
    
    # Combine scores with weights
    importance = (
        0.4 * size_score +           # Size is most important
        0.2 * position_score +       # Central objects important
        0.2 * height_score +         # Upper objects important  
        0.2 * complexity_score       # Simple shapes preferred
    )
    
    return importance


def create_clean_object_masks(objects, image_shape):
    """
    Create clean, non-overlapping masks for each object
    """
    h, w = image_shape
    object_masks = []
    
    # Create combined mask to avoid overlaps
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    
    for i, obj in enumerate(objects):
        # Create mask for this object
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [obj['contour']], 255)
        
        # Remove areas already assigned to other objects
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(combined_mask))
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        if np.sum(mask) > 0:  # Only add non-empty masks
            object_masks.append(mask > 0)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    return object_masks


def analyze_objects_for_strokes(object_masks, image):
    """
    Analyze each object for single-stroke drawing optimization
    """
    object_info = {}
    
    for i, mask in enumerate(object_masks):
        if not np.any(mask):
            continue
            
        # Find contour of cleaned mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            continue
            
        main_contour = max(contours, key=cv2.contourArea)
        
        # Analyze for stroke planning
        stroke_info = plan_single_stroke(main_contour, mask)
        
        # Calculate object properties
        area = np.sum(mask)
        centroid_coords = np.column_stack(np.where(mask))
        centroid = (np.mean(centroid_coords[:, 1]), np.mean(centroid_coords[:, 0]))
        
        object_info[i + 1] = {
            'area': area,
            'centroid': centroid,
            'contour': main_contour,
            'stroke_plan': stroke_info,
            'drawing_priority': calculate_single_stroke_priority(area, centroid, stroke_info)
        }
    
    return object_info


def plan_single_stroke(contour, mask):
    """
    Plan how to draw this object as a single continuous stroke
    """
    # Simplify contour for smoother drawing
    epsilon = 0.02 * cv2.arcLength(contour, True)
    simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
    
    # Determine if object should be drawn as outline or filled
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Simple objects -> outline, complex -> filled pattern
    compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    if compactness > 0.7 and area > 1000:  # Compact, large object
        stroke_type = 'filled_spiral'
        stroke_path = create_spiral_fill_path(mask)
    else:  # Draw as outline
        stroke_type = 'outline'
        stroke_path = simplified_contour.reshape(-1, 2)
    
    return {
        'type': stroke_type,
        'path': stroke_path,
        'length': len(stroke_path),
        'is_closed': stroke_type == 'outline'
    }


def create_spiral_fill_path(mask):
    """
    Create a spiral path to fill an object with a single continuous stroke
    """
    # Find the center of the mask
    coords = np.column_stack(np.where(mask))
    if len(coords) == 0:
        return np.array([])
    
    center_y, center_x = np.mean(coords, axis=0)
    
    # Create spiral path from center outward
    spiral_points = []
    radius = 5
    angle = 0
    max_radius = np.max(np.sqrt((coords[:, 0] - center_y)**2 + (coords[:, 1] - center_x)**2))
    
    while radius < max_radius:
        # Calculate point on spiral
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        # Check if point is inside mask
        if (0 <= int(y) < mask.shape[0] and 0 <= int(x) < mask.shape[1] and 
            mask[int(y), int(x)]):
            spiral_points.append([x, y])
        
        # Update spiral parameters
        angle += 0.3  # Angular step
        radius += 0.5  # Radial step
    
    return np.array(spiral_points) if spiral_points else np.array([[center_x, center_y]])


def calculate_single_stroke_priority(area, centroid, stroke_info):
    """
    Calculate drawing priority for single-stroke objects
    """
    # Larger objects first
    size_priority = area / 10000
    
    # Upper objects first (sky, buildings before ground)
    height_priority = (1000 - centroid[1]) / 1000
    
    # Simpler strokes first (outlines before fills)
    complexity_priority = 1.0 if stroke_info['type'] == 'outline' else 0.5
    
    return 0.5 * size_priority + 0.3 * height_priority + 0.2 * complexity_priority


def create_tactile_object_segmentation(image, max_objects=6, min_object_size=0.05):
    """
    Create object-focused segmentation for minimal pen lifts
    
    Args:
        image: Input RGB image
        max_objects: Maximum number of objects (= maximum pen lifts)
        min_object_size: Minimum object size ratio
    
    Returns:
        tuple: (object_masks, object_info, stroke_count)
    """
    print(f"   🎯 Extracting up to {max_objects} major objects...")
    
    # Extract individual objects
    object_masks, object_info = extract_individual_objects(
        image, max_objects=max_objects, min_object_size=min_object_size
    )
    
    # Calculate total stroke count
    stroke_count = len(object_masks)
    
    print(f"   🖊️  Found {stroke_count} objects = {stroke_count} pen lifts maximum")
    
    return object_masks, object_info, stroke_count


def visualize_object_segmentation(image, object_masks, object_info, save_path=None):
    """
    Visualize object-focused segmentation with stroke planning
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Object masks overlay
    axes[0, 1].imshow(image, alpha=0.7)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(object_masks)))
    for i, mask in enumerate(object_masks):
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask] = [*colors[i][:3], 0.6]
        axes[0, 1].imshow(colored_mask)
    
    axes[0, 1].set_title(f'Object Segmentation ({len(object_masks)} objects)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Pure object masks
    combined_mask = np.zeros(image.shape[:2])
    for i, mask in enumerate(object_masks):
        combined_mask[mask] = i + 1
    
    axes[1, 0].imshow(combined_mask, cmap='tab10')
    axes[1, 0].set_title('Object Regions', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Stroke planning visualization
    axes[1, 1].imshow(image, alpha=0.5)
    
    # Draw stroke paths and priorities
    for obj_id, info in object_info.items():
        stroke_plan = info['stroke_plan']
        centroid = info['centroid']
        
        # Draw stroke path
        if len(stroke_plan['path']) > 1:
            path = stroke_plan['path']
            axes[1, 1].plot(path[:, 0], path[:, 1], 'r-', linewidth=2, alpha=0.8)
            
            # Mark start point
            axes[1, 1].plot(path[0, 0], path[0, 1], 'go', markersize=8, label='Start' if obj_id == 1 else "")
            
            # Mark end point (if not closed)
            if not stroke_plan['is_closed']:
                axes[1, 1].plot(path[-1, 0], path[-1, 1], 'ro', markersize=8, label='End' if obj_id == 1 else "")
        
        # Add object number and priority
        axes[1, 1].text(centroid[0], centroid[1], f'{obj_id}', 
                        ha='center', va='center', fontweight='bold', 
                        color='white', fontsize=12,
                        bbox=dict(boxstyle='circle', facecolor='blue', alpha=0.8))
    
    axes[1, 1].set_title('Single-Stroke Planning', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    if len(object_info) > 0:
        axes[1, 1].legend()
    
    plt.suptitle(f'Object-Focused Segmentation for Tactile Drawing\n{len(object_masks)} Objects = {len(object_masks)} Pen Lifts', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved object segmentation: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test object-focused segmentation
    test_image_path = "../examples/tower.jpg"
    
    if Path(test_image_path).exists():
        # Load image
        image = cv2.imread(test_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Test with different object limits
        for max_objects in [4, 6, 8]:
            print(f"\nTesting with max {max_objects} objects...")
            
            object_masks, object_info, stroke_count = create_tactile_object_segmentation(
                image, max_objects=max_objects, min_object_size=0.04
            )
            
            print(f"Result: {stroke_count} objects found")
            for obj_id, info in object_info.items():
                area_pct = info['area'] / (image.shape[0] * image.shape[1]) * 100
                stroke_type = info['stroke_plan']['type']
                print(f"  Object {obj_id}: {area_pct:.1f}% area, {stroke_type} stroke, priority={info['drawing_priority']:.3f}")
            
            # Visualize
            save_path = f"object_segmentation_{max_objects}objects.png"
            visualize_object_segmentation(image, object_masks, object_info, save_path)
    
    else:
        print(f"Test image not found: {test_image_path}")
