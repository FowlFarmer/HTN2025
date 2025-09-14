"""
Outline-focused segmentation for tactile drawing - detects objects and traces
only their contours/outlines, ignoring filled regions and backgrounds.
Perfect for tactile drawing where object shapes matter, not filled areas.
"""
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy import ndimage
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def detect_object_outlines(image, min_contour_area=500, max_contours=8):
    """
    Detect meaningful object outlines in the image
    
    Args:
        image: Input RGB image (uint8)
        min_contour_area: Minimum area for a contour to be considered
        max_contours: Maximum number of object outlines to extract
    
    Returns:
        tuple: (object_contours, contour_info)
    """
    print(f"   🔍 Detecting object outlines (max {max_contours} objects)...")
    
    # Multi-method outline detection
    all_contours = []
    
    # Method 1: Edge-based contour detection
    edge_contours = detect_contours_from_edges(image, min_contour_area)
    all_contours.extend(edge_contours)
    
    # Method 2: Adaptive threshold contours
    adaptive_contours = detect_contours_adaptive(image, min_contour_area)
    all_contours.extend(adaptive_contours)
    
    # Method 3: Color-based object boundaries
    color_contours = detect_contours_from_color_regions(image, min_contour_area)
    all_contours.extend(color_contours)
    
    # Remove duplicate and overlapping contours
    unique_contours = remove_duplicate_contours(all_contours, overlap_threshold=0.6)
    
    # Rank contours by importance for tactile drawing
    ranked_contours = rank_contours_for_tactile_drawing(unique_contours, image)
    
    # Select top contours
    selected_contours = ranked_contours[:max_contours]
    
    # Analyze contours for drawing optimization
    contour_info = analyze_contours_for_drawing(selected_contours, image)
    
    print(f"   📝 Found {len(selected_contours)} object outlines for tracing")
    
    return selected_contours, contour_info


def detect_contours_from_edges(image, min_area):
    """
    Detect object contours using edge detection methods
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    contours = []
    
    # Multi-scale edge detection for different object sizes
    for blur_size in [1, 3, 5]:
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (blur_size*2+1, blur_size*2+1), blur_size)
        
        # Canny edge detection with different thresholds
        for low_thresh, high_thresh in [(30, 90), (50, 150), (80, 200)]:
            edges = cv2.Canny(blurred, low_thresh, high_thresh)
            
            # Find contours
            found_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in found_contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    # Simplify contour for smoother drawing
                    epsilon = 0.01 * cv2.arcLength(contour, True)
                    simplified = cv2.approxPolyDP(contour, epsilon, True)
                    
                    contour_info = {
                        'contour': simplified,
                        'area': area,
                        'method': 'edges',
                        'params': {'blur': blur_size, 'thresh': (low_thresh, high_thresh)}
                    }
                    contours.append(contour_info)
    
    return contours


def detect_contours_adaptive(image, min_area):
    """
    Detect object contours using adaptive thresholding
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    contours = []
    
    # Multiple adaptive threshold parameters for different object types
    adaptive_params = [
        (11, 2),   # Fine details
        (15, 5),   # Medium objects  
        (21, 10),  # Large objects
        (31, 15)   # Very large objects
    ]
    
    for block_size, C in adaptive_params:
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
        )
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        found_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in found_contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Simplify for smooth drawing
                epsilon = 0.015 * cv2.arcLength(contour, True)
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                
                contour_info = {
                    'contour': simplified,
                    'area': area,
                    'method': 'adaptive',
                    'params': {'block_size': block_size, 'C': C}
                }
                contours.append(contour_info)
    
    return contours


def detect_contours_from_color_regions(image, min_area):
    """
    Detect object contours by finding boundaries of color regions
    """
    # Convert to different color spaces for better object separation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    contours = []
    
    # Process HSV channels (good for distinct colored objects)
    h, s, v = cv2.split(hsv)
    
    for channel, name in [(s, 'saturation'), (v, 'value')]:
        # Use Otsu's thresholding to find natural breakpoints
        _, thresh = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        found_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in found_contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Simplify contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                
                contour_info = {
                    'contour': simplified,
                    'area': area,
                    'method': 'color_regions',
                    'params': {'channel': name}
                }
                contours.append(contour_info)
    
    return contours


def remove_duplicate_contours(contours, overlap_threshold=0.6):
    """
    Remove duplicate and heavily overlapping contours
    """
    if len(contours) <= 1:
        return contours
    
    # Calculate overlap matrix
    n_contours = len(contours)
    overlap_matrix = np.zeros((n_contours, n_contours))
    
    for i in range(n_contours):
        for j in range(i+1, n_contours):
            overlap = calculate_contour_overlap_iou(
                contours[i]['contour'], contours[j]['contour']
            )
            overlap_matrix[i, j] = overlap
            overlap_matrix[j, i] = overlap
    
    # Remove duplicates (keep the one with larger area)
    unique_contours = []
    used = set()
    
    for i in range(n_contours):
        if i in used:
            continue
            
        # Find all contours that overlap significantly with contour i
        overlapping = [i]
        for j in range(n_contours):
            if j != i and j not in used and overlap_matrix[i, j] > overlap_threshold:
                overlapping.append(j)
        
        # Keep the contour with the largest area
        best_idx = max(overlapping, key=lambda idx: contours[idx]['area'])
        unique_contours.append(contours[best_idx])
        
        # Mark all overlapping contours as used
        for idx in overlapping:
            used.add(idx)
    
    return unique_contours


def calculate_contour_overlap_iou(contour1, contour2):
    """
    Calculate Intersection over Union (IoU) between two contours
    """
    # Get bounding boxes
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    
    # Calculate union bounding box
    min_x = min(x1, x2)
    min_y = min(y1, y2)
    max_x = max(x1 + w1, x2 + w2)
    max_y = max(y1 + h1, y2 + h2)
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width <= 0 or height <= 0:
        return 0
    
    # Create masks
    mask1 = np.zeros((height, width), dtype=np.uint8)
    mask2 = np.zeros((height, width), dtype=np.uint8)
    
    # Adjust contours to mask coordinate system
    adjusted_contour1 = contour1.copy()
    adjusted_contour1[:, :, 0] -= min_x
    adjusted_contour1[:, :, 1] -= min_y
    
    adjusted_contour2 = contour2.copy()
    adjusted_contour2[:, :, 0] -= min_x
    adjusted_contour2[:, :, 1] -= min_y
    
    # Fill contours
    cv2.fillPoly(mask1, [adjusted_contour1], 255)
    cv2.fillPoly(mask2, [adjusted_contour2], 255)
    
    # Calculate IoU
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    if np.sum(union) == 0:
        return 0
    
    return np.sum(intersection) / np.sum(union)


def rank_contours_for_tactile_drawing(contours, image):
    """
    Rank contours by importance for tactile drawing
    """
    h, w = image.shape[:2]
    image_area = h * w
    
    for contour_info in contours:
        contour = contour_info['contour']
        
        # Calculate ranking factors
        area = contour_info['area']
        
        # Size importance (larger objects more important)
        size_score = area / image_area
        
        # Position importance (center and upper objects more important)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            cx, cy = x + w_rect//2, y + h_rect//2
        
        # Distance from center
        center_distance = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
        max_distance = np.sqrt((w/2)**2 + (h/2)**2)
        position_score = 1 - (center_distance / max_distance)
        
        # Height importance (upper objects often more important)
        height_score = (h - cy) / h
        
        # Shape complexity (simpler shapes better for tactile)
        perimeter = cv2.arcLength(contour, True)
        compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Convexity (more convex shapes easier to understand tactilely)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # Combined importance score
        importance = (
            0.4 * size_score +      # Size most important
            0.2 * position_score +  # Central objects important
            0.2 * height_score +    # Upper objects important
            0.1 * compactness +     # Simple shapes preferred
            0.1 * convexity         # Convex shapes preferred
        )
        
        contour_info['importance'] = importance
        contour_info['centroid'] = (cx, cy)
        contour_info['compactness'] = compactness
        contour_info['convexity'] = convexity
    
    # Sort by importance (highest first)
    contours.sort(key=lambda x: x['importance'], reverse=True)
    
    return contours


def analyze_contours_for_drawing(contours, image):
    """
    Analyze contours for optimal drawing sequence and stroke planning
    """
    contour_info = {}
    
    for i, contour_data in enumerate(contours):
        contour = contour_data['contour']
        
        # Plan drawing stroke for this contour
        stroke_plan = plan_contour_stroke(contour)
        
        # Calculate drawing priority based on size and position
        drawing_priority = calculate_contour_drawing_priority(contour_data, image)
        
        contour_info[i + 1] = {
            'contour': contour,
            'area': contour_data['area'],
            'centroid': contour_data['centroid'],
            'importance': contour_data['importance'],
            'drawing_priority': drawing_priority,
            'stroke_plan': stroke_plan,
            'detection_method': contour_data['method']
        }
    
    return contour_info


def plan_contour_stroke(contour):
    """
    Plan how to draw a contour as a single smooth stroke
    """
    # Simplify contour for smoother drawing
    epsilon = 0.005 * cv2.arcLength(contour, True)
    simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
    
    # Convert to path points
    path_points = simplified_contour.reshape(-1, 2)
    
    # Determine if contour should be closed
    is_closed = True  # Outlines are typically closed
    
    # Find optimal starting point (topmost, then leftmost)
    if len(path_points) > 0:
        top_points = path_points[path_points[:, 1] == np.min(path_points[:, 1])]
        start_idx = np.argmin(top_points[:, 0])
        start_point_y = np.min(path_points[:, 1])
        start_candidates = np.where(path_points[:, 1] == start_point_y)[0]
        start_idx = start_candidates[np.argmin(path_points[start_candidates, 0])]
        
        # Reorder path to start from optimal point
        reordered_path = np.concatenate([
            path_points[start_idx:],
            path_points[:start_idx]
        ])
    else:
        reordered_path = path_points
    
    return {
        'type': 'outline',
        'path': reordered_path,
        'is_closed': is_closed,
        'length': len(reordered_path),
        'perimeter': cv2.arcLength(contour, True)
    }


def calculate_contour_drawing_priority(contour_data, image):
    """
    Calculate drawing priority for contours (larger, higher objects first)
    """
    h, w = image.shape[:2]
    
    # Size priority (larger first)
    size_priority = contour_data['area'] / (h * w)
    
    # Height priority (upper objects first)
    cy = contour_data['centroid'][1]
    height_priority = (h - cy) / h
    
    # Importance from detection
    importance_priority = contour_data['importance']
    
    # Combined priority
    priority = (
        0.4 * size_priority +
        0.3 * height_priority +
        0.3 * importance_priority
    )
    
    return priority


def create_outline_focused_segmentation(image, max_contours=6, min_contour_area=800):
    """
    Create outline-focused segmentation for tactile drawing
    
    Args:
        image: Input RGB image
        max_contours: Maximum number of object outlines
        min_contour_area: Minimum area for contours
    
    Returns:
        tuple: (object_contours, contour_info, outline_count)
    """
    # Detect object outlines
    object_contours, contour_info = detect_object_outlines(
        image, min_contour_area=min_contour_area, max_contours=max_contours
    )
    
    outline_count = len(object_contours)
    
    print(f"   ✏️  {outline_count} object outlines = {outline_count} pen lifts for tracing")
    
    return object_contours, contour_info, outline_count


def create_outline_image(image, contours, contour_info):
    """
    Create simplified image showing only object outlines
    
    Args:
        image: Original image
        contours: List of contour data
        contour_info: Contour analysis information
    
    Returns:
        numpy.ndarray: Image with only object outlines drawn
    """
    h, w = image.shape[:2]
    
    # Create white background
    outline_image = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Draw each contour with different colors for SAM
    colors = generate_distinct_outline_colors(len(contours))
    
    for i, contour_data in enumerate(contours):
        contour = contour_data['contour']
        color = colors[i]
        
        # Draw thick outline for better SAM detection
        cv2.drawContours(outline_image, [contour], -1, color, thickness=8)
        
        # Draw thinner line for visual clarity
        cv2.drawContours(outline_image, [contour], -1, (0, 0, 0), thickness=2)
    
    return outline_image


def generate_distinct_outline_colors(n_colors):
    """
    Generate distinct colors for outline drawing
    """
    if n_colors <= 0:
        return []
    
    colors = []
    for i in range(n_colors):
        # Use HSV to generate evenly spaced hues, avoid white/black
        # OpenCV HSV: H=0-179, S=0-255, V=0-255
        hue = int((i * 179 / n_colors) % 179)  # Scale to 0-179 for OpenCV
        hsv_color = np.array([[[hue, 200, 180]]], dtype=np.uint8)
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0, 0]
        colors.append(tuple(map(int, rgb_color)))
    
    return colors


def visualize_outline_segmentation(image, contours, contour_info, save_path=None):
    """
    Visualize outline-focused segmentation results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Detected outlines overlay
    axes[0, 1].imshow(image, alpha=0.7)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(contours)))
    for i, contour_data in enumerate(contours):
        contour = contour_data['contour']
        # Draw contour
        contour_points = contour.reshape(-1, 2)
        axes[0, 1].plot(contour_points[:, 0], contour_points[:, 1], 
                       color=colors[i], linewidth=3, alpha=0.8)
        
        # Close the contour
        if len(contour_points) > 0:
            axes[0, 1].plot([contour_points[-1, 0], contour_points[0, 0]], 
                           [contour_points[-1, 1], contour_points[0, 1]], 
                           color=colors[i], linewidth=3, alpha=0.8)
    
    axes[0, 1].set_title(f'Detected Outlines ({len(contours)} objects)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Pure outline drawing
    outline_image = create_outline_image(image, contours, contour_info)
    axes[1, 0].imshow(outline_image)
    axes[1, 0].set_title('Pure Outlines (for SAM)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Drawing sequence visualization
    axes[1, 1].imshow(image, alpha=0.3)
    
    # Draw outlines with drawing order
    for obj_id, info in contour_info.items():
        contour = info['contour']
        stroke_plan = info['stroke_plan']
        
        # Draw outline path
        if len(stroke_plan['path']) > 1:
            path = stroke_plan['path']
            axes[1, 1].plot(path[:, 0], path[:, 1], 'r-', linewidth=2, alpha=0.8)
            
            # Mark start point
            axes[1, 1].plot(path[0, 0], path[0, 1], 'go', markersize=8)
            
            # Mark direction with arrows
            if len(path) > 3:
                mid_idx = len(path) // 2
                dx = path[mid_idx+1, 0] - path[mid_idx-1, 0]
                dy = path[mid_idx+1, 1] - path[mid_idx-1, 1]
                axes[1, 1].arrow(path[mid_idx, 0], path[mid_idx, 1], dx*0.1, dy*0.1,
                               head_width=10, head_length=15, fc='blue', ec='blue')
        
        # Add object number
        centroid = info['centroid']
        axes[1, 1].text(centroid[0], centroid[1], f'{obj_id}', 
                        ha='center', va='center', fontweight='bold', 
                        color='white', fontsize=12,
                        bbox=dict(boxstyle='circle', facecolor='red', alpha=0.8))
    
    axes[1, 1].set_title('Drawing Sequence (Outline Tracing)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.suptitle(f'Outline-Focused Segmentation\n{len(contours)} Object Outlines = {len(contours)} Pen Lifts', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved outline segmentation: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test outline-focused segmentation
    test_image_path = "../examples/tower.jpg"
    
    if Path(test_image_path).exists():
        # Load image
        image = cv2.imread(test_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Test with different contour limits
        for max_contours in [4, 6, 8]:
            print(f"\nTesting with max {max_contours} outlines...")
            
            contours, contour_info, outline_count = create_outline_focused_segmentation(
                image, max_contours=max_contours, min_contour_area=1000
            )
            
            print(f"Result: {outline_count} object outlines found")
            for obj_id, info in contour_info.items():
                area_pct = info['area'] / (image.shape[0] * image.shape[1]) * 100
                method = info['detection_method']
                print(f"  Outline {obj_id}: {area_pct:.1f}% area, {method}, priority={info['drawing_priority']:.3f}")
            
            # Visualize
            save_path = f"outline_segmentation_{max_contours}outlines.png"
            visualize_outline_segmentation(image, contours, contour_info, save_path)
    
    else:
        print(f"Test image not found: {test_image_path}")
