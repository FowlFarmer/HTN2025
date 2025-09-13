import cv2
import numpy as np
from skimage.morphology import skeletonize
from datetime import datetime

def crop_to_mask(image, mask):
    """
    Crop image to mask bounding box for efficiency

    Args:
        image: Input image (numpy array)
        mask: Binary mask (boolean array)

    Returns:
        tuple: (cropped_image, cropped_mask, offset)
            cropped_image: Cropped image region
            cropped_mask: Cropped mask region
            offset: (x_min, y_min) offset for coordinate mapping
    """
    coords = np.where(mask)
    if len(coords[0]) == 0:  # Empty mask
        return None, None, (0, 0)

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # Add padding to avoid edge artifacts
    pad = 5
    y_min = max(0, y_min - pad)
    y_max = min(image.shape[0], y_max + pad + 1)
    x_min = max(0, x_min - pad)
    x_max = min(image.shape[1], x_max + pad + 1)

    cropped_img = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    return cropped_img, cropped_mask, (x_min, y_min)

def preprocess_for_edges(cropped_img, cropped_mask):
    """
    Preprocess cropped image for edge detection

    Args:
        cropped_img: Cropped RGB image (float32, [0,1])
        cropped_mask: Binary mask for the region

    Returns:
        numpy.ndarray: Preprocessed grayscale image
    """
    # Convert to uint8 for OpenCV
    img_uint8 = (cropped_img * 255).astype(np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    # Apply bilateral filter to remove texture noise while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Mask out areas outside the region of interest
    gray[~cropped_mask] = 255  # Set background to white

    return gray

def extract_object_contours(gray, cropped_mask):
    """
    Extract clean object contours focusing on boundaries, not internal texture

    Args:
        gray: Grayscale image (uint8)
        cropped_mask: Binary mask for the region

    Returns:
        numpy.ndarray: Clean contour map
    """
    # Step 1: Strong smoothing to remove brush texture but preserve object boundaries
    smooth = cv2.GaussianBlur(gray, (7, 7), 2.0)

    # Step 2: Use higher Canny thresholds to get only strong boundaries
    edges = cv2.Canny(smooth, 80, 160)

    # Step 3: Find contours and filter by significance
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create clean contour image
    contour_img = np.zeros_like(gray)

    # Filter contours by area and draw only significant ones
    min_contour_area = max(50, np.sum(cropped_mask) * 0.001)  # At least 0.1% of mask area

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_contour_area:
            # Simplify contour to reduce noise
            epsilon = 0.005 * cv2.arcLength(contour, True)  # Approximation accuracy
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(contour_img, [approx], -1, 255, 1)

    return contour_img

def extract_boundary_edges(gray, cropped_mask):
    """
    Extract boundary-focused edges with minimal internal texture

    Args:
        gray: Grayscale image (uint8)
        cropped_mask: Binary mask for the region

    Returns:
        numpy.ndarray: Clean boundary edge map
    """
    # Method 1: Contour-based approach
    contour_edges = extract_object_contours(gray, cropped_mask)

    # Method 2: Conservative Canny for strong boundaries only
    # Apply stronger smoothing to reduce texture
    smooth = cv2.bilateralFilter(gray, 9, 80, 80)

    # Use higher thresholds to get only prominent edges
    canny_edges = cv2.Canny(smooth, 100, 200)

    # Method 3: Morphological gradient to find boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gradient = cv2.morphologyEx(smooth, cv2.MORPH_GRADIENT, kernel)
    _, gradient_edges = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)

    # Combine the best of all methods
    combined = cv2.bitwise_or(contour_edges, canny_edges)
    combined = cv2.bitwise_or(combined, gradient_edges)

    # Final cleanup - remove very small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined)
    min_area = max(30, np.sum(cropped_mask) * 0.0005)  # At least 0.05% of mask area

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            combined[labels == i] = 0

    return combined

def detect_structural_lines(edges):
    """
    Detect structural lines (horizontal, vertical, geometric patterns)

    Args:
        edges: Binary edge map

    Returns:
        numpy.ndarray: Filtered edges containing only structural elements
    """
    # Find all line segments using HoughLinesP
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30,
                           minLineLength=20, maxLineGap=10)

    if lines is None:
        return np.zeros_like(edges)

    structural_lines = np.zeros_like(edges)

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate line properties
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length < 15:  # Skip very short lines
            continue

        # Calculate angle
        angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
        angle = abs(angle)

        # Keep lines that are roughly horizontal, vertical, or diagonal
        is_horizontal = abs(angle) < 15 or abs(angle-180) < 15
        is_vertical = abs(angle-90) < 15
        is_diagonal = abs(angle-45) < 15 or abs(angle-135) < 15

        if is_horizontal or is_vertical or is_diagonal:
            # Draw structural lines
            cv2.line(structural_lines, (x1, y1), (x2, y2), 255, 1)

    return structural_lines

def analyze_geometric_patterns(edges):
    """
    Analyze edges for geometric patterns and regular structures

    Args:
        edges: Binary edge map

    Returns:
        numpy.ndarray: Edges containing geometric patterns
    """
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pattern_edges = np.zeros_like(edges)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # Skip tiny contours
            continue

        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Analyze shape characteristics
        num_vertices = len(approx)

        # Keep geometric shapes (rectangles, triangles, etc.)
        if 3 <= num_vertices <= 8:  # Triangles to octagons
            # Check if it's roughly rectangular
            if num_vertices == 4:
                # Calculate aspect ratio and angles for rectangles
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect_ratio = max(width/height, height/width)
                    # Keep rectangles and squares
                    if aspect_ratio < 10:  # Not too elongated
                        cv2.drawContours(pattern_edges, [approx], -1, 255, 1)
            else:
                # Keep other geometric shapes
                cv2.drawContours(pattern_edges, [approx], -1, 255, 1)

    return pattern_edges

def classify_texture_vs_structure(edges, cropped_mask):
    """
    Classify edges as texture or structural elements

    Args:
        edges: Binary edge map
        cropped_mask: Mask region

    Returns:
        numpy.ndarray: Structural edges only
    """
    # Stage 1: Structural line detection
    structural_lines = detect_structural_lines(edges)

    # Stage 2: Geometric pattern recognition
    geometric_patterns = analyze_geometric_patterns(edges)

    # Stage 3: Size-based filtering
    # Find connected components and filter by size
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges)

    size_filtered = np.zeros_like(edges)
    mask_area = np.sum(cropped_mask)
    min_component_area = max(50, mask_area * 0.002)  # 0.2% of mask area

    for i in range(1, num_labels):
        component_area = stats[i, cv2.CC_STAT_AREA]
        if component_area >= min_component_area:
            # Keep significant components
            component_mask = labels == i
            size_filtered[component_mask] = 255

    # Combine all structural elements
    structural_edges = cv2.bitwise_or(structural_lines, geometric_patterns)
    structural_edges = cv2.bitwise_or(structural_edges, size_filtered)

    # Final cleanup - remove isolated pixels
    kernel = np.ones((3, 3), np.uint8)
    structural_edges = cv2.morphologyEx(structural_edges, cv2.MORPH_OPEN, kernel)

    return structural_edges

def extract_clean_structural_edges(gray, cropped_mask):
    """
    Extract ultra-clean structural edges with coloring-book style outlines

    Args:
        gray: Grayscale image
        cropped_mask: Binary mask for the region

    Returns:
        numpy.ndarray: Clean structural edge map
    """
    # Stage 1: Heavy smoothing to remove artistic texture
    # But preserve sharp structural boundaries
    smooth_heavy = cv2.GaussianBlur(gray, (15, 15), 4.0)

    # Stage 2: Edge-preserving bilateral filter for structures
    smooth_bilateral = cv2.bilateralFilter(gray, 15, 100, 100)

    # Stage 3: Multi-scale structural detection

    # Method 1: High-threshold Canny on heavily smoothed image (major boundaries)
    major_edges = cv2.Canny(smooth_heavy, 150, 250)

    # Method 2: Medium-threshold Canny on bilateral filtered image (structural details)
    structural_edges = cv2.Canny(smooth_bilateral, 100, 180)

    # Method 3: Morphological gradient for boundary detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    gradient = cv2.morphologyEx(smooth_bilateral, cv2.MORPH_GRADIENT, kernel)
    _, gradient_edges = cv2.threshold(gradient, 40, 255, cv2.THRESH_BINARY)

    # Combine edge maps
    combined = cv2.bitwise_or(major_edges, structural_edges)
    combined = cv2.bitwise_or(combined, gradient_edges)

    # Stage 4: Intelligent structural filtering
    filtered_edges = classify_texture_vs_structure(combined, cropped_mask)

    # Stage 5: Final structural enhancement
    # Connect nearby structural elements
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    connected = cv2.morphologyEx(filtered_edges, cv2.MORPH_CLOSE, kernel_connect)

    # Remove very small components (noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(connected)
    final_edges = np.zeros_like(connected)

    mask_area = np.sum(cropped_mask)
    min_area = max(30, mask_area * 0.001)  # 0.1% of mask area

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            final_edges[labels == i] = 255

    return final_edges

def detect_pattern_regularity(edges):
    """
    Detect regular patterns in edges using spatial consistency analysis

    Args:
        edges: Binary edge map

    Returns:
        numpy.ndarray: Mask indicating pattern regions
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges)

    pattern_mask = np.zeros_like(edges)

    if num_labels < 3:  # Need at least 2 components plus background
        return pattern_mask

    # Analyze spatial relationships between components
    components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 10:  # Skip tiny components
            continue

        x, y = centroids[i]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

        components.append({
            'id': i,
            'center': (x, y),
            'area': area,
            'size': (w, h),
            'aspect_ratio': max(w, h) / (min(w, h) + 1e-6)
        })

    # Group components by similar size and regular spacing
    for i, comp1 in enumerate(components):
        similar_components = []

        for j, comp2 in enumerate(components):
            if i == j:
                continue

            # Check size similarity
            area_ratio = min(comp1['area'], comp2['area']) / max(comp1['area'], comp2['area'])
            if area_ratio < 0.3:  # Must be similar size
                continue

            # Check aspect ratio similarity
            aspect_diff = abs(comp1['aspect_ratio'] - comp2['aspect_ratio'])
            if aspect_diff > 2.0:  # Must have similar shape
                continue

            similar_components.append(comp2)

        # If we found similar components, check for regular spacing
        if len(similar_components) >= 1:  # At least one similar component
            # Mark these as pattern components
            pattern_mask[labels == comp1['id']] = 255
            for comp in similar_components:
                pattern_mask[labels == comp['id']] = 255

    return pattern_mask

def analyze_local_complexity(gray, window_size=31):
    """
    Analyze local complexity to determine filtering strength

    Args:
        gray: Grayscale image
        window_size: Size of analysis window

    Returns:
        numpy.ndarray: Complexity map (higher values = more structured)
    """
    # Calculate local standard deviation (texture measure)
    kernel = np.ones((window_size, window_size), np.float32) / (window_size ** 2)
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_var = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
    texture_measure = np.sqrt(local_var)

    # Calculate edge density (structure measure)
    edges_temp = cv2.Canny(gray, 50, 100)
    edge_density = cv2.filter2D(edges_temp.astype(np.float32), -1, kernel)

    # Calculate gradient orientation consistency (structure measure)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Local gradient orientation consistency
    orientation = np.arctan2(grad_y, grad_x + 1e-6)
    local_orientation_consistency = np.zeros_like(gray, dtype=np.float32)

    half_window = window_size // 2
    for y in range(half_window, gray.shape[0] - half_window):
        for x in range(half_window, gray.shape[1] - half_window):
            local_orientations = orientation[y-half_window:y+half_window+1,
                                           x-half_window:x+half_window+1]
            # Measure orientation consistency (lower std = more consistent)
            consistency = 1.0 / (np.std(local_orientations) + 0.1)
            local_orientation_consistency[y, x] = consistency

    # Combine measures (higher = more structured)
    structure_score = edge_density * local_orientation_consistency / (texture_measure + 1.0)

    # Normalize to [0, 1]
    if np.max(structure_score) > 0:
        structure_score = structure_score / np.max(structure_score)

    return structure_score

def extract_adaptive_structural_edges(gray, cropped_mask):
    """
    Extract edges using content-adaptive pattern preservation

    Args:
        gray: Grayscale image
        cropped_mask: Binary mask for the region

    Returns:
        numpy.ndarray: Adaptively filtered edge map
    """
    # Stage 1: Multi-scale edge detection
    # Light smoothing for fine patterns
    light_smooth = cv2.bilateralFilter(gray, 5, 50, 50)
    fine_edges = cv2.Canny(light_smooth, 80, 160)

    # Medium smoothing for major structures
    medium_smooth = cv2.GaussianBlur(gray, (7, 7), 1.5)
    major_edges = cv2.Canny(medium_smooth, 100, 200)

    # Heavy smoothing for overall boundaries
    heavy_smooth = cv2.GaussianBlur(gray, (11, 11), 3.0)
    boundary_edges = cv2.Canny(heavy_smooth, 120, 240)

    # Stage 2: Analyze local complexity
    complexity_map = analyze_local_complexity(gray)

    # Stage 3: Detect pattern regions
    combined_edges = cv2.bitwise_or(fine_edges, major_edges)
    combined_edges = cv2.bitwise_or(combined_edges, boundary_edges)

    pattern_regions = detect_pattern_regularity(combined_edges)

    # Stage 4: Adaptive filtering based on local content
    adaptive_edges = np.zeros_like(gray)

    # Create adaptive threshold based on complexity
    height, width = gray.shape
    for y in range(0, height, 16):  # Process in blocks for efficiency
        for x in range(0, width, 16):
            block_y_end = min(y + 16, height)
            block_x_end = min(x + 16, width)

            # Get local complexity score
            local_complexity = np.mean(complexity_map[y:block_y_end, x:block_x_end])

            # Get local pattern strength
            local_pattern = np.mean(pattern_regions[y:block_y_end, x:block_x_end]) / 255.0

            # Determine which edge map to use based on content
            if local_pattern > 0.3:  # High pattern region - keep fine details
                adaptive_edges[y:block_y_end, x:block_x_end] = fine_edges[y:block_y_end, x:block_x_end]
            elif local_complexity > 0.6:  # High structure region - use major edges
                adaptive_edges[y:block_y_end, x:block_x_end] = major_edges[y:block_y_end, x:block_x_end]
            else:  # Low complexity region - use only boundaries
                adaptive_edges[y:block_y_end, x:block_x_end] = boundary_edges[y:block_y_end, x:block_x_end]

    # Stage 5: Enhance pattern regions
    pattern_enhanced = cv2.bitwise_or(adaptive_edges.astype(np.uint8), pattern_regions)

    # Stage 6: Final cleanup based on significance
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pattern_enhanced)
    final_edges = np.zeros_like(pattern_enhanced)

    mask_area = np.sum(cropped_mask)

    for i in range(1, num_labels):
        component_area = stats[i, cv2.CC_STAT_AREA]

        # Dynamic threshold based on mask size and local context
        # Smaller threshold for pattern regions, larger for non-pattern areas
        component_mask = labels == i
        local_pattern_strength = np.mean(pattern_regions[component_mask]) / 255.0

        if local_pattern_strength > 0.1:  # In pattern region
            min_area = max(5, mask_area * 0.0001)  # Very permissive
        else:  # In non-pattern region
            min_area = max(20, mask_area * 0.0005)  # More restrictive

        if component_area >= min_area:
            final_edges[component_mask] = 255

    # Stage 7: Light morphological enhancement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_edges = cv2.morphologyEx(final_edges, cv2.MORPH_CLOSE, kernel)

    return final_edges

def multi_scale_edges(gray):
    """
    DEPRECATED: Old multi-scale method - too noisy
    Use extract_boundary_edges instead for cleaner results
    """
    # Low threshold for faint edges
    edges_low = cv2.Canny(gray, 30, 100)

    # High threshold for strong edges
    edges_high = cv2.Canny(gray, 80, 200)

    # Medium threshold
    edges_med = cv2.Canny(gray, 50, 150)

    # Combine all scales
    edges_combined = cv2.bitwise_or(edges_low, edges_high)
    edges_combined = cv2.bitwise_or(edges_combined, edges_med)

    return edges_combined

def morphological_cleaning(edges, gentle=False):
    """
    Clean edge map using morphological operations

    Args:
        edges: Binary edge map (uint8)
        gentle: Use gentler cleaning for boundary method

    Returns:
        numpy.ndarray: Cleaned edge map
    """
    if gentle:
        # Gentler cleaning for boundary method - preserve clean contours
        kernel = np.ones((2, 2), np.uint8)
        # Only light closing to fill tiny gaps
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Remove only very small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges)
        min_area = 10  # Very small threshold

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                edges[labels == i] = 0
    else:
        # Original aggressive cleaning for texture-heavy methods
        # Closing to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Opening to remove small specks
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

        # Remove small connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges)
        min_area = 20  # pixels

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                edges[labels == i] = 0

    return edges

def extract_skeleton(edges):
    """
    Extract skeleton from binary edge map

    Args:
        edges: Binary edge map (uint8)

    Returns:
        numpy.ndarray: Skeletonized image (boolean)
    """
    # Convert to binary
    binary = edges > 0

    # Skeletonize
    skeleton = skeletonize(binary)

    return skeleton

def extract_edges_from_mask(image, mask, method="adaptive"):
    """
    Extract clean binary line image from a masked region

    Args:
        image: Input image (numpy array, float32, [0,1])
        mask: Binary mask (boolean array)
        method: Edge detection method ("adaptive", "structural", "boundary", "multi_scale", "canny")

    Returns:
        dict: Dictionary containing extraction results
            - skeleton: Skeletonized edge map (boolean array)
            - edges: Raw edge map before skeletonization
            - cropped_region: Cropped image region
            - offset: Coordinate offset for mapping back
            - stats: Extraction statistics
    """
    # Crop to mask bounding box
    cropped_img, cropped_mask, offset = crop_to_mask(image, mask)

    if cropped_img is None:
        return None

    # Preprocess for edge detection
    gray = preprocess_for_edges(cropped_img, cropped_mask)

    # Edge detection
    if method == "adaptive":
        edges = extract_adaptive_structural_edges(gray, cropped_mask)
    elif method == "structural":
        edges = extract_clean_structural_edges(gray, cropped_mask)
    elif method == "boundary":
        edges = extract_boundary_edges(gray, cropped_mask)
    elif method == "multi_scale":
        edges = multi_scale_edges(gray)
    else:  # Default to Canny
        edges = cv2.Canny(gray, 50, 150)

    # Morphological cleaning
    gentle_cleaning = (method in ["boundary", "structural", "adaptive"])
    edges_cleaned = morphological_cleaning(edges, gentle=gentle_cleaning)

    # Skeletonization
    skeleton = extract_skeleton(edges_cleaned)

    # Calculate statistics
    total_pixels = np.sum(cropped_mask)
    edge_pixels = np.sum(edges_cleaned > 0)
    skeleton_pixels = np.sum(skeleton)

    stats = {
        'total_pixels': int(total_pixels),
        'edge_pixels': int(edge_pixels),
        'skeleton_pixels': int(skeleton_pixels),
        'edge_density': edge_pixels / total_pixels if total_pixels > 0 else 0,
        'skeleton_density': skeleton_pixels / total_pixels if total_pixels > 0 else 0,
        'crop_size': cropped_img.shape[:2]
    }

    return {
        'skeleton': skeleton,
        'edges': edges_cleaned,
        'cropped_region': cropped_img,
        'cropped_mask': cropped_mask,
        'offset': offset,
        'stats': stats
    }

def process_all_masks(image, masks, method="adaptive"):
    """
    Extract edges from all masks

    Args:
        image: Input image (numpy array, float32, [0,1])
        masks: List of binary masks
        method: Edge detection method

    Returns:
        list: List of extraction results for each mask
    """
    results = []

    for i, mask in enumerate(masks):
        print(f"   🎨 Processing mask {i+1}/{len(masks)}...")

        result = extract_edges_from_mask(image, mask, method)

        if result is not None:
            results.append(result)
            stats = result['stats']
            print(f"      📏 Crop size: {stats['crop_size']}")
            print(f"      ⚡ Edge density: {stats['edge_density']:.3f}")
            print(f"      🖋️  Skeleton density: {stats['skeleton_density']:.3f}")
        else:
            print(f"      ⚠️  Empty mask, skipping...")
            results.append(None)

    return results

def save_edge_results(results, masks, image_shape, output_dir="examples"):
    """
    Save edge extraction results with timestamp

    Args:
        results: List of extraction results
        masks: List of original masks
        image_shape: Shape of original image
        output_dir: Directory to save results

    Returns:
        str: Path to saved visualization
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    valid_results = [r for r in results if r is not None]

    for i, result in enumerate(valid_results[:6]):
        if i >= 6:
            break

        ax = axes[i]

        # Show skeleton on white background
        skeleton_display = np.ones(result['cropped_region'].shape[:2], dtype=np.uint8) * 255
        skeleton_display[result['skeleton']] = 0  # Black lines

        ax.imshow(skeleton_display, cmap='gray')
        ax.set_title(f'Edge Extraction {i+1}\n'
                    f'Skeleton: {result["stats"]["skeleton_pixels"]} pixels')
        ax.axis('off')

    # Hide unused subplots
    for i in range(len(valid_results), 6):
        axes[i].axis('off')

    plt.suptitle(f'Edge Extraction Results - {timestamp}', fontsize=16)
    plt.tight_layout()

    # Save with timestamp
    output_path = f"{output_dir}/edge_extraction_results_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   💾 Saved visualization: {output_path}")
    return output_path

if __name__ == "__main__":
    # Test edge extraction
    import sys
    sys.path.append('..')

    from step1_preprocessing.image_preprocessing import preprocess_image
    from step2_segmentation.segmentation import segment_painting

    # Process test image
    image_path = "../examples/monet.jpeg"

    print("🎨 Testing Edge Extraction Pipeline")
    print("=" * 40)

    # Get preprocessed image and masks
    processed_img, _, _ = preprocess_image(image_path)
    masks, scores, _ = segment_painting(image_path)

    print(f"📐 Processing {len(masks)} masks...")

    # Extract edges from all masks
    edge_results = process_all_masks(processed_img, masks)

    # Save results
    output_path = save_edge_results(edge_results, masks, processed_img.shape)

    print("✅ Edge extraction complete!")
    print(f"📊 Results: {len([r for r in edge_results if r is not None])} successful extractions")