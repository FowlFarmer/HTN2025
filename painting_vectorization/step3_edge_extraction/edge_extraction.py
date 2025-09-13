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

def multi_scale_edges(gray):
    """
    Combine multiple edge scales for robust detection

    Args:
        gray: Grayscale image (uint8)

    Returns:
        numpy.ndarray: Combined edge map
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

def morphological_cleaning(edges):
    """
    Clean edge map using morphological operations

    Args:
        edges: Binary edge map (uint8)

    Returns:
        numpy.ndarray: Cleaned edge map
    """
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

def extract_edges_from_mask(image, mask, method="multi_scale"):
    """
    Extract clean binary line image from a masked region

    Args:
        image: Input image (numpy array, float32, [0,1])
        mask: Binary mask (boolean array)
        method: Edge detection method ("canny", "multi_scale")

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
    if method == "multi_scale":
        edges = multi_scale_edges(gray)
    else:  # Default to Canny
        edges = cv2.Canny(gray, 50, 150)

    # Morphological cleaning
    edges_cleaned = morphological_cleaning(edges)

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

def process_all_masks(image, masks, method="multi_scale"):
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