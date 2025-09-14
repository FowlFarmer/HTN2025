import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from step1_preprocessing.image_preprocessing import preprocess_image

def calculate_iou(mask1, mask2):
    """Calculate Intersection over Union between two masks"""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)

def score_mask(mask, image, edge_map):
    """Score mask based on area, edge strength, and color distinctiveness"""
    # Area score (larger elements are more important)
    area_score = np.sum(mask) / (image.shape[0] * image.shape[1])

    # Edge strength (more structured strokes)
    edge_strength = np.mean(edge_map[mask]) if np.any(mask) else 0

    # Color distinctiveness
    if np.sum(mask) == 0 or np.sum(~mask) == 0:
        color_distinctiveness = 0
    else:
        mask_pixels = image[mask]
        outside_pixels = image[~mask]
        color_distinctiveness = np.std(mask_pixels) / (np.std(outside_pixels) + 1e-6)

    # Combined salience score
    salience = area_score * edge_strength * color_distinctiveness

    return {
        'area_score': area_score,
        'edge_strength': edge_strength,
        'color_distinctiveness': color_distinctiveness,
        'salience': salience
    }

def merge_masks(masks, iou_threshold=0.6):
    """Merge overlapping masks with IoU > threshold"""
    if len(masks) == 0:
        return []

    merged = []
    used = set()

    for i, mask1 in enumerate(masks):
        if i in used:
            continue

        current_mask = mask1.copy()
        for j, mask2 in enumerate(masks[i+1:], i+1):
            if j in used:
                continue

            iou = calculate_iou(mask1, mask2)
            if iou > iou_threshold:
                current_mask = np.logical_or(current_mask, mask2)
                used.add(j)

        merged.append(current_mask)
        used.add(i)

    return merged

def segment_painting(image_path, model_path=None, processed_img=None):
    """
    Segment painting into meaningful elements using SAM

    Args:
        image_path (str): Path to the painting image
        model_path (str): Path to SAM model checkpoint
        processed_img (numpy.ndarray): Optional pre-processed image to avoid double processing

    Returns:
        tuple: (selected_masks, mask_scores, processed_image)
    """
    # Determine model path
    if model_path is None:
        # Try different possible locations
        possible_paths = [
            "models/sam_vit_h_4b8939.pth",
            "../models/sam_vit_h_4b8939.pth",
            "sam_vit_h_4b8939.pth"
        ]
        model_path = None
        for path in possible_paths:
            from pathlib import Path
            if Path(path).exists():
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError("SAM model not found. Please download sam_vit_h_4b8939.pth to models/ directory")

    # Use provided processed image or load and preprocess
    if processed_img is not None:
        # Use the already processed image
        pass
    else:
        # Load and preprocess image
        processed_img, scale_factor, original_dims = preprocess_image(image_path)

    # Convert back to uint8 for SAM (expects 0-255 range)
    sam_input = (processed_img * 255).astype(np.uint8)

    # Create edge map for scoring
    gray = cv2.cvtColor(sam_input, cv2.COLOR_RGB2GRAY)
    edge_map = cv2.Canny(gray, 50, 150)
    edge_map = edge_map.astype(np.float32) / 255.0

    # Initialize SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=model_path)
    sam.to(device=device)

    # Generate masks automatically
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    masks = mask_generator.generate(sam_input)

    # Extract mask arrays and score them
    mask_arrays = []
    mask_scores = []

    for mask_data in masks:
        mask = mask_data['segmentation']

        # Filter by area constraints (1% to 80% of image)
        area_ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1])
        if area_ratio < 0.01 or area_ratio > 0.8:
            continue

        scores = score_mask(mask, processed_img, edge_map)
        mask_arrays.append(mask)
        mask_scores.append(scores)

    # Sort by salience score (highest first)
    sorted_indices = sorted(range(len(mask_scores)),
                          key=lambda i: mask_scores[i]['salience'],
                          reverse=True)

    # Select top masks (3-6 for demo)
    selected_masks = []
    selected_scores = []
    max_masks = 4

    for idx in sorted_indices[:max_masks]:
        selected_masks.append(mask_arrays[idx])
        selected_scores.append(mask_scores[idx])

    # Merge highly overlapping masks
    selected_masks = merge_masks(selected_masks, iou_threshold=0.6)

    # Limit to target count
    target_count = min(6, len(selected_masks))
    selected_masks = selected_masks[:target_count]

    return selected_masks, selected_scores[:len(selected_masks)], processed_img

def save_segmentation_results(masks, scores, processed_img, output_dir="results/step2_segmentation"):
    """
    Save segmentation results with timestamp and consistent scaling

    Args:
        masks: List of segmentation masks
        scores: List of mask quality scores
        processed_img: Preprocessed image
        output_dir: Directory to save results

    Returns:
        str: Path to saved visualization
    """
    import sys
    from pathlib import Path
    import matplotlib.pyplot as plt
    sys.path.append(str(Path(__file__).parent.parent))
    
    from visualization_utils import (
        get_consistent_figure_layout, create_consistent_background_image,
        create_consistent_mask_overlay, set_consistent_axis_properties,
        save_visualization_with_timestamp, hide_unused_subplots,
        NEON_GREEN_OVERLAY, STANDARD_TARGET_SIZE
    )

    # Create consistent figure layout
    fig, axes = get_consistent_figure_layout()

    # Create consistent background image
    background_img = create_consistent_background_image(processed_img, STANDARD_TARGET_SIZE)
    original_shape = processed_img.shape

    for i, mask in enumerate(masks):
        if i >= 6:
            break
        
        # Show consistent background
        axes[i].imshow(background_img)
        
        # Create consistent mask overlay
        mask_overlay = create_consistent_mask_overlay(
            mask, original_shape, STANDARD_TARGET_SIZE, NEON_GREEN_OVERLAY
        )
        axes[i].imshow(mask_overlay)
        
        # Set consistent axis properties
        title = f'Mask {i+1} (salience: {scores[i]["salience"]:.3f})'
        set_consistent_axis_properties(axes[i], title, STANDARD_TARGET_SIZE)

    # Hide unused subplots
    hide_unused_subplots(axes, len(masks))

    plt.suptitle(f'Step 2: Segmentation Results', fontsize=16)
    plt.tight_layout()

    # Save with consistent timestamp and path handling
    return save_visualization_with_timestamp(fig, output_dir, 'segmentation_results')

if __name__ == "__main__":
    # Test segmentation on monet.jpeg
    masks, scores, processed_img = segment_painting("monet.jpeg")

    print(f"Generated {len(masks)} masks")
    for i, score in enumerate(scores):
        print(f"Mask {i}: area={score['area_score']:.3f}, "
              f"edge={score['edge_strength']:.3f}, "
              f"color={score['color_distinctiveness']:.3f}, "
              f"salience={score['salience']:.3f}")

    # Save visualization
    save_segmentation_results(masks, scores, processed_img)