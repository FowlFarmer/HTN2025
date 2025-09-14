#!/usr/bin/env python3
"""
Step 7: Color Detection - Warm vs Cool Classification

Analyze color properties of image regions to classify strokes as warm (0) or cool (1)
based on perceptual color analysis using CIELAB color space.

Features:
- CIELAB color space analysis for perceptual uniformity
- HSV alternative method for hue-based classification
- Quality assurance and edge case handling
- Association with stroke sequences from previous steps
- Robust handling of neutral/grayscale regions
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    from skimage.color import rgb2lab
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Install with: pip install scikit-image")

def classify_mask_warmth_lab(rgb_img, mask, a_thresh=5.0, b_thresh=5.0,
                           warm_ratio_thresh=0.6, chroma_thresh=6.0):
    """
    Classify mask as warm (0) or cool (1) using CIELAB color space analysis

    Args:
        rgb_img: RGB image (0-255 range)
        mask: Binary mask for the region
        a_thresh: Threshold for a* channel (green-red axis)
        b_thresh: Threshold for b* channel (blue-yellow axis)
        warm_ratio_thresh: Minimum ratio for warm classification
        chroma_thresh: Minimum chroma for non-neutral classification

    Returns:
        tuple: (warmth_class, analysis_dict)
            - warmth_class: 0 for warm, 1 for cool
            - analysis_dict: Detailed analysis metrics
    """
    if not SKIMAGE_AVAILABLE:
        # Fallback to HSV method
        return classify_mask_warmth_hsv(rgb_img, mask)

    if np.sum(mask) == 0:
        return 1, {  # Default to cool for empty masks
            'warm_ratio': 0.0,
            'median_chroma': 0.0,
            'classification': 'empty->cool',
            'pixel_count': 0,
            'method': 'lab'
        }

    # Convert to LAB color space (expects 0-1 range)
    lab = rgb2lab(rgb_img.astype(np.float64) / 255.0)

    # Extract LAB values for masked pixels
    a_values = lab[..., 1][mask]  # Green-Red axis
    b_values = lab[..., 2][mask]  # Blue-Yellow axis
    l_values = lab[..., 0][mask]  # Lightness

    # Calculate chroma (saturation measure)
    chroma = np.sqrt(a_values**2 + b_values**2)

    # Per-pixel warm classification
    # Warm: Red (a* > thresh) OR Yellow (b* > thresh)
    warm_votes = ((a_values > a_thresh) | (b_values > b_thresh))
    warm_ratio = np.sum(warm_votes) / len(a_values)

    # Statistical measures
    median_chroma = np.median(chroma)
    mean_a = np.mean(a_values)
    mean_b = np.mean(b_values)
    mean_lightness = np.mean(l_values)

    # Handle low saturation (neutral/grayscale) regions
    if median_chroma < chroma_thresh:
        # Neutral region - use slight bias toward cool (safer for most art)
        warmth_class = 1
        classification = 'neutral->cool'
    else:
        # Classify based on warm ratio
        warmth_class = 0 if warm_ratio >= warm_ratio_thresh else 1
        classification = 'warm' if warmth_class == 0 else 'cool'

    return warmth_class, {
        'warm_ratio': warm_ratio,
        'median_chroma': median_chroma,
        'mean_a': mean_a,
        'mean_b': mean_b,
        'mean_lightness': mean_lightness,
        'classification': classification,
        'pixel_count': len(a_values),
        'method': 'lab'
    }

def classify_mask_warmth_hsv(rgb_img, mask, warm_hue_range=(330, 60),
                           warm_ratio_thresh=0.6, saturation_thresh=0.2):
    """
    Alternative classification using HSV hue analysis

    Args:
        rgb_img: RGB image (0-255 range)
        mask: Binary mask for the region
        warm_hue_range: Tuple of (start_angle, end_angle) for warm colors in degrees
        warm_ratio_thresh: Minimum ratio for warm classification
        saturation_thresh: Minimum saturation to consider color (0-1 range)

    Returns:
        tuple: (warmth_class, analysis_dict)
    """
    if np.sum(mask) == 0:
        return 1, {  # Default to cool for empty masks
            'warm_ratio': 0.0,
            'mean_saturation': 0.0,
            'classification': 'empty->cool',
            'pixel_count': 0,
            'method': 'hsv'
        }

    # Convert to HSV
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    # Extract HSV values for masked pixels
    hue_values = hsv[..., 0][mask].astype(np.float32) * 2  # Convert to 0-360 degrees
    saturation_values = hsv[..., 1][mask] / 255.0  # Convert to 0-1 range
    value_values = hsv[..., 2][mask] / 255.0  # Convert to 0-1 range

    # Filter out low saturation pixels (near grayscale)
    saturated_mask = saturation_values >= saturation_thresh

    if np.sum(saturated_mask) == 0:
        # No saturated colors - default to cool
        return 1, {
            'warm_ratio': 0.0,
            'mean_saturation': np.mean(saturation_values),
            'classification': 'unsaturated->cool',
            'pixel_count': len(hue_values),
            'method': 'hsv'
        }

    # Apply saturation filter
    filtered_hues = hue_values[saturated_mask]

    # Handle hue wraparound for warm colors (red spans 330-360 and 0-60 degrees)
    start_hue, end_hue = warm_hue_range
    if start_hue > end_hue:  # Wraparound case (red)
        warm_votes = ((filtered_hues >= start_hue) | (filtered_hues <= end_hue))
    else:
        warm_votes = ((filtered_hues >= start_hue) & (filtered_hues <= end_hue))

    warm_ratio = np.sum(warm_votes) / len(filtered_hues) if len(filtered_hues) > 0 else 0.0

    # Classification
    warmth_class = 0 if warm_ratio >= warm_ratio_thresh else 1

    return warmth_class, {
        'warm_ratio': warm_ratio,
        'mean_saturation': np.mean(saturation_values),
        'mean_hue': np.mean(filtered_hues) if len(filtered_hues) > 0 else 0.0,
        'saturated_pixels': np.sum(saturated_mask),
        'classification': 'warm' if warmth_class == 0 else 'cool',
        'pixel_count': len(hue_values),
        'method': 'hsv'
    }

def validate_classification_quality(rgb_img, mask, min_pixels=200):
    """
    Validate classification quality and handle edge cases

    Args:
        rgb_img: RGB image
        mask: Binary mask
        min_pixels: Minimum pixels required for reliable classification

    Returns:
        tuple: (is_reliable, suggested_method)
    """
    pixel_count = np.sum(mask)

    # Check minimum pixel count
    if pixel_count < min_pixels:
        return False, 'insufficient_pixels'

    # Check for extreme lighting conditions
    if detect_lighting_bias(rgb_img, mask):
        return False, 'lighting_bias'

    # Check color diversity
    color_diversity = calculate_color_diversity(rgb_img, mask)
    if color_diversity < 0.1:  # Very uniform color
        return False, 'uniform_color'

    return True, 'reliable'

def detect_lighting_bias(rgb_img, mask, bias_threshold=0.8):
    """
    Detect extreme lighting conditions that might bias color classification

    Args:
        rgb_img: RGB image
        mask: Binary mask
        bias_threshold: Threshold for detecting bias (0-1)

    Returns:
        bool: True if lighting bias detected
    """
    if np.sum(mask) == 0:
        return False

    # Get masked pixels
    masked_pixels = rgb_img[mask]

    # Check for very dark or very bright regions
    mean_brightness = np.mean(masked_pixels) / 255.0

    if mean_brightness > bias_threshold or mean_brightness < (1 - bias_threshold):
        return True

    return False

def calculate_color_diversity(rgb_img, mask):
    """
    Calculate color diversity within a masked region

    Args:
        rgb_img: RGB image
        mask: Binary mask

    Returns:
        float: Diversity measure (0-1, higher = more diverse)
    """
    if np.sum(mask) == 0:
        return 0.0

    masked_pixels = rgb_img[mask]

    # Calculate standard deviation across color channels
    color_std = np.std(masked_pixels, axis=0)
    diversity = np.mean(color_std) / 255.0  # Normalize to 0-1

    return diversity

def classify_all_masks_warmth(rgb_img, masks, method='lab', **kwargs):
    """
    Classify warmth for all masks in an image

    Args:
        rgb_img: RGB image
        masks: List of binary masks
        method: Classification method ('lab' or 'hsv')
        **kwargs: Additional parameters for classification functions

    Returns:
        list: List of (warmth_class, analysis_dict) tuples
    """
    results = []

    for i, mask in enumerate(masks):
        print(f"   🎨 Analyzing color for mask {i+1}...")

        # Validate classification quality
        is_reliable, quality_issue = validate_classification_quality(rgb_img, mask)

        # Choose classification method
        if method == 'lab' and SKIMAGE_AVAILABLE:
            warmth_class, analysis = classify_mask_warmth_lab(rgb_img, mask, **kwargs)
        else:
            warmth_class, analysis = classify_mask_warmth_hsv(rgb_img, mask, **kwargs)

        # Add quality information
        analysis['is_reliable'] = is_reliable
        analysis['quality_issue'] = quality_issue if not is_reliable else 'none'
        analysis['mask_id'] = i

        # Print analysis
        print(f"      {'🔥' if warmth_class == 0 else '❄️ '} Classification: {analysis['classification']}")
        print(f"      📊 Pixels: {analysis['pixel_count']:,}")
        if 'warm_ratio' in analysis:
            print(f"      📈 Warm ratio: {analysis['warm_ratio']:.2f}")
        if 'median_chroma' in analysis:
            print(f"      🌈 Chroma: {analysis['median_chroma']:.2f}")

        results.append((warmth_class, analysis))

    return results

def associate_colors_with_strokes(sampling_results, color_classifications):
    """
    Associate color classifications with stroke sequences

    Args:
        sampling_results: Results from Step 6 sampling
        color_classifications: List of color classification results

    Returns:
        dict: Enhanced sampling results with color information
    """
    enhanced_results = sampling_results.copy()

    # Add color information to overall metadata
    enhanced_results['color_info'] = {
        'total_masks': len(color_classifications),
        'warm_masks': sum(1 for cls, _ in color_classifications if cls == 0),
        'cool_masks': sum(1 for cls, _ in color_classifications if cls == 1),
        'classification_method': color_classifications[0][1]['method'] if color_classifications else 'none'
    }

    # Associate colors with each graph's strokes
    for graph_idx, graph_result in enumerate(enhanced_results['sampled_graphs']):
        if graph_result is None or graph_idx >= len(color_classifications):
            continue

        warmth_class, color_analysis = color_classifications[graph_idx]

        # Add color information to graph metadata
        graph_result['color_info'] = {
            'warmth_class': warmth_class,
            'warmth_name': 'warm' if warmth_class == 0 else 'cool',
            'analysis': color_analysis
        }

        # Add color information to each stroke in all semantic phases
        for phase_name in ['boundary', 'internal', 'detail']:
            for stroke in graph_result[phase_name]:
                stroke['color_info'] = {
                    'warmth_class': warmth_class,
                    'warmth_name': 'warm' if warmth_class == 0 else 'cool',
                    'mask_id': graph_idx
                }

    return enhanced_results

def process_all_color_detection(rgb_img, masks, sampling_results, method='lab'):
    """
    Process all color detection for the complete pipeline

    Args:
        rgb_img: Original RGB image
        masks: List of segmentation masks
        sampling_results: Results from Step 6 sampling
        method: Classification method ('lab' or 'hsv')

    Returns:
        dict: Results with color classifications and overall statistics
    """
    print(f"   🎨 Color analysis method: {method.upper()}")
    print(f"   📊 Analyzing {len(masks)} masks...")

    # Classify warmth for all masks
    color_classifications = classify_all_masks_warmth(rgb_img, masks, method=method)

    # Associate with stroke sequences
    enhanced_results = associate_colors_with_strokes(sampling_results, color_classifications)

    # Calculate overall statistics
    total_classifications = len(color_classifications)
    warm_count = sum(1 for cls, _ in color_classifications if cls == 0)
    cool_count = total_classifications - warm_count
    warm_percentage = (warm_count / total_classifications * 100) if total_classifications > 0 else 0
    cool_percentage = (cool_count / total_classifications * 100) if total_classifications > 0 else 0

    # Calculate average confidence
    confidences = []
    for cls, analysis in color_classifications:
        if 'confidence' in analysis:
            confidences.append(analysis['confidence'])
        else:
            # Estimate confidence from warm_ratio
            warm_ratio = analysis.get('warm_ratio', 0.5)
            confidence = max(warm_ratio, 1.0 - warm_ratio)  # Distance from 0.5
            confidences.append(confidence)

    avg_confidence = np.mean(confidences) if confidences else 0.0
    method_used = color_classifications[0][1]['method'] if color_classifications else method.upper()

    # Print summary statistics
    print(f"   🔥 Warm regions: {warm_count}")
    print(f"   ❄️  Cool regions: {cool_count}")
    print(f"   📊 Method: {method_used}")

    # Create comprehensive results
    results = {
        'enhanced_sampling_results': enhanced_results,
        'color_classifications': color_classifications,
        'overall_statistics': {
            'successful_classifications': total_classifications,
            'warm_strokes': warm_count,
            'cool_strokes': cool_count,
            'warm_percentage': warm_percentage,
            'cool_percentage': cool_percentage,
            'average_confidence': avg_confidence,
            'method_used': method_used
        }
    }

    return results

def save_color_detection_results(results, rgb_img, masks, output_dir="results/step7_color_detection"):
    """
    Save color detection visualization and results with consistent scaling

    Args:
        results: Color detection results with classifications and statistics
        rgb_img: Original RGB image
        masks: Segmentation masks
        output_dir: Directory to save results

    Returns:
        str: Path to saved visualization
    """
    import sys
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    sys.path.append(str(Path(__file__).parent.parent))
    
    from visualization_utils import (
        get_consistent_figure_layout, save_visualization_with_timestamp,
        hide_unused_subplots, STANDARD_TARGET_SIZE
    )

    # Create consistent figure layout
    fig, axes = get_consistent_figure_layout()

    enhanced_results = results['enhanced_sampling_results']
    valid_graphs = [g for g in enhanced_results['sampled_graphs'] if g is not None]

    # Color scheme for warm/cool
    warm_color = '#FF6B6B'  # Warm red-orange
    cool_color = '#4ECDC4'  # Cool teal-blue

    for i, graph_result in enumerate(valid_graphs[:6]):
        if i >= 6:
            break

        ax = axes[i]

        # Get color information
        color_info = graph_result.get('color_info', {})
        warmth_class = color_info.get('warmth_class', 1)
        warmth_name = color_info.get('warmth_name', 'unknown')

        # Choose colors based on warmth
        stroke_color = warm_color if warmth_class == 0 else cool_color

        # Scale millimeter coordinates to consistent display size
        canvas_size = enhanced_results['scale_info']['canvas_size_mm']
        mm_to_display_scale = STANDARD_TARGET_SIZE / canvas_size

        # Plot canvas boundaries (scaled to display size)
        canvas_display = STANDARD_TARGET_SIZE
        ax.plot([0, canvas_display, canvas_display, 0, 0],
               [0, 0, canvas_display, canvas_display, 0],
               'k--', alpha=0.3, linewidth=1)

        # Plot strokes with warm/cool coloring and consistent scaling
        for phase_name in ['boundary', 'internal', 'detail']:
            phase_strokes = graph_result[phase_name]
            alpha_values = {'boundary': 1.0, 'internal': 0.8, 'detail': 0.6}
            width_values = {'boundary': 3, 'internal': 2, 'detail': 1}

            for stroke in phase_strokes:
                if stroke['success'] and len(stroke['sampled_points_mm']) > 1:
                    points_mm = stroke['sampled_points_mm']
                    alpha = alpha_values[phase_name]
                    width = width_values[phase_name]

                    # Scale points to display coordinates
                    points_display = points_mm * mm_to_display_scale

                    # Plot stroke path
                    ax.plot(points_display[:, 0], points_display[:, 1],
                           color=stroke_color, linewidth=width, alpha=alpha)

                    # Mark start/end for open strokes
                    if not stroke['is_closed_loop']:
                        ax.scatter(points_display[0, 0], points_display[0, 1],
                                 c='green', s=15, marker='o', zorder=6, alpha=0.8)
                        ax.scatter(points_display[-1, 0], points_display[-1, 1],
                                 c='red', s=15, marker='s', zorder=6, alpha=0.8)

        # Set title with color classification
        meta = graph_result['metadata']
        analysis = color_info.get('analysis', {})

        title = f'Graph {i+1}: {warmth_name.title()} Colors\n'
        title += f'{meta["successful_samplings"]} strokes, {meta["total_length_mm"]:.1f}mm\n'

        if 'warm_ratio' in analysis:
            title += f'Warm ratio: {analysis["warm_ratio"]:.2f}'

        ax.set_title(title, fontsize=10,
                    color=warm_color if warmth_class == 0 else cool_color,
                    fontweight='bold')
        ax.set_xlim(0, STANDARD_TARGET_SIZE)
        ax.set_ylim(0, STANDARD_TARGET_SIZE)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add scaled mm labels
        mm_ticks = np.linspace(0, canvas_size, 5)
        display_ticks = mm_ticks * mm_to_display_scale
        ax.set_xticks(display_ticks)
        ax.set_yticks(display_ticks)
        ax.set_xticklabels([f'{mm:.0f}' for mm in mm_ticks])
        ax.set_yticklabels([f'{mm:.0f}' for mm in mm_ticks])
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.invert_yaxis()  # Match image coordinates

    # Hide unused subplots
    hide_unused_subplots(axes, len(valid_graphs))

    # Add legend
    legend_elements = [
        Patch(facecolor=warm_color, label='Warm Colors (Red/Orange/Yellow)'),
        Patch(facecolor=cool_color, label='Cool Colors (Blue/Green/Purple)'),
        plt.scatter([], [], c='green', s=15, marker='o', label='Start point'),
        plt.scatter([], [], c='red', s=15, marker='s', label='End point'),
        plt.Line2D([0], [0], color='black', lw=1, linestyle='--', label='Canvas bounds')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)

    # Add overall statistics
    color_info = enhanced_results['color_info']
    fig_title = f'Step 7: Color Classification\n'
    fig_title += f'🔥 {color_info["warm_masks"]} Warm • ❄️ {color_info["cool_masks"]} Cool • '
    fig_title += f'Method: {color_info["classification_method"].upper()}'

    plt.suptitle(fig_title, fontsize=16)
    plt.tight_layout()

    # Save with consistent timestamp and path handling
    return save_visualization_with_timestamp(fig, output_dir, 'color_detection_results')