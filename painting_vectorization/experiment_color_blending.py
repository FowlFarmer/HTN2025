#!/usr/bin/env python3
"""
Color Blending Experiment Script

Independent script to experiment with color blending techniques
for identifying major compositional elements without affecting the main pipeline.

Usage:
    python experiment_color_blending.py examples/monet.jpeg
    python experiment_color_blending.py examples/monet.jpeg --method kmeans --colors 8
    python experiment_color_blending.py examples/monet.jpeg --method tolerance --tolerance 30
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime
from sklearn.cluster import KMeans

def load_and_resize_image(image_path, max_size=1024):
    """Load and resize image for processing"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize if too large
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return img

def blend_colors_kmeans(image, n_colors=8):
    """
    Blend similar colors using K-means clustering
    
    Args:
        image: Input RGB image
        n_colors: Number of color clusters
    
    Returns:
        Blended image with reduced colors
    """
    # Reshape image to list of pixels
    data = image.reshape((-1, 3))
    data = np.float32(data)
    
    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8 and reshape
    centers = np.uint8(centers)
    blended_data = centers[labels.flatten()]
    blended_image = blended_data.reshape(image.shape)
    
    return blended_image, centers

def blend_colors_tolerance(image, tolerance=30):
    """
    Blend similar colors using color tolerance in LAB color space
    
    Args:
        image: Input RGB image
        tolerance: Color difference tolerance (0-100)
    
    Returns:
        Blended image with similar colors merged
    """
    # Convert to LAB color space for better perceptual distance
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Find unique colors and group similar ones
    h, w, c = lab_image.shape
    lab_pixels = lab_image.reshape(-1, 3)
    
    # Group similar colors
    color_groups = []
    processed = np.zeros(len(lab_pixels), dtype=bool)
    
    for i, pixel in enumerate(lab_pixels):
        if processed[i]:
            continue
            
        # Find all pixels similar to this one
        distances = np.linalg.norm(lab_pixels - pixel, axis=1)
        similar_mask = distances <= tolerance
        
        # Average the similar colors
        similar_pixels = lab_pixels[similar_mask]
        avg_color = np.mean(similar_pixels, axis=0).astype(np.uint8)
        
        # Mark as processed and store group
        processed[similar_mask] = True
        color_groups.append((similar_mask, avg_color))
    
    # Apply color groups
    blended_lab = lab_pixels.copy()
    for mask, avg_color in color_groups:
        blended_lab[mask] = avg_color
    
    # Reshape and convert back to RGB
    blended_lab_image = blended_lab.reshape(h, w, c)
    blended_image = cv2.cvtColor(blended_lab_image, cv2.COLOR_LAB2RGB)
    
    return blended_image, len(color_groups)

def blend_colors_adaptive(image, initial_colors=16, merge_threshold=20):
    """
    Adaptive color blending: start with K-means then merge similar clusters
    
    Args:
        image: Input RGB image
        initial_colors: Initial number of K-means clusters
        merge_threshold: Threshold for merging similar clusters
    
    Returns:
        Blended image with adaptively merged colors
    """
    # Step 1: Initial K-means clustering
    blended_kmeans, initial_centers = blend_colors_kmeans(image, initial_colors)
    
    # Step 2: Convert centers to LAB for better distance calculation
    centers_lab = cv2.cvtColor(initial_centers.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    
    # Step 3: Merge similar clusters
    merged_centers = []
    used = np.zeros(len(centers_lab), dtype=bool)
    
    for i, center in enumerate(centers_lab):
        if used[i]:
            continue
            
        # Find similar centers
        distances = np.linalg.norm(centers_lab - center, axis=1)
        similar_mask = distances <= merge_threshold
        
        # Average similar centers
        similar_centers = centers_lab[similar_mask]
        merged_center = np.mean(similar_centers, axis=0)
        merged_centers.append(merged_center)
        
        used[similar_mask] = True
    
    # Step 4: Apply merged colors
    merged_centers = np.array(merged_centers)
    merged_centers_rgb = cv2.cvtColor(merged_centers.reshape(1, -1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    
    # Reassign pixels to nearest merged center
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_pixels = lab_image.reshape(-1, 3)
    
    # Find nearest merged center for each pixel
    final_pixels = np.zeros_like(lab_pixels)
    for i, pixel in enumerate(lab_pixels):
        distances = np.linalg.norm(merged_centers - pixel, axis=1)
        nearest_idx = np.argmin(distances)
        final_pixels[i] = merged_centers[nearest_idx]
    
    # Convert back to RGB
    final_lab_image = final_pixels.reshape(lab_image.shape)
    final_image = cv2.cvtColor(final_lab_image, cv2.COLOR_LAB2RGB)
    
    return final_image, len(merged_centers)

def apply_smoothing(image, method='bilateral', **kwargs):
    """Apply smoothing to reduce texture while preserving edges"""
    if method == 'bilateral':
        d = kwargs.get('d', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    elif method == 'edge_preserving':
        flags = kwargs.get('flags', 2)
        sigma_s = kwargs.get('sigma_s', 50)
        sigma_r = kwargs.get('sigma_r', 0.4)
        return cv2.edgePreservingFilter(image, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r)
    
    elif method == 'gaussian':
        kernel_size = kwargs.get('kernel_size', 5)
        sigma = kwargs.get('sigma', 1.0)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    else:
        return image

def experiment_color_blending(image_path, method='kmeans', **kwargs):
    """
    Main experiment function
    
    Args:
        image_path: Path to input image
        method: Blending method ('kmeans', 'tolerance', 'adaptive')
        **kwargs: Method-specific parameters
    
    Returns:
        Dictionary with results
    """
    # Load image
    original = load_and_resize_image(image_path)
    
    # Apply pre-smoothing if requested
    smoothed = original
    if kwargs.get('pre_smooth', False):
        smoothing_method = kwargs.get('smooth_method', 'bilateral')
        smoothed = apply_smoothing(original, smoothing_method)
    
    # Apply color blending
    start_time = datetime.now()
    
    if method == 'kmeans':
        n_colors = kwargs.get('colors', 8)
        blended, extra_info = blend_colors_kmeans(smoothed, n_colors)
        method_info = f"K-means with {n_colors} colors"
        
    elif method == 'tolerance':
        tolerance = kwargs.get('tolerance', 30)
        blended, extra_info = blend_colors_tolerance(smoothed, tolerance)
        method_info = f"Tolerance-based with threshold {tolerance} ({extra_info} groups)"
        
    elif method == 'adaptive':
        initial_colors = kwargs.get('initial_colors', 16)
        merge_threshold = kwargs.get('merge_threshold', 20)
        blended, extra_info = blend_colors_adaptive(smoothed, initial_colors, merge_threshold)
        method_info = f"Adaptive: {initial_colors}→{extra_info} colors"
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    # Apply post-smoothing if requested
    final = blended
    if kwargs.get('post_smooth', False):
        smoothing_method = kwargs.get('smooth_method', 'bilateral')
        final = apply_smoothing(blended, smoothing_method)
    
    return {
        'original': original,
        'smoothed': smoothed,
        'blended': blended,
        'final': final,
        'method_info': method_info,
        'processing_time': processing_time,
        'extra_info': extra_info
    }

def visualize_results(results, save_path=None):
    """Visualize the color blending results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(results['original'])
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Smoothed (if different from original)
    if not np.array_equal(results['original'], results['smoothed']):
        axes[0, 1].imshow(results['smoothed'])
        axes[0, 1].set_title('Pre-smoothed')
    else:
        axes[0, 1].imshow(results['original'])
        axes[0, 1].set_title('Original (no pre-smoothing)')
    axes[0, 1].axis('off')
    
    # Color blended
    axes[1, 0].imshow(results['blended'])
    axes[1, 0].set_title(f'Color Blended\n{results["method_info"]}')
    axes[1, 0].axis('off')
    
    # Final result
    axes[1, 1].imshow(results['final'])
    axes[1, 1].set_title(f'Final Result\n({results["processing_time"]:.3f}s)')
    axes[1, 1].axis('off')
    
    plt.suptitle('Color Blending Experiment Results', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Color Blending Experiment')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--method', choices=['kmeans', 'tolerance', 'adaptive'], 
                       default='kmeans', help='Blending method')
    parser.add_argument('--colors', type=int, default=8, 
                       help='Number of colors for K-means method')
    parser.add_argument('--tolerance', type=float, default=30, 
                       help='Color tolerance for tolerance method')
    parser.add_argument('--initial-colors', type=int, default=16, 
                       help='Initial colors for adaptive method')
    parser.add_argument('--merge-threshold', type=float, default=20, 
                       help='Merge threshold for adaptive method')
    parser.add_argument('--pre-smooth', action='store_true', 
                       help='Apply smoothing before color blending')
    parser.add_argument('--post-smooth', action='store_true', 
                       help='Apply smoothing after color blending')
    parser.add_argument('--smooth-method', choices=['bilateral', 'edge_preserving', 'gaussian'], 
                       default='bilateral', help='Smoothing method')
    parser.add_argument('--save', help='Save visualization to file')
    
    args = parser.parse_args()
    
    # Prepare parameters
    kwargs = {
        'colors': args.colors,
        'tolerance': args.tolerance,
        'initial_colors': args.initial_colors,
        'merge_threshold': args.merge_threshold,
        'pre_smooth': args.pre_smooth,
        'post_smooth': args.post_smooth,
        'smooth_method': args.smooth_method
    }
    
    # Run experiment
    print(f"🎨 Running color blending experiment on {args.image_path}")
    print(f"📋 Method: {args.method}")
    print(f"⚙️  Parameters: {kwargs}")
    
    try:
        results = experiment_color_blending(args.image_path, args.method, **kwargs)
        
        # Show results
        print(f"\n✅ Experiment complete!")
        print(f"🕒 Processing time: {results['processing_time']:.3f}s")
        print(f"📊 Method info: {results['method_info']}")
        
        # Visualize
        save_path = args.save
        if not save_path and args.method:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"color_blend_experiment_{args.method}_{timestamp}.png"
        
        visualize_results(results, save_path)
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
