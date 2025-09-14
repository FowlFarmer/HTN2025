#!/usr/bin/env python3
"""
Test script to visualize and compare different K-means clustering methods
with detailed results saved to step1_preprocessing directory
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append('..')

from image_preprocessing import preprocess_image, enhanced_color_clustering


def test_all_clustering_methods(image_path):
    """
    Test all available clustering methods and save detailed comparisons
    
    Args:
        image_path: Path to test image
    """
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"🎨 Testing clustering methods on {Path(image_path).name}")
    
    # Load and resize image (same as preprocessing)
    img = cv2.imread(image_path)[:, :, ::-1]  # BGR -> RGB
    h, w = img.shape[:2]
    scale = 1024 / max(h, w) if max(h, w) > 1024 else 1.0
    img_resized = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    
    # Test different methods
    methods = [
        ('basic_kmeans', {'n_colors': 8}),
        ('adaptive_spatial', {'spatial_weight': 0.3, 'preserve_boundaries': True}),
        ('region_aware', {'n_colors': 8, 'region_coherence': 0.5}),
    ]
    
    # Create comprehensive comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("../results/step1_preprocessing")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create main comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Show original
    axes[0].imshow(img_resized)
    axes[0].set_title('Original (Resized)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Test each method
    results = {}
    for i, (method, params) in enumerate(methods):
        print(f"   🔄 Testing {method}...")
        
        try:
            clustered = enhanced_color_clustering(img_resized, method=method, **params)
            results[method] = clustered
            
            # Show in comparison
            axes[i+1].imshow(clustered)
            axes[i+1].set_title(f'{method.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[i+1].axis('off')
            
            # Count unique colors
            unique_colors = len(np.unique(clustered.reshape(-1, 3), axis=0))
            axes[i+1].text(0.02, 0.98, f'{unique_colors} colors', 
                          transform=axes[i+1].transAxes, fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          verticalalignment='top')
            
        except Exception as e:
            print(f"   ❌ Error with {method}: {e}")
            axes[i+1].text(0.5, 0.5, f'Error: {method}', ha='center', va='center')
            axes[i+1].set_xlim(0, 1)
            axes[i+1].set_ylim(0, 1)
    
    # Overall title
    image_name = Path(image_path).stem
    plt.suptitle(f'K-means Clustering Method Comparison - {image_name}', fontsize=18)
    plt.tight_layout()
    
    # Save main comparison
    comparison_path = results_dir / f"method_comparison_{image_name}_{timestamp}.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved method comparison: {comparison_path}")
    
    # Create detailed color palette comparisons
    create_palette_comparison(img_resized, results, image_name, timestamp, results_dir)
    
    # Create parameter variation tests
    test_parameter_variations(img_resized, image_name, timestamp, results_dir)
    
    print(f"✅ All clustering tests completed! Check {results_dir} for results.")


def create_palette_comparison(original, results, image_name, timestamp, results_dir):
    """
    Create detailed color palette comparison
    """
    print("   🎨 Creating color palette comparison...")
    
    fig, axes = plt.subplots(len(results) + 1, 2, figsize=(12, 3 * (len(results) + 1)))
    
    # Original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Original palette (sample colors)
    original_colors = sample_colors_from_image(original, n_samples=20)
    show_color_palette(axes[0, 1], original_colors, 'Original Colors (Sample)')
    
    # Each method
    for i, (method, clustered) in enumerate(results.items()):
        axes[i+1, 0].imshow(clustered)
        axes[i+1, 0].set_title(f'{method.replace("_", " ").title()}', fontweight='bold')
        axes[i+1, 0].axis('off')
        
        # Extract palette
        unique_colors = np.unique(clustered.reshape(-1, 3), axis=0)
        show_color_palette(axes[i+1, 1], unique_colors, f'{method} Palette ({len(unique_colors)} colors)')
    
    plt.suptitle(f'Color Palette Analysis - {image_name}', fontsize=16)
    plt.tight_layout()
    
    palette_path = results_dir / f"palette_comparison_{image_name}_{timestamp}.png"
    plt.savefig(palette_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved palette comparison: {palette_path}")


def sample_colors_from_image(image, n_samples=20):
    """
    Sample representative colors from original image
    """
    pixels = image.reshape(-1, 3)
    
    # Use K-means to find representative colors
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=min(n_samples, len(np.unique(pixels, axis=0))), random_state=42)
    kmeans.fit(pixels)
    
    return kmeans.cluster_centers_.astype(np.uint8)


def show_color_palette(ax, colors, title):
    """
    Display color palette as horizontal bars
    """
    n_colors = len(colors)
    
    if n_colors == 0:
        ax.text(0.5, 0.5, 'No colors found', ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # Sort by brightness
        brightness = np.sum(colors, axis=1)
        sorted_indices = np.argsort(brightness)
        sorted_colors = colors[sorted_indices]
        
        # Create color bars
        for i, color in enumerate(sorted_colors):
            bar_height = 1.0 / n_colors
            rect = plt.Rectangle((0, i * bar_height), 1, bar_height, 
                               facecolor=color/255.0, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # Add RGB values
            text_color = 'white' if np.sum(color) < 384 else 'black'
            if n_colors <= 12:  # Only show text if not too crowded
                ax.text(0.5, i * bar_height + bar_height/2, 
                       f'({color[0]},{color[1]},{color[2]})',
                       ha='center', va='center', fontsize=8, color=text_color, weight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')


def test_parameter_variations(original, image_name, timestamp, results_dir):
    """
    Test different parameter settings for adaptive spatial clustering
    """
    print("   🔧 Testing parameter variations...")
    
    # Test spatial weight variations
    spatial_weights = [0.0, 0.2, 0.4, 0.6]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, weight in enumerate(spatial_weights):
        try:
            clustered = enhanced_color_clustering(
                original, 
                method='adaptive_spatial',
                spatial_weight=weight,
                preserve_boundaries=True
            )
            
            axes[i].imshow(clustered)
            unique_colors = len(np.unique(clustered.reshape(-1, 3), axis=0))
            axes[i].set_title(f'Spatial Weight: {weight}\n({unique_colors} colors)', fontweight='bold')
            axes[i].axis('off')
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
            axes[i].set_title(f'Spatial Weight: {weight} (Error)')
    
    plt.suptitle(f'Spatial Weight Parameter Test - {image_name}', fontsize=16)
    plt.tight_layout()
    
    param_path = results_dir / f"parameter_test_{image_name}_{timestamp}.png"
    plt.savefig(param_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved parameter test: {param_path}")


if __name__ == "__main__":
    # Test with available images
    test_images = [
        "../examples/tower.jpg",
        "../examples/monet.jpeg", 
        "../examples/monalisa.jpg",
        "../examples/e7.jpg"
    ]
    
    print("🎨 K-means Clustering Visualization Test")
    print("=" * 50)
    
    found_image = False
    for image_path in test_images:
        if Path(image_path).exists():
            test_all_clustering_methods(image_path)
            found_image = True
            break
    
    if not found_image:
        print("❌ No test images found in examples/ directory")
        print("Available test images should be:")
        for img in test_images:
            print(f"   - {img}")
    
    print("\n✅ Test completed! Check results/step1_preprocessing/ for detailed visualizations.")
