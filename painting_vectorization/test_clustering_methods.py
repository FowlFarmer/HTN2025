#!/usr/bin/env python3
"""
Test script to compare different color clustering methods
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from step1_preprocessing.image_preprocessing import preprocess_image

def compare_clustering_methods(image_path):
    """
    Compare different clustering methods on the same image
    
    Args:
        image_path: Path to test image
    """
    # Load original image for comparison
    original = cv2.imread(image_path)
    if original is None:
        print(f"Could not load image: {image_path}")
        return
    
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Test different clustering methods
    methods = [
        ('Original', None, {}),
        ('Basic K-means', 'basic_kmeans', {'n_colors': 8}),
        ('Adaptive Spatial', 'adaptive_spatial', {'spatial_weight': 0.3}),
        ('Region Aware', 'region_aware', {'n_colors': 8, 'region_coherence': 0.5}),
    ]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (name, method, params) in enumerate(methods):
        if method is None:
            # Show original
            # Resize to match preprocessing
            h, w = original.shape[:2]
            scale = 1024 / max(h, w) if max(h, w) > 1024 else 1.0
            resized = cv2.resize(original, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            result = resized
        else:
            # Apply preprocessing with specific method
            try:
                processed, _, _ = preprocess_image(image_path, clustering_method=method, clustering_params=params)
                result = (processed * 255).astype(np.uint8)
            except Exception as e:
                print(f"Error with {name}: {e}")
                result = original
        
        axes[i].imshow(result)
        axes[i].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle(f'Color Clustering Comparison: {Path(image_path).name}', fontsize=16)
    plt.tight_layout()
    
    # Save comparison
    output_path = f"clustering_comparison_{Path(image_path).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved comparison: {output_path}")
    
    plt.show()

def test_parameter_variations():
    """
    Test different parameter settings for adaptive spatial clustering
    """
    image_path = "examples/tower.jpg"
    if not Path(image_path).exists():
        print(f"Test image not found: {image_path}")
        return
    
    # Test different spatial weights
    spatial_weights = [0.0, 0.2, 0.4, 0.6]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, weight in enumerate(spatial_weights):
        try:
            processed, _, _ = preprocess_image(
                image_path, 
                clustering_method='adaptive_spatial',
                clustering_params={'spatial_weight': weight, 'preserve_boundaries': True}
            )
            result = (processed * 255).astype(np.uint8)
            
            axes[i].imshow(result)
            axes[i].set_title(f'Spatial Weight: {weight}', fontsize=11)
            axes[i].axis('off')
        except Exception as e:
            print(f"Error with spatial weight {weight}: {e}")
    
    plt.suptitle('Spatial Weight Comparison (Adaptive Spatial Clustering)', fontsize=14)
    plt.tight_layout()
    plt.savefig("spatial_weight_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Test with available images
    test_images = [
        "examples/tower.jpg",
        "examples/monet.jpeg", 
        "examples/monalisa.jpg"
    ]
    
    for image_path in test_images:
        if Path(image_path).exists():
            print(f"\n🎨 Testing clustering methods on {image_path}")
            compare_clustering_methods(image_path)
            break
    else:
        print("No test images found in examples/ directory")
    
    # Test parameter variations
    print("\n🔧 Testing parameter variations...")
    test_parameter_variations()
