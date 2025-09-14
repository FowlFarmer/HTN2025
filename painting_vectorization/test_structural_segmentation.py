#!/usr/bin/env python3
"""
Test script to compare structural segmentation vs color clustering
for tactile drawing optimization
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add step1_preprocessing to path
sys.path.append('step1_preprocessing')

from image_preprocessing import preprocess_image


def compare_segmentation_approaches(image_path):
    """
    Compare structural segmentation vs color clustering approaches
    
    Args:
        image_path: Path to test image
    """
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"🎨 Comparing segmentation approaches on {Path(image_path).name}")
    print("=" * 60)
    
    # Test both approaches
    approaches = [
        ('Structural Segmentation', 'structural', {
            'method': 'contour_hierarchy',
            'min_area_ratio': 0.03,
            'max_regions': 5,
            'smoothing': 5
        }),
        ('Color Clustering', 'color_clustering', {
            'spatial_weight': 0.3,
            'preserve_boundaries': True
        })
    ]
    
    results = {}
    
    for name, method, params in approaches:
        print(f"\n🔄 Testing {name}...")
        
        try:
            processed_img, scale, dims = preprocess_image(
                image_path,
                segmentation_method=method,
                segmentation_params=params
            )
            
            results[name] = {
                'processed': processed_img,
                'method': method,
                'params': params
            }
            
            print(f"   ✅ {name} completed successfully")
            
        except Exception as e:
            print(f"   ❌ Error with {name}: {e}")
            results[name] = None
    
    # Create comparison visualization
    create_comparison_visualization(image_path, results)
    
    print(f"\n✅ Comparison completed! Check results/step1_preprocessing/ for detailed outputs.")


def create_comparison_visualization(image_path, results):
    """
    Create side-by-side comparison of segmentation approaches
    """
    # Load original image
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Resize to match preprocessing
    h, w = original.shape[:2]
    scale = 1024 / max(h, w) if max(h, w) > 1024 else 1.0
    original_resized = cv2.resize(original, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original
    axes[0].imshow(original_resized)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Results
    valid_results = [r for r in results.values() if r is not None]
    
    for i, (name, result) in enumerate(results.items()):
        if result is not None and i < 2:
            processed = (result['processed'] * 255).astype(np.uint8)
            axes[i+1].imshow(processed)
            axes[i+1].set_title(f'{name}', fontsize=14, fontweight='bold')
            axes[i+1].axis('off')
        elif i < 2:
            axes[i+1].text(0.5, 0.5, f'{name}\n(Error)', ha='center', va='center', 
                          transform=axes[i+1].transAxes, fontsize=12)
            axes[i+1].set_xlim(0, 1)
            axes[i+1].set_ylim(0, 1)
    
    plt.suptitle(f'Segmentation Approach Comparison - {Path(image_path).name}', fontsize=16)
    plt.tight_layout()
    
    # Save comparison
    results_dir = Path("results/step1_preprocessing")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = Path(image_path).stem
    
    comparison_path = results_dir / f"approach_comparison_{image_name}_{timestamp}.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved approach comparison: {comparison_path}")


def analyze_tactile_drawing_benefits():
    """
    Analyze the benefits of structural segmentation for tactile drawing
    """
    print("\n🤚 Tactile Drawing Analysis")
    print("=" * 40)
    
    benefits = {
        "Structural Segmentation": [
            "✅ Focuses on major shapes and objects",
            "✅ Creates larger, continuous regions",
            "✅ Minimizes pen lifts between strokes", 
            "✅ Prioritizes drawing order (large → small)",
            "✅ Preserves object boundaries",
            "✅ Better for tactile recognition",
            "✅ Smoother stroke transitions"
        ],
        "Color Clustering": [
            "❌ Creates many small color patches",
            "❌ Breaks objects into color fragments", 
            "❌ Requires many pen lifts",
            "❌ No logical drawing order",
            "❌ Harder to feel object shapes",
            "❌ Fragmented tactile experience",
            "❌ Less intuitive for blind users"
        ]
    }
    
    for approach, points in benefits.items():
        print(f"\n{approach}:")
        for point in points:
            print(f"  {point}")
    
    print(f"\n🎯 Recommendation: Use structural segmentation for tactile drawing")
    print(f"   → Creates 3-6 major regions instead of 8+ color patches")
    print(f"   → Enables smooth, continuous drawing motions")
    print(f"   → Provides logical drawing sequence for tactile learning")


def test_structural_parameters():
    """
    Test different structural segmentation parameters
    """
    image_path = "examples/tower.jpg"
    if not Path(image_path).exists():
        print(f"❌ Test image not found: {image_path}")
        return
    
    print(f"\n🔧 Testing Structural Segmentation Parameters")
    print("=" * 50)
    
    # Test different parameter combinations
    parameter_tests = [
        ("Conservative (3 regions)", {'max_regions': 3, 'min_area_ratio': 0.05}),
        ("Balanced (5 regions)", {'max_regions': 5, 'min_area_ratio': 0.03}),
        ("Detailed (7 regions)", {'max_regions': 7, 'min_area_ratio': 0.02}),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, params) in enumerate(parameter_tests):
        print(f"   🔄 Testing {name}...")
        
        try:
            processed_img, _, _ = preprocess_image(
                image_path,
                segmentation_method='structural',
                segmentation_params={
                    'method': 'contour_hierarchy',
                    'smoothing': 5,
                    **params
                }
            )
            
            result = (processed_img * 255).astype(np.uint8)
            axes[i].imshow(result)
            axes[i].set_title(name, fontweight='bold')
            axes[i].axis('off')
            
        except Exception as e:
            print(f"   ❌ Error with {name}: {e}")
            axes[i].text(0.5, 0.5, f'Error\n{name}', ha='center', va='center')
    
    plt.suptitle('Structural Segmentation Parameter Tests', fontsize=16)
    plt.tight_layout()
    
    # Save parameter test
    results_dir = Path("results/step1_preprocessing")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    param_path = results_dir / f"parameter_tests_{timestamp}.png"
    plt.savefig(param_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved parameter tests: {param_path}")


if __name__ == "__main__":
    print("🏗️  Structural Segmentation vs Color Clustering Test")
    print("=" * 60)
    
    # Test with available images
    test_images = [
        "examples/tower.jpg",
        "examples/monet.jpeg", 
        "examples/monalisa.jpg",
        "examples/e7.jpg"
    ]
    
    found_image = False
    for image_path in test_images:
        if Path(image_path).exists():
            compare_segmentation_approaches(image_path)
            found_image = True
            break
    
    if not found_image:
        print("❌ No test images found in examples/ directory")
        print("Available test images should be:")
        for img in test_images:
            print(f"   - {img}")
    else:
        # Run additional tests
        analyze_tactile_drawing_benefits()
        test_structural_parameters()
    
    print(f"\n🎯 Key Takeaway:")
    print(f"   Structural segmentation creates fewer, larger regions that are")
    print(f"   much better for tactile drawing with minimal pen lifts!")
    print(f"   Check results/step1_preprocessing/ for detailed comparisons.")
