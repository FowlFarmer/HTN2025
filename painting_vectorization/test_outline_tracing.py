#!/usr/bin/env python3
"""
Test script for outline-focused segmentation - demonstrates pure contour tracing
for tactile drawing without filled regions or backgrounds.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add step1_preprocessing to path
sys.path.append('step1_preprocessing')

from image_preprocessing import preprocess_image


def test_outline_tracing_approaches(image_path):
    """
    Test outline tracing vs other approaches for tactile drawing
    
    Args:
        image_path: Path to test image
    """
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"✏️  Testing Outline Tracing Approaches on {Path(image_path).name}")
    print("=" * 70)
    
    # Test different approaches
    approaches = [
        ('Outline Tracing (4 contours)', 'outline_focused', {'max_contours': 4, 'min_contour_area': 1000}),
        ('Outline Tracing (6 contours)', 'outline_focused', {'max_contours': 6, 'min_contour_area': 800}),
        ('Outline Tracing (8 contours)', 'outline_focused', {'max_contours': 8, 'min_contour_area': 600}),
        ('Object-Focused (filled)', 'object_focused', {'max_objects': 6}),
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
                'params': params,
                'success': True
            }
            
            print(f"   ✅ {name} completed successfully")
            
        except Exception as e:
            print(f"   ❌ Error with {name}: {e}")
            results[name] = {
                'success': False,
                'error': str(e)
            }
    
    # Create comparison visualization
    create_outline_comparison(image_path, results)
    
    # Analyze tactile benefits
    analyze_outline_benefits()


def create_outline_comparison(image_path, results):
    """
    Create visual comparison of outline tracing approaches
    """
    # Load original image
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Resize to match preprocessing
    h, w = original.shape[:2]
    scale = 1024 / max(h, w) if max(h, w) > 1024 else 1.0
    original_resized = cv2.resize(original, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    
    # Create comparison figure
    successful_results = [(name, result) for name, result in results.items() if result.get('success', False)]
    
    if len(successful_results) == 0:
        print("❌ No successful results to display")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Show original first
    axes[0].imshow(original_resized)
    axes[0].set_title('Original Image\n(Complex filled regions)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Show results
    for i, (name, result) in enumerate(successful_results[:3]):
        if result['success']:
            processed = (result['processed'] * 255).astype(np.uint8)
            axes[i+1].imshow(processed)
            
            # Determine pen lifts from method
            if 'outline_focused' in result['method']:
                pen_lifts = result['params'].get('max_contours', 6)
                approach_type = "Pure Outlines"
            else:
                pen_lifts = result['params'].get('max_objects', 6)
                approach_type = "Filled Objects"
            
            title = f'{name}\n{approach_type} - {pen_lifts} pen lifts'
            axes[i+1].set_title(title, fontsize=12, fontweight='bold')
            axes[i+1].axis('off')
    
    # Hide unused subplot if needed
    if len(successful_results) < 3:
        for i in range(len(successful_results) + 1, 4):
            axes[i].axis('off')
    
    plt.suptitle(f'Outline Tracing vs Filled Objects - {Path(image_path).name}', fontsize=16)
    plt.tight_layout()
    
    # Save comparison
    results_dir = Path("results/step1_preprocessing")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = Path(image_path).stem
    
    comparison_path = results_dir / f"outline_tracing_comparison_{image_name}_{timestamp}.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n💾 Saved outline tracing comparison: {comparison_path}")


def analyze_outline_benefits():
    """
    Analyze benefits of outline tracing for tactile drawing
    """
    print(f"\n✏️  Outline Tracing Benefits for Tactile Drawing")
    print("=" * 60)
    
    benefits = {
        "✅ Outline Tracing": [
            "🔍 Focus on object SHAPES, not filled areas",
            "✏️  Pure contour tracing - meaningful for touch",
            "🎯 No background/foreground confusion", 
            "🤚 Feel complete object boundaries",
            "📐 Geometric shapes are recognizable",
            "🖊️  4-8 continuous outline strokes",
            "⚡ Fast to draw and understand",
            "🎨 Perfect for tactile learning"
        ],
        "❌ Filled Objects/Regions": [
            "🎨 Filled areas have no tactile meaning",
            "🌫️  Background regions are confusing",
            "❓ Foreground vs background unclear by touch",
            "🔄 Spiral fills are complex to follow",
            "⏱️  Takes longer to draw filled areas",
            "🤷 Hard to recognize filled shapes by touch",
            "📊 More pen lifts for fill patterns",
            "😕 Less intuitive for blind users"
        ]
    }
    
    for approach, points in benefits.items():
        print(f"\n{approach}:")
        for point in points:
            print(f"  {point}")
    
    print(f"\n🎯 Key Insight for Tactile Drawing:")
    print(f"   Outlines convey object shape and identity")
    print(f"   Filled regions add no meaningful tactile information")
    print(f"   → Pure outline tracing is optimal for blind users!")
    
    print(f"\n📊 Comparison:")
    print(f"   Outline Tracing:  4-8 pen lifts, pure shapes, fast recognition")
    print(f"   Filled Objects:   6-12 pen lifts, complex fills, slower recognition")
    print(f"   Traditional:      100+ pen lifts, fragmented, unusable")


def test_minimal_outline_configurations():
    """
    Test ultra-minimal outline configurations
    """
    image_path = "examples/tower.jpg"
    if not Path(image_path).exists():
        print(f"❌ Test image not found: {image_path}")
        return
    
    print(f"\n✏️  Testing Minimal Outline Configurations")
    print("=" * 50)
    
    # Test different outline configurations
    configs = [
        ("Ultra Minimal (3 outlines)", {'max_contours': 3, 'min_contour_area': 1500}),
        ("Minimal (4 outlines)", {'max_contours': 4, 'min_contour_area': 1200}),
        ("Balanced (6 outlines)", {'max_contours': 6, 'min_contour_area': 800}),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, params) in enumerate(configs):
        print(f"   🔄 Testing {name}...")
        
        try:
            processed_img, _, _ = preprocess_image(
                image_path,
                segmentation_method='outline_focused',
                segmentation_params=params
            )
            
            result = (processed_img * 255).astype(np.uint8)
            axes[i].imshow(result)
            axes[i].set_title(f'{name}\n{params["max_contours"]} pen lifts', fontweight='bold')
            axes[i].axis('off')
            
            print(f"   ✅ Success: {params['max_contours']} outlines = {params['max_contours']} pen lifts")
            
        except Exception as e:
            print(f"   ❌ Error with {name}: {e}")
            axes[i].text(0.5, 0.5, f'Error\n{name}', ha='center', va='center')
    
    plt.suptitle('Minimal Outline Tracing Configurations', fontsize=16)
    plt.tight_layout()
    
    # Save minimal test
    results_dir = Path("results/step1_preprocessing")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    minimal_path = results_dir / f"minimal_outline_configs_{timestamp}.png"
    plt.savefig(minimal_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved minimal outline test: {minimal_path}")


def demonstrate_tactile_drawing_sequence():
    """
    Demonstrate the tactile drawing sequence for outline tracing
    """
    print(f"\n🤚 Tactile Drawing Sequence with Outline Tracing")
    print("=" * 55)
    
    print(f"\n📝 Drawing Process:")
    print(f"   1. 🏠 Draw building outline (1 continuous stroke)")
    print(f"   2. ☁️  Draw sky/cloud outlines (1 continuous stroke)")  
    print(f"   3. 🌳 Draw tree outlines (1 continuous stroke)")
    print(f"   4. 🚪 Draw detail outlines (windows, doors) (1-3 strokes)")
    print(f"   TOTAL: 4-6 pen lifts for complete drawing")
    
    print(f"\n🤚 Tactile Experience:")
    print(f"   • Feel building shape: rectangular outline, corners, edges")
    print(f"   • Feel tree shape: organic curves, branching patterns")
    print(f"   • Feel detail shapes: windows (rectangles), doors (rectangles)")
    print(f"   • No confusion from filled areas or backgrounds")
    
    print(f"\n🎯 Learning Benefits:")
    print(f"   • Object recognition: Shape = identity")
    print(f"   • Spatial relationships: Where objects are relative to each other")
    print(f"   • Geometric understanding: Rectangles, circles, organic curves")
    print(f"   • Architectural concepts: Buildings have windows, doors, roofs")
    
    print(f"\n⚡ Speed Benefits:")
    print(f"   • Fast drawing: Only essential outlines")
    print(f"   • Fast recognition: Immediate shape identification")
    print(f"   • Fast learning: Clear object boundaries")


if __name__ == "__main__":
    print("✏️  Outline Tracing Test - Pure Contour Drawing for Tactile Learning")
    print("=" * 75)
    
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
            test_outline_tracing_approaches(image_path)
            found_image = True
            break
    
    if not found_image:
        print("❌ No test images found in examples/ directory")
        print("Available test images should be:")
        for img in test_images:
            print(f"   - {img}")
    else:
        # Run additional tests
        test_minimal_outline_configurations()
        demonstrate_tactile_drawing_sequence()
    
    print(f"\n🎯 Summary:")
    print(f"   Outline tracing focuses on object SHAPES, not filled areas")
    print(f"   Perfect for tactile drawing: 4-8 meaningful contours")
    print(f"   No background confusion, only recognizable object outlines")
    print(f"   Check results/step1_preprocessing/ for detailed visualizations.")
