#!/usr/bin/env python3
"""
Painting Vectorization Pipeline Demo

This script demonstrates the complete pipeline for processing paintings:
1. Image preprocessing
2. Segmentation with SAM
3. Results visualization and analysis

Usage:
    python demo.py [image_path]

If no image_path is provided, uses examples/monet.jpeg
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from step1_preprocessing.image_preprocessing import preprocess_image
from step2_segmentation.segmentation import segment_painting
from step3_edge_extraction.edge_extraction import process_all_masks, save_edge_results

def run_demo(image_path="examples/monet.jpeg"):
    """
    Run the complete painting vectorization demo

    Args:
        image_path (str): Path to input painting image
    """

    print("🎨 Painting Vectorization Pipeline Demo")
    print("=" * 50)

    # Validate input
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found at {image_path}")
        print("💡 Try: python demo.py examples/monet.jpeg")
        return False

    try:
        # Step 1: Preprocessing
        print("📐 Step 1: Image Preprocessing")
        print("-" * 30)

        processed_img, scale_factor, original_dims = preprocess_image(image_path)

        print(f"   📏 Original dimensions: {original_dims}")
        print(f"   📐 Scale factor: {scale_factor:.3f}")
        print(f"   🖼️  Processed shape: {processed_img.shape}")
        print(f"   📊 Data type: {processed_img.dtype}")
        print(f"   🎯 Value range: [{processed_img.min():.3f}, {processed_img.max():.3f}]")
        print("   ✅ Preprocessing complete!")
        print()

        # Step 2: Segmentation
        print("🎯 Step 2: Segmentation with SAM")
        print("-" * 30)
        print("   🤖 Loading SAM model...")

        masks, scores, _ = segment_painting(image_path)

        print(f"   🎭 Generated {len(masks)} segmentation masks")
        print("   📊 Mask analysis:")

        total_pixels = masks[0].shape[0] * masks[0].shape[1]
        total_coverage = 0

        for i, (mask, score) in enumerate(zip(masks, scores)):
            coverage = np.sum(mask) / total_pixels
            total_coverage += coverage
            print(f"      Mask {i+1}: {coverage:.1%} area, "
                  f"salience={score['salience']:.3f}, "
                  f"edge={score['edge_strength']:.3f}")

        print(f"   🎯 Total coverage: {total_coverage:.1%}")
        print("   ✅ Segmentation complete!")
        print()

        # Step 3: Edge Extraction
        print("🖋️  Step 3: Edge/Line Extraction")
        print("-" * 30)
        print("   🎨 Extracting edges from each mask...")

        edge_results = process_all_masks(processed_img, masks, method="multi_scale")

        # Count successful extractions
        successful_extractions = len([r for r in edge_results if r is not None])
        print(f"   ✅ Edge extraction complete!")
        print(f"   📊 Successful extractions: {successful_extractions}/{len(masks)}")

        # Analyze edge statistics
        if successful_extractions > 0:
            valid_results = [r for r in edge_results if r is not None]
            avg_edge_density = np.mean([r['stats']['edge_density'] for r in valid_results])
            avg_skeleton_density = np.mean([r['stats']['skeleton_density'] for r in valid_results])
            total_skeleton_pixels = sum([r['stats']['skeleton_pixels'] for r in valid_results])

            print(f"   📏 Average edge density: {avg_edge_density:.3f}")
            print(f"   🖋️  Average skeleton density: {avg_skeleton_density:.3f}")
            print(f"   📊 Total skeleton pixels: {total_skeleton_pixels:,}")

        # Save edge extraction results
        edge_output_path = save_edge_results(edge_results, masks, processed_img.shape)
        print()

        # Results Summary
        print("📋 Pipeline Summary")
        print("-" * 30)
        print(f"   🖼️  Input: {Path(image_path).name}")
        print(f"   📐 Resolution: {original_dims} -> {processed_img.shape[:2]}")
        print(f"   🎭 Segments: {len(masks)} meaningful regions")
        print(f"   🖋️  Edge extractions: {successful_extractions} successful")
        print(f"   🏆 Best segment: {max(scores, key=lambda x: x['salience'])['salience']:.3f} salience")
        print(f"   📊 Coverage: {total_coverage:.1%} of image")

        # Quality assessment
        if len(masks) >= 3 and total_coverage >= 0.8:
            print("   🎉 Quality: Excellent segmentation!")
        elif len(masks) >= 2 and total_coverage >= 0.6:
            print("   👍 Quality: Good segmentation!")
        else:
            print("   ⚠️  Quality: Consider tuning parameters")

        print()
        print("🎯 Next Steps:")
        print(f"   • Check {Path(edge_output_path).name} for edge visualization")
        print("   • Use extracted skeletons for stroke parameterization")
        print("   • Implement vector reconstruction pipeline")
        print()
        print("🎨 Demo complete! ✨")

        return True

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        print("💡 Troubleshooting:")
        print("   • Check that all dependencies are installed")
        print("   • Ensure SAM model is downloaded")
        print("   • Verify image file is readable")
        return False

def analyze_masks(masks, scores):
    """
    Provide detailed analysis of segmentation results

    Args:
        masks: List of segmentation masks
        scores: List of mask quality scores
    """

    print("🔍 Detailed Mask Analysis")
    print("=" * 40)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        print(f"\n🎭 Mask {i+1}:")
        print(f"   📏 Dimensions: {mask.shape}")
        print(f"   📊 Area ratio: {score['area_score']:.3f}")
        print(f"   ⚡ Edge strength: {score['edge_strength']:.3f}")
        print(f"   🎨 Color distinctiveness: {score['color_distinctiveness']:.3f}")
        print(f"   ⭐ Salience score: {score['salience']:.3f}")

        # Mask statistics
        mask_area = np.sum(mask)
        total_area = mask.shape[0] * mask.shape[1]
        print(f"   📐 Pixels: {mask_area:,} / {total_area:,} ({mask_area/total_area:.1%})")

        # Quality assessment
        if score['salience'] > 0.1:
            quality = "🌟 Excellent"
        elif score['salience'] > 0.05:
            quality = "👍 Good"
        elif score['salience'] > 0.01:
            quality = "⚠️  Fair"
        else:
            quality = "❌ Poor"

        print(f"   🏆 Quality: {quality}")

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "examples/monet.jpeg"

    # Run the demo
    success = run_demo(image_path)

    # Exit with appropriate code
    sys.exit(0 if success else 1)