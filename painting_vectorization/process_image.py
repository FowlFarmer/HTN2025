#!/usr/bin/env python3
"""
Painting Vectorization - Image Processor

Process any image through the complete pipeline (Steps 1-3) and display results.

Usage:
    python process_image.py                    # Interactive mode - choose from examples
    python process_image.py image.jpg          # Process specific image
    python process_image.py examples/monet.jpeg # Process with path
    python process_image.py --list             # List available images
    python process_image.py --show-results     # View recent results
"""

import sys
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from step1_preprocessing.image_preprocessing import preprocess_image
from step2_segmentation.segmentation import segment_painting, save_segmentation_results
from step3_edge_extraction.edge_extraction import process_all_masks, save_edge_results

def find_images_in_examples():
    """Find all image files in examples directory"""
    examples_dir = Path("examples")
    if not examples_dir.exists():
        return []

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images = []

    for file_path in examples_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            images.append(file_path)

    return sorted(images)

def find_recent_results():
    """Find recent segmentation and edge extraction results"""
    examples_dir = Path("examples")
    if not examples_dir.exists():
        return [], []

    segmentation_results = list(examples_dir.glob("segmentation_results_*.png"))
    edge_results = list(examples_dir.glob("edge_extraction_results_*.png"))

    # Sort by timestamp in filename
    segmentation_results.sort(key=lambda x: x.name, reverse=True)
    edge_results.sort(key=lambda x: x.name, reverse=True)

    return segmentation_results[:5], edge_results[:5]  # Latest 5

def interactive_image_selection():
    """Interactive image selection from examples directory"""
    images = find_images_in_examples()

    if not images:
        print("❌ No images found in examples/ directory")
        print("💡 Add some image files to examples/ and try again")
        return None

    print("📂 Available images in examples/ directory:")
    print("=" * 50)

    for i, image_path in enumerate(images, 1):
        file_size = image_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"   {i}. {image_path.name} ({size_mb:.1f} MB)")

    print()
    try:
        choice = input(f"Select image (1-{len(images)}) or 'q' to quit: ").strip()

        if choice.lower() == 'q':
            return None

        choice_num = int(choice)
        if 1 <= choice_num <= len(images):
            return images[choice_num - 1]
        else:
            print(f"❌ Invalid choice. Please select 1-{len(images)}")
            return None

    except (ValueError, KeyboardInterrupt):
        print("\n👋 Goodbye!")
        return None

def process_painting_pipeline(image_path):
    """
    Run the complete painting processing pipeline

    Args:
        image_path: Path to input image

    Returns:
        dict: Processing results and metadata
    """
    print(f"🎨 Processing: {Path(image_path).name}")
    print("=" * 60)

    start_time = datetime.now()

    try:
        # Validate input
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Step 1: Preprocessing
        print("📐 Step 1: Image Preprocessing")
        print("-" * 30)
        step1_start = datetime.now()

        processed_img, scale_factor, original_dims = preprocess_image(str(image_path))

        step1_time = (datetime.now() - step1_start).total_seconds()
        print(f"   📏 Original: {original_dims}")
        print(f"   📐 Scale: {scale_factor:.3f}")
        print(f"   🖼️  Shape: {processed_img.shape}")
        print(f"   ⏱️  Time: {step1_time:.3f}s")
        print("   ✅ Complete!")
        print()

        # Step 2: Segmentation
        print("🎯 Step 2: Segmentation")
        print("-" * 30)
        step2_start = datetime.now()

        masks, scores, _ = segment_painting(str(image_path))

        step2_time = (datetime.now() - step2_start).total_seconds()
        print(f"   🎭 Masks: {len(masks)}")

        total_coverage = 0
        for i, (mask, score) in enumerate(zip(masks, scores)):
            coverage = np.sum(mask) / (mask.shape[0] * mask.shape[1])
            total_coverage += coverage
            print(f"   • Mask {i+1}: {coverage:.1%} area, salience={score['salience']:.3f}")

        print(f"   📊 Coverage: {total_coverage:.1%}")
        print(f"   ⏱️  Time: {step2_time:.1f}s")

        # Save segmentation visualization
        segmentation_output_path = save_segmentation_results(masks, scores, processed_img)
        print("   ✅ Complete!")
        print()

        # Step 3: Edge Extraction
        print("🖋️  Step 3: Edge Extraction")
        print("-" * 30)
        step3_start = datetime.now()

        edge_results = process_all_masks(processed_img, masks, method="multi_scale")

        step3_time = (datetime.now() - step3_start).total_seconds()
        successful_extractions = len([r for r in edge_results if r is not None])

        if successful_extractions > 0:
            valid_results = [r for r in edge_results if r is not None]
            avg_edge_density = np.mean([r['stats']['edge_density'] for r in valid_results])
            avg_skeleton_density = np.mean([r['stats']['skeleton_density'] for r in valid_results])
            total_skeleton_pixels = sum([r['stats']['skeleton_pixels'] for r in valid_results])

            print(f"   ✅ Success: {successful_extractions}/{len(masks)} masks")
            print(f"   📏 Avg edge density: {avg_edge_density:.3f}")
            print(f"   🖋️  Avg skeleton density: {avg_skeleton_density:.3f}")
            print(f"   📊 Skeleton pixels: {total_skeleton_pixels:,}")
        else:
            print("   ⚠️  No successful extractions")

        # Save results
        edge_output_path = save_edge_results(edge_results, masks, processed_img.shape)

        print(f"   ⏱️  Time: {step3_time:.3f}s")
        print("   ✅ Complete!")
        print()

        # Summary
        total_time = (datetime.now() - start_time).total_seconds()
        print("📋 Processing Summary")
        print("-" * 30)
        print(f"   🖼️  Input: {image_path.name}")
        print(f"   📐 Resolution: {original_dims} -> {processed_img.shape[:2]}")
        print(f"   🎭 Segments: {len(masks)}")
        print(f"   🖋️  Extractions: {successful_extractions}")
        print(f"   ⏱️  Total time: {total_time:.1f}s")
        print()
        print("📁 Output Files:")
        print(f"   🎯 Segmentation: {segmentation_output_path}")
        print(f"   🖋️  Edge extraction: {edge_output_path}")
        print()

        return {
            'success': True,
            'image_path': image_path,
            'processing_time': total_time,
            'masks': len(masks),
            'extractions': successful_extractions,
            'coverage': total_coverage,
            'edge_output_path': edge_output_path,
            'segmentation_output_path': segmentation_output_path
        }

    except Exception as e:
        print(f"❌ Error processing {image_path.name}: {str(e)}")
        print("💡 Troubleshooting:")
        print("   • Check image file is readable")
        print("   • Ensure all dependencies are installed")
        print("   • Verify SAM model is available")
        return {'success': False, 'error': str(e)}

def list_images():
    """List all available images"""
    images = find_images_in_examples()

    print("📂 Available Images")
    print("=" * 40)

    if not images:
        print("❌ No images found in examples/ directory")
        print("💡 Add image files (.jpg, .png, etc.) to examples/")
        return

    for image_path in images:
        file_size = image_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        modified = datetime.fromtimestamp(image_path.stat().st_mtime)
        print(f"   📷 {image_path.name}")
        print(f"      Size: {size_mb:.1f} MB")
        print(f"      Modified: {modified.strftime('%Y-%m-%d %H:%M')}")
        print()

def show_recent_results():
    """Show recent processing results"""
    seg_results, edge_results = find_recent_results()

    print("📊 Recent Results")
    print("=" * 40)

    if seg_results:
        print("🎯 Segmentation Results:")
        for result in seg_results:
            timestamp = result.name.split('_')[-1].replace('.png', '')
            if len(timestamp) == 6:  # HHMMSS format
                time_str = f"{timestamp[:2]}:{timestamp[2:4]}:{timestamp[4:6]}"
            else:
                time_str = timestamp
            print(f"   • {result.name} ({time_str})")
        print()

    if edge_results:
        print("🖋️  Edge Extraction Results:")
        for result in edge_results:
            timestamp = result.name.split('_')[-1].replace('.png', '')
            if len(timestamp) == 6:  # HHMMSS format
                time_str = f"{timestamp[:2]}:{timestamp[2:4]}:{timestamp[4:6]}"
            else:
                time_str = timestamp
            print(f"   • {result.name} ({time_str})")
        print()

    if not seg_results and not edge_results:
        print("❌ No recent results found")
        print("💡 Run some processing first to generate results")

def main():
    parser = argparse.ArgumentParser(
        description="Process paintings through vectorization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_image.py                    # Interactive mode
  python process_image.py monet.jpeg         # Process specific image
  python process_image.py examples/art.png   # Process with path
  python process_image.py --list             # List available images
  python process_image.py --show-results     # View recent results
        """
    )

    parser.add_argument('image', nargs='?', help='Image file to process')
    parser.add_argument('--list', action='store_true', help='List available images')
    parser.add_argument('--show-results', action='store_true', help='Show recent results')

    args = parser.parse_args()

    print("🎨 Painting Vectorization Pipeline")
    print("=" * 50)

    if args.list:
        list_images()
        return

    if args.show_results:
        show_recent_results()
        return

    # Determine image to process
    if args.image:
        # Handle different path formats
        image_path = Path(args.image)

        # If not found, try in examples directory
        if not image_path.exists():
            examples_path = Path("examples") / args.image
            if examples_path.exists():
                image_path = examples_path
            else:
                print(f"❌ Image not found: {args.image}")
                print("💡 Try:")
                print(f"   python process_image.py --list")
                return
    else:
        # Interactive mode
        image_path = interactive_image_selection()
        if image_path is None:
            return

    # Process the image
    result = process_painting_pipeline(image_path)

    if result['success']:
        print("🎉 Processing complete! ✨")
        print()
        print("🎯 Next steps:")
        print(f"   • Open {Path(result['edge_output_path']).name} to view edge extraction")
        print("   • Use --show-results to see all recent outputs")
        print("   • Process more images or implement vector reconstruction")
    else:
        print("😞 Processing failed")
        sys.exit(1)

if __name__ == "__main__":
    main()