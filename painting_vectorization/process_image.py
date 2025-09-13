#!/usr/bin/env python3
"""
Painting Vectorization - Image Processor

Process any image through the complete pipeline (Steps 1-8) and display results.

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
from step4_stroke_graph.stroke_graph import process_all_stroke_graphs, save_graph_visualization
from step5_vectorization.vectorization import process_all_vectorizations, save_vectorization_results
from step6_sampling.sampling import process_all_sampling, save_sampling_results
from step7_color_detection.color_detection import process_all_color_detection, save_color_detection_results
from step8_stroke_ordering.stroke_ordering import process_all_stroke_ordering, save_stroke_ordering_results

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
    """Find recent segmentation, edge extraction, stroke graph, vectorization, sampling, and color detection results"""
    results_base = Path("results")

    # Define directories for each step
    directories = {
        'segmentation': results_base / "step2_segmentation",
        'edge': results_base / "step3_edge_extraction",
        'stroke_graph': results_base / "step4_stroke_graphs",
        'vectorization': results_base / "step5_vectorization",
        'sampling': results_base / "step6_sampling",
        'color_detection': results_base / "step7_color_detection",
        'stroke_ordering': results_base / "step8_stroke_ordering"
    }

    # Find results in each directory
    segmentation_results = []
    edge_results = []
    stroke_graph_results = []
    vectorization_results = []
    sampling_results = []
    color_detection_results = []
    stroke_ordering_results = []

    if directories['segmentation'].exists():
        segmentation_results = list(directories['segmentation'].glob("segmentation_results_*.png"))
    if directories['edge'].exists():
        edge_results = list(directories['edge'].glob("edge_extraction_results_*.png"))
    if directories['stroke_graph'].exists():
        stroke_graph_results = list(directories['stroke_graph'].glob("stroke_graphs_*.png"))
    if directories['vectorization'].exists():
        vectorization_results = list(directories['vectorization'].glob("vectorization_results_*.png"))
    if directories['sampling'].exists():
        sampling_results = list(directories['sampling'].glob("sampling_results_*.png"))
    if directories['color_detection'].exists():
        color_detection_results = list(directories['color_detection'].glob("color_detection_results_*.png"))
    if directories['stroke_ordering'].exists():
        stroke_ordering_results = list(directories['stroke_ordering'].glob("stroke_ordering_*.*"))

    # Sort by timestamp in filename
    segmentation_results.sort(key=lambda x: x.name, reverse=True)
    edge_results.sort(key=lambda x: x.name, reverse=True)
    stroke_graph_results.sort(key=lambda x: x.name, reverse=True)
    vectorization_results.sort(key=lambda x: x.name, reverse=True)
    sampling_results.sort(key=lambda x: x.name, reverse=True)
    color_detection_results.sort(key=lambda x: x.name, reverse=True)
    stroke_ordering_results.sort(key=lambda x: x.name, reverse=True)

    return segmentation_results[:5], edge_results[:5], stroke_graph_results[:5], vectorization_results[:5], sampling_results[:5], color_detection_results[:5], stroke_ordering_results[:5]  # Latest 5

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

        edge_results = process_all_masks(processed_img, masks, method="adaptive")

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

        # Step 4: Stroke Graph Construction
        print("🕸️  Step 4: Stroke Graph Construction")
        print("-" * 30)
        step4_start = datetime.now()

        stroke_graph_results = process_all_stroke_graphs(edge_results)

        step4_time = (datetime.now() - step4_start).total_seconds()
        successful_graphs = len([r for r in stroke_graph_results if r is not None])

        if successful_graphs > 0:
            valid_graph_results = [r for r in stroke_graph_results if r is not None]
            total_nodes = sum([r.stats['num_nodes'] for r in valid_graph_results])
            total_edges = sum([r.stats['num_edges'] for r in valid_graph_results])
            eulerian_paths = sum([1 for r in valid_graph_results if r.stats['is_eulerian']])

            print(f"   ✅ Success: {successful_graphs}/{len(edge_results)} graphs")
            print(f"   🔗 Total nodes: {total_nodes:,}")
            print(f"   📊 Total edges: {total_edges:,}")
            print(f"   🛤️  Eulerian paths: {eulerian_paths}/{successful_graphs}")
        else:
            print("   ⚠️  No successful graph constructions")

        # Save stroke graph results
        stroke_graph_output_path = save_graph_visualization(stroke_graph_results)

        print(f"   ⏱️  Time: {step4_time:.3f}s")
        print("   ✅ Complete!")
        print()

        # Step 5: Vectorization & Simplification
        print("🔧 Step 5: Vectorization & Simplification")
        print("-" * 30)
        step5_start = datetime.now()

        vectorized_results = process_all_vectorizations(stroke_graph_results, method='adaptive', max_error=1.0)

        step5_time = (datetime.now() - step5_start).total_seconds()
        successful_vectorizations = len([r for r in vectorized_results if r is not None])

        if successful_vectorizations > 0:
            valid_vectorized_results = [r for r in vectorized_results if r is not None]
            total_original_points = sum([r['metadata']['total_points_original'] for r in valid_vectorized_results])
            total_simplified_points = sum([r['metadata']['total_points_simplified'] for r in valid_vectorized_results])
            avg_compression = np.mean([r['metadata']['average_compression'] for r in valid_vectorized_results])
            avg_error = np.mean([r['metadata']['average_error'] for r in valid_vectorized_results])
            total_vectorized_strokes = sum([r['metadata']['total_strokes'] for r in valid_vectorized_results])

            print(f"   ✅ Success: {successful_vectorizations}/{len(stroke_graph_results)} vectorizations")
            print(f"   🔧 Total strokes: {total_vectorized_strokes}")
            print(f"   📉 Compression: {avg_compression:.2f}x ({total_original_points:,} → {total_simplified_points:,} points)")
            print(f"   📏 Avg error: {avg_error:.2f} pixels")
        else:
            print("   ⚠️  No successful vectorizations")

        # Save vectorization results
        vectorization_output_path = save_vectorization_results(vectorized_results)

        print(f"   ⏱️  Time: {step5_time:.3f}s")
        print("   ✅ Complete!")
        print()

        # Step 6: Sampling & Scaling to Millimeters
        print("📐 Step 6: Sampling & Scaling to Millimeters")
        print("-" * 30)
        step6_start = datetime.now()

        sampling_results = process_all_sampling(vectorized_results, processed_img.shape,
                                              spacing_mm=1.0, canvas_width_mm=160)

        step6_time = (datetime.now() - step6_start).total_seconds()

        # Extract statistics
        stats = sampling_results['overall_statistics']
        scale_info = sampling_results['scale_info']

        print(f"   ✅ Success: {stats['successful_samplings']} samplings")
        print(f"   📐 Scale: {scale_info['scale_mm_per_px']:.3f} mm/pixel")
        print(f"   📏 Total length: {stats['total_length_mm']:.1f}mm")
        print(f"   🎯 Canvas usage: {stats['canvas_utilization']:.1%}")
        print(f"   🔄 Closed loops: {stats['closed_loops']}")

        # Save sampling results
        sampling_output_path = save_sampling_results(sampling_results)

        print(f"   ⏱️  Time: {step6_time:.3f}s")
        print("   ✅ Complete!")
        print()

        # Step 7: Color Detection - Warm vs Cool Classification
        print("🎨 Step 7: Color Detection - Warm vs Cool Classification")
        print("-" * 30)
        step7_start = datetime.now()

        color_detection_results = process_all_color_detection(processed_img, masks, sampling_results)

        step7_time = (datetime.now() - step7_start).total_seconds()

        # Extract color statistics
        color_stats = color_detection_results['overall_statistics']

        print(f"   ✅ Success: {color_stats['successful_classifications']} classifications")
        print(f"   🔥 Warm strokes: {color_stats['warm_strokes']} ({color_stats['warm_percentage']:.1f}%)")
        print(f"   ❄️  Cool strokes: {color_stats['cool_strokes']} ({color_stats['cool_percentage']:.1f}%)")
        print(f"   🎨 Avg confidence: {color_stats['average_confidence']:.3f}")
        print(f"   ⚙️  Method: {color_stats['method_used']}")

        # Save color detection results
        color_detection_output_path = save_color_detection_results(color_detection_results, processed_img, masks)

        print(f"   ⏱️  Time: {step7_time:.3f}s")
        print("   ✅ Complete!")
        print()

        # Step 8: Flattening & Stroke Ordering (Travel Optimization)
        print("🚀 Step 8: Flattening & Stroke Ordering (Travel Optimization)")
        print("-" * 30)
        step8_start = datetime.now()

        ordering_results = process_all_stroke_ordering(color_detection_results, strategy='hybrid')

        step8_time = (datetime.now() - step8_start).total_seconds()

        # Extract ordering statistics
        ordering_stats = ordering_results['statistics']

        print(f"   ✅ Success: {ordering_stats['total_strokes']} strokes flattened and optimized")
        print(f"   🎯 Strategy: {ordering_stats['optimization_method']}")
        print(f"   📏 Drawing length: {ordering_stats['total_length_mm']:.1f}mm")
        print(f"   ✈️  Travel distance: {ordering_stats['estimated_travel_distance_mm']:.1f}mm")
        print(f"   🔄 Pen lifts: {ordering_stats['total_pen_lifts']}")
        print(f"   ⏱️  Estimated execution: {ordering_stats['estimated_duration_s']:.1f}s ({ordering_stats['estimated_duration_s']/60:.1f} min)")

        # Save stroke ordering results
        stroke_ordering_output_path = save_stroke_ordering_results(ordering_results)

        print(f"   ⏱️  Time: {step8_time:.3f}s")
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
        print(f"   🕸️  Graphs: {successful_graphs if 'successful_graphs' in locals() else 0}")
        print(f"   🔧 Vectorizations: {successful_vectorizations if 'successful_vectorizations' in locals() else 0}")
        print(f"   📐 Samplings: {stats['successful_samplings'] if 'stats' in locals() else 0} ({stats['total_length_mm']:.1f}mm)" if 'stats' in locals() else "   📐 Samplings: 0")
        print(f"   🎨 Color classifications: {color_stats['successful_classifications'] if 'color_stats' in locals() else 0} ({color_stats['warm_percentage'] if 'color_stats' in locals() else 0:.1f}% warm)" if 'color_stats' in locals() else "   🎨 Color classifications: 0")
        print(f"   🚀 Stroke ordering: {ordering_stats['total_strokes'] if 'ordering_stats' in locals() else 0} strokes, {ordering_stats['total_pen_lifts'] if 'ordering_stats' in locals() else 0} pen lifts" if 'ordering_stats' in locals() else "   🚀 Stroke ordering: 0")
        print(f"   ⏱️  Total time: {total_time:.1f}s")
        print()
        print("📁 Output Files:")
        print(f"   🎯 Segmentation: {segmentation_output_path}")
        print(f"   🖋️  Edge extraction: {edge_output_path}")
        if 'stroke_graph_output_path' in locals():
            print(f"   🕸️  Stroke graphs: {stroke_graph_output_path}")
        if 'vectorization_output_path' in locals():
            print(f"   🔧 Vectorization: {vectorization_output_path}")
        if 'sampling_output_path' in locals():
            print(f"   📐 Sampling: {sampling_output_path}")
        if 'color_detection_output_path' in locals():
            print(f"   🎨 Color detection: {color_detection_output_path}")
        if 'stroke_ordering_output_path' in locals():
            print(f"   🚀 Stroke ordering: {stroke_ordering_output_path}")
        print()

        return {
            'success': True,
            'image_path': image_path,
            'processing_time': total_time,
            'masks': len(masks),
            'extractions': successful_extractions,
            'graphs': successful_graphs if 'successful_graphs' in locals() else 0,
            'vectorizations': successful_vectorizations if 'successful_vectorizations' in locals() else 0,
            'samplings': stats['successful_samplings'] if 'stats' in locals() else 0,
            'total_length_mm': stats['total_length_mm'] if 'stats' in locals() else 0.0,
            'color_classifications': color_stats['successful_classifications'] if 'color_stats' in locals() else 0,
            'warm_percentage': color_stats['warm_percentage'] if 'color_stats' in locals() else 0.0,
            'stroke_ordering': ordering_stats['total_strokes'] if 'ordering_stats' in locals() else 0,
            'pen_lifts': ordering_stats['total_pen_lifts'] if 'ordering_stats' in locals() else 0,
            'coverage': total_coverage,
            'edge_output_path': edge_output_path,
            'segmentation_output_path': segmentation_output_path,
            'stroke_graph_output_path': stroke_graph_output_path if 'stroke_graph_output_path' in locals() else None,
            'vectorization_output_path': vectorization_output_path if 'vectorization_output_path' in locals() else None,
            'sampling_output_path': sampling_output_path if 'sampling_output_path' in locals() else None,
            'color_detection_output_path': color_detection_output_path if 'color_detection_output_path' in locals() else None,
            'stroke_ordering_output_path': stroke_ordering_output_path if 'stroke_ordering_output_path' in locals() else None
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
    seg_results, edge_results, stroke_graph_results, vectorization_results, sampling_results, color_detection_results, stroke_ordering_results = find_recent_results()

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

    if stroke_graph_results:
        print("🕸️  Stroke Graph Results:")
        for result in stroke_graph_results:
            timestamp = result.name.split('_')[-1].replace('.png', '')
            if len(timestamp) == 6:  # HHMMSS format
                time_str = f"{timestamp[:2]}:{timestamp[2:4]}:{timestamp[4:6]}"
            else:
                time_str = timestamp
            print(f"   • {result.name} ({time_str})")
        print()

    if vectorization_results:
        print("🔧 Vectorization Results:")
        for result in vectorization_results:
            timestamp = result.name.split('_')[-1].replace('.png', '')
            if len(timestamp) == 6:  # HHMMSS format
                time_str = f"{timestamp[:2]}:{timestamp[2:4]}:{timestamp[4:6]}"
            else:
                time_str = timestamp
            print(f"   • {result.name} ({time_str})")
        print()

    if sampling_results:
        print("📐 Sampling Results:")
        for result in sampling_results:
            timestamp = result.name.split('_')[-1].replace('.png', '')
            if len(timestamp) == 6:  # HHMMSS format
                time_str = f"{timestamp[:2]}:{timestamp[2:4]}:{timestamp[4:6]}"
            else:
                time_str = timestamp
            print(f"   • {result.name} ({time_str})")
        print()

    if color_detection_results:
        print("🎨 Color Detection Results:")
        for result in color_detection_results:
            timestamp = result.name.split('_')[-1].replace('.png', '')
            if len(timestamp) == 6:  # HHMMSS format
                time_str = f"{timestamp[:2]}:{timestamp[2:4]}:{timestamp[4:6]}"
            else:
                time_str = timestamp
            print(f"   • {result.name} ({time_str})")
        print()

    if stroke_ordering_results:
        print("🚀 Stroke Ordering Results:")
        for result in stroke_ordering_results:
            timestamp = result.name.split('_')[-1].replace('.json', '').replace('.png', '').replace('.gcode', '').replace('.csv', '')
            if len(timestamp) == 6:  # HHMMSS format
                time_str = f"{timestamp[:2]}:{timestamp[2:4]}:{timestamp[4:6]}"
            else:
                time_str = timestamp
            print(f"   • {result.name} ({time_str})")
        print()

    if not seg_results and not edge_results and not stroke_graph_results and not vectorization_results and not sampling_results and not color_detection_results and not stroke_ordering_results:
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
        if result.get('stroke_graph_output_path'):
            print(f"   • Open {Path(result['stroke_graph_output_path']).name} to view stroke graphs")
        if result.get('vectorization_output_path'):
            print(f"   • Open {Path(result['vectorization_output_path']).name} to view vectorized strokes")
        if result.get('sampling_output_path'):
            print(f"   • Open {Path(result['sampling_output_path']).name} to view 1mm sampled points")
        if result.get('color_detection_output_path'):
            print(f"   • Open {Path(result['color_detection_output_path']).name} to view warm/cool color classification")
        if result.get('stroke_ordering_output_path'):
            print(f"   • Open {Path(result['stroke_ordering_output_path']).name} to view optimized robot instructions")
        print("   • Use --show-results to see all recent outputs")
        print("   • Ready for hardware execution with JSON/G-code robot instructions!")
    else:
        print("😞 Processing failed")
        sys.exit(1)

if __name__ == "__main__":
    main()