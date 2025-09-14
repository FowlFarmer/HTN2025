#!/usr/bin/env python3
"""
Step 6: Sampling & Scaling to Millimeters

Convert vectorized strokes to continuous sampled points with ~1mm spacing
in real-world coordinates for a 16x16cm canvas.

Features:
- Scale calculation from pixel coordinates to millimeters
- Arc-length resampling for uniform point spacing
- Edge case handling for short segments and closed loops
- Canvas-aware coordinate conversion
- Robot-ready millimeter output
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to path for constants import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import CANVAS_SIZE_MM, CANVAS_WIDTH_MM, CANVAS_HEIGHT_MM

def calculate_scale_factor(image_width_px, canvas_width_mm=CANVAS_SIZE_MM):
    """
    Calculate mm per pixel scaling factor

    Args:
        image_width_px: Width of processed image in pixels
        canvas_width_mm: Physical canvas width in millimeters (default: 160mm = 16cm)

    Returns:
        float: Scale factor in mm per pixel
    """
    return canvas_width_mm / image_width_px

def convert_to_mm(pts_px, scale_mm_per_px, canvas_height_mm=CANVAS_SIZE_MM, offset=(0, 0)):
    """
    Convert pixel coordinates to millimeters
    
    Converts from sknw coordinates (y,x format) to millimeter coordinates (x,y format)
    and flips Y-axis to match expected orientation, applying mask offset

    Args:
        pts_px: numpy.ndarray - Points in pixel coordinates (y,x format from sknw)
        scale_mm_per_px: float - Scale factor from calculate_scale_factor
        canvas_height_mm: float - Canvas height for coordinate system
        offset: tuple - (x_offset, y_offset) from mask cropping

    Returns:
        numpy.ndarray: Points in millimeter coordinates (x,y format)
    """
    if len(pts_px) == 0:
        return pts_px

    # sknw returns points in (y,x) format, but we need (x,y) for plotting
    # Swap coordinates: pts_px[:, 0] is y, pts_px[:, 1] is x
    if len(pts_px.shape) == 2 and pts_px.shape[1] >= 2:
        # Swap x and y coordinates and apply offset to place in original image coordinates
        # offset is (x_offset, y_offset) so we add offset[0] to x and offset[1] to y
        pts_mm = np.column_stack([pts_px[:, 1] + offset[0], pts_px[:, 0] + offset[1]]) * scale_mm_per_px
        # Flip Y-axis to correct orientation (sknw Y increases downward, we want upward)
        pts_mm[:, 1] = canvas_height_mm - pts_mm[:, 1]
    else:
        pts_mm = pts_px * scale_mm_per_px
    
    return pts_mm

def resample_polyline_mm(pts_px, scale_mm_per_px, spacing_mm=1.0, canvas_height_mm=CANVAS_SIZE_MM, offset=(0, 0)):
    """
    Resample polyline with uniform spacing in millimeters

    Args:
        pts_px: numpy.ndarray - Points in pixel coordinates
        scale_mm_per_px: float - Scale factor
        spacing_mm: float - Desired spacing between points in mm
        canvas_height_mm: float - Canvas height for coordinate conversion
        offset: tuple - (x_offset, y_offset) from mask cropping

    Returns:
        numpy.ndarray: Uniformly spaced points in mm coordinates
    """
    if len(pts_px) < 2:
        return convert_to_mm(pts_px, scale_mm_per_px, canvas_height_mm, offset)

    # Convert to mm with coordinate flip and offset
    pts_mm = convert_to_mm(pts_px, scale_mm_per_px, canvas_height_mm, offset)

    # Calculate cumulative distances
    deltas = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
    cum_distances = np.concatenate(([0], np.cumsum(deltas)))
    total_length = cum_distances[-1]

    # Handle very short segments
    if total_length < spacing_mm * 0.5:
        return np.array([pts_mm[0], pts_mm[-1]])

    # Generate sample points
    num_samples = max(2, int(total_length / spacing_mm) + 1)
    sample_distances = np.linspace(0, total_length, num_samples)

    # Interpolate coordinates
    x_interp = np.interp(sample_distances, cum_distances, pts_mm[:, 0])
    y_interp = np.interp(sample_distances, cum_distances, pts_mm[:, 1])

    return np.column_stack([x_interp, y_interp])

def handle_short_segments(pts_mm, min_length_mm=0.5):
    """
    Handle segments shorter than minimum length

    Args:
        pts_mm: numpy.ndarray - Points in mm coordinates
        min_length_mm: float - Minimum segment length to preserve

    Returns:
        numpy.ndarray: Processed points
    """
    if len(pts_mm) < 2:
        return pts_mm

    # Calculate total length
    deltas = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
    total_length = np.sum(deltas)

    if total_length < min_length_mm:
        # Keep only start and end points for very short segments
        return np.array([pts_mm[0], pts_mm[-1]])

    return pts_mm

def detect_closed_loop(pts_mm, closure_threshold_mm=2.0):
    """
    Detect if a polyline forms a closed loop

    Args:
        pts_mm: numpy.ndarray - Points in mm coordinates
        closure_threshold_mm: float - Max distance for loop closure

    Returns:
        bool: True if the polyline forms a closed loop
    """
    if len(pts_mm) < 3:
        return False

    start_end_distance = np.linalg.norm(pts_mm[0] - pts_mm[-1])
    return start_end_distance <= closure_threshold_mm

def handle_closed_loops(pts_mm, is_closed=None, closure_threshold_mm=2.0):
    """
    Handle closed loops with proper wrapping

    Args:
        pts_mm: numpy.ndarray - Points in mm coordinates
        is_closed: bool or None - Force closed/open, or None for auto-detection
        closure_threshold_mm: float - Max distance for loop closure

    Returns:
        tuple: (processed_points, is_closed_loop)
    """
    if len(pts_mm) < 3:
        return pts_mm, False

    # Auto-detect if not specified
    if is_closed is None:
        is_closed = detect_closed_loop(pts_mm, closure_threshold_mm)

    if is_closed:
        # Ensure proper loop closure
        if not np.allclose(pts_mm[0], pts_mm[-1], atol=0.1):
            # Close the loop by adding the start point at the end
            pts_mm = np.vstack([pts_mm, pts_mm[0:1]])

        return pts_mm, True

    return pts_mm, False

def calculate_sampling_statistics(original_pts, sampled_pts_mm, scale_mm_per_px):
    """
    Calculate statistics about the sampling process

    Args:
        original_pts: numpy.ndarray - Original pixel coordinates
        sampled_pts_mm: numpy.ndarray - Sampled mm coordinates
        scale_mm_per_px: float - Scale factor

    Returns:
        dict: Sampling statistics
    """
    if len(original_pts) == 0 or len(sampled_pts_mm) == 0:
        return {
            'original_points': 0,
            'sampled_points': 0,
            'original_length_mm': 0.0,
            'sampled_length_mm': 0.0,
            'average_spacing_mm': 0.0,
            'length_error_mm': 0.0
        }

    # Calculate original length in mm
    original_mm = original_pts * scale_mm_per_px
    original_deltas = np.linalg.norm(np.diff(original_mm, axis=0), axis=1)
    original_length_mm = np.sum(original_deltas)

    # Calculate sampled length
    sampled_deltas = np.linalg.norm(np.diff(sampled_pts_mm, axis=0), axis=1)
    sampled_length_mm = np.sum(sampled_deltas)

    # Average spacing
    average_spacing_mm = sampled_length_mm / max(1, len(sampled_pts_mm) - 1)

    # Length preservation error
    length_error_mm = abs(original_length_mm - sampled_length_mm)

    return {
        'original_points': len(original_pts),
        'sampled_points': len(sampled_pts_mm),
        'original_length_mm': original_length_mm,
        'sampled_length_mm': sampled_length_mm,
        'average_spacing_mm': average_spacing_mm,
        'length_error_mm': length_error_mm,
        'length_error_percent': (length_error_mm / max(0.1, original_length_mm)) * 100
    }

def sample_stroke_sequence(stroke_result, scale_mm_per_px, spacing_mm=1.0, min_length_mm=0.5, offset=(0, 0)):
    """
    Sample a single stroke sequence from Step 5 output

    Args:
        stroke_result: dict - Stroke result from vectorization
        scale_mm_per_px: float - Scale factor
        spacing_mm: float - Desired spacing in mm
        min_length_mm: float - Minimum segment length
        offset: tuple - (x_offset, y_offset) from mask cropping

    Returns:
        dict: Sampled stroke with metadata
    """
    vectorized_points = stroke_result['vectorized_points']

    if len(vectorized_points) == 0:
        return {
            'original_stroke': stroke_result,
            'sampled_points_mm': np.array([]),
            'is_closed_loop': False,
            'statistics': calculate_sampling_statistics(np.array([]), np.array([]), scale_mm_per_px),
            'canvas_bounds_mm': {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0},
            'success': False
        }

    try:
        # Resample with uniform spacing, applying offset
        sampled_pts_mm = resample_polyline_mm(vectorized_points, scale_mm_per_px, spacing_mm, CANVAS_SIZE_MM, offset)

        # Handle short segments
        sampled_pts_mm = handle_short_segments(sampled_pts_mm, min_length_mm)

        # Handle closed loops
        sampled_pts_mm, is_closed = handle_closed_loops(sampled_pts_mm)

        # Calculate statistics
        stats = calculate_sampling_statistics(vectorized_points, sampled_pts_mm, scale_mm_per_px)

        # Calculate canvas bounds
        if len(sampled_pts_mm) > 0:
            bounds = {
                'min_x': float(np.min(sampled_pts_mm[:, 0])),
                'max_x': float(np.max(sampled_pts_mm[:, 0])),
                'min_y': float(np.min(sampled_pts_mm[:, 1])),
                'max_y': float(np.max(sampled_pts_mm[:, 1]))
            }
        else:
            bounds = {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}

        return {
            'original_stroke': stroke_result,
            'sampled_points_mm': sampled_pts_mm,
            'is_closed_loop': is_closed,
            'statistics': stats,
            'canvas_bounds_mm': bounds,
            'success': len(sampled_pts_mm) >= 2
        }

    except Exception as e:
        print(f"Warning: Sampling failed for stroke: {e}")
        return {
            'original_stroke': stroke_result,
            'sampled_points_mm': np.array([]),
            'is_closed_loop': False,
            'statistics': calculate_sampling_statistics(vectorized_points, np.array([]), scale_mm_per_px),
            'canvas_bounds_mm': {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0},
            'success': False
        }

def sample_vectorized_results(vectorized_results, stroke_graphs, image_shape, spacing_mm=1.0,
                            canvas_width_mm=CANVAS_SIZE_MM, min_length_mm=0.5):
    """
    Sample all vectorized results from Step 5

    Args:
        vectorized_results: list - Results from Step 5 vectorization
        stroke_graphs: list - Stroke graph objects with offset information
        image_shape: tuple - (height, width) of processed image
        spacing_mm: float - Desired point spacing in mm
        canvas_width_mm: float - Physical canvas width in mm
        min_length_mm: float - Minimum segment length to preserve

    Returns:
        dict: Complete sampling results with metadata
    """
    # Calculate scale factor based on image width
    # Handle both 2D (height, width) and 3D (height, width, channels) shapes
    if len(image_shape) == 3:
        image_height, image_width, _ = image_shape
    else:
        image_height, image_width = image_shape
    scale_mm_per_px = calculate_scale_factor(image_width, canvas_width_mm)

    sampled_results = {
        'scale_info': {
            'canvas_size_mm': canvas_width_mm,
            'image_size_px': (image_width, image_height),
            'scale_mm_per_px': scale_mm_per_px,
            'spacing_mm': spacing_mm,
            'min_length_mm': min_length_mm
        },
        'sampled_graphs': [],
        'overall_statistics': {
            'total_strokes': 0,
            'successful_samplings': 0,
            'total_original_points': 0,
            'total_sampled_points': 0,
            'total_length_mm': 0.0,
            'closed_loops': 0,
            'canvas_utilization': 0.0
        }
    }

    total_stats = {
        'original_points': 0,
        'sampled_points': 0,
        'total_length_mm': 0.0,
        'closed_loops': 0,
        'successful_samplings': 0,
        'all_bounds': []
    }

    for i, vectorized_result in enumerate(vectorized_results):
        if vectorized_result is None:
            print(f"   ⚠️  Skipping empty vectorization result {i+1}")
            sampled_results['sampled_graphs'].append(None)
            continue

        print(f"   🎯 Sampling graph {i+1}...")

        # Get offset from corresponding stroke graph
        offset = (0, 0)  # Default offset
        if i < len(stroke_graphs) and stroke_graphs[i] is not None:
            if hasattr(stroke_graphs[i], 'offset'):
                offset = stroke_graphs[i].offset

        graph_result = {
            'boundary': [],
            'internal': [],
            'detail': [],
            'metadata': {
                'graph_id': i,
                'total_strokes': 0,
                'successful_samplings': 0,
                'total_length_mm': 0.0,
                'closed_loops': 0,
                'offset': offset
            }
        }

        # Process each semantic phase
        for phase_name in ['boundary', 'internal', 'detail']:
            phase_strokes = vectorized_result[phase_name]

            for stroke_result in phase_strokes:
                sampled_stroke = sample_stroke_sequence(stroke_result, scale_mm_per_px,
                                                      spacing_mm, min_length_mm, offset)

                graph_result[phase_name].append(sampled_stroke)

                # Update statistics
                if sampled_stroke['success']:
                    total_stats['successful_samplings'] += 1
                    total_stats['total_length_mm'] += sampled_stroke['statistics']['sampled_length_mm']

                    if sampled_stroke['is_closed_loop']:
                        total_stats['closed_loops'] += 1

                    # Collect bounds for canvas utilization
                    bounds = sampled_stroke['canvas_bounds_mm']
                    if bounds['max_x'] > bounds['min_x'] and bounds['max_y'] > bounds['min_y']:
                        total_stats['all_bounds'].append(bounds)

                total_stats['original_points'] += sampled_stroke['statistics']['original_points']
                total_stats['sampled_points'] += sampled_stroke['statistics']['sampled_points']

        # Calculate graph metadata
        all_graph_strokes = (graph_result['boundary'] +
                           graph_result['internal'] +
                           graph_result['detail'])

        graph_result['metadata'] = {
            'graph_id': i,
            'total_strokes': len(all_graph_strokes),
            'successful_samplings': sum(1 for s in all_graph_strokes if s['success']),
            'total_length_mm': sum(s['statistics']['sampled_length_mm']
                                 for s in all_graph_strokes if s['success']),
            'closed_loops': sum(1 for s in all_graph_strokes if s['is_closed_loop'])
        }

        # Print graph statistics
        meta = graph_result['metadata']
        print(f"      📊 Strokes: {meta['successful_samplings']}/{meta['total_strokes']}")
        print(f"      📏 Length: {meta['total_length_mm']:.1f}mm")
        print(f"      🔄 Closed loops: {meta['closed_loops']}")

        sampled_results['sampled_graphs'].append(graph_result)

    # Calculate canvas utilization
    canvas_utilization = 0.0
    if total_stats['all_bounds']:
        all_x = [b['min_x'] for b in total_stats['all_bounds']] + [b['max_x'] for b in total_stats['all_bounds']]
        all_y = [b['min_y'] for b in total_stats['all_bounds']] + [b['max_y'] for b in total_stats['all_bounds']]

        used_width = max(all_x) - min(all_x)
        used_height = max(all_y) - min(all_y)

        canvas_area = canvas_width_mm * canvas_width_mm  # Square canvas
        used_area = used_width * used_height
        canvas_utilization = min(1.0, used_area / canvas_area)

    # Update overall statistics
    sampled_results['overall_statistics'] = {
        'total_strokes': total_stats['original_points'],  # Using original points count as stroke count
        'successful_samplings': total_stats['successful_samplings'],
        'total_original_points': total_stats['original_points'],
        'total_sampled_points': total_stats['sampled_points'],
        'total_length_mm': total_stats['total_length_mm'],
        'closed_loops': total_stats['closed_loops'],
        'canvas_utilization': canvas_utilization,
        'average_spacing_mm': total_stats['total_length_mm'] / max(1, total_stats['sampled_points'] - total_stats['successful_samplings'])
    }

    return sampled_results

def process_all_sampling(vectorized_results, stroke_graphs, image_shape, spacing_mm=1.0,
                        canvas_width_mm=CANVAS_SIZE_MM, min_length_mm=0.5):
    """
    Process all vectorized results through sampling

    Args:
        vectorized_results: list - Results from Step 5
        stroke_graphs: list - Stroke graph objects with offset information
        image_shape: tuple - Image dimensions
        spacing_mm: float - Desired spacing in mm
        canvas_width_mm: float - Canvas size in mm
        min_length_mm: float - Minimum segment length

    Returns:
        dict: Complete sampling results
    """
    print(f"   📐 Canvas: {canvas_width_mm}mm x {canvas_width_mm}mm")
    print(f"   📏 Target spacing: {spacing_mm}mm")

    results = sample_vectorized_results(vectorized_results, stroke_graphs, image_shape,
                                      spacing_mm, canvas_width_mm, min_length_mm)

    # Print overall statistics
    stats = results['overall_statistics']
    scale_info = results['scale_info']

    print(f"   📊 Total: {stats['successful_samplings']} successful samplings")
    print(f"   📉 Compression: {stats['total_original_points']} → {stats['total_sampled_points']} points")
    print(f"   📏 Total length: {stats['total_length_mm']:.1f}mm")
    print(f"   🔄 Closed loops: {stats['closed_loops']}")
    print(f"   📐 Scale: {scale_info['scale_mm_per_px']:.3f} mm/px")
    print(f"   🎯 Canvas use: {stats['canvas_utilization']:.1%}")

    return results

def save_sampling_results(sampling_results, output_dir="results/step6_sampling"):
    """
    Save sampling visualization and results

    Args:
        sampling_results: dict - Results from sampling process
        output_dir: str - Directory to save results

    Returns:
        str: Path to saved visualization
    """
    import sys
    from pathlib import Path
    import matplotlib.pyplot as plt
    sys.path.append(str(Path(__file__).parent.parent))
    from visualization_utils import SEMANTIC_COLORS, SEMANTIC_ALPHAS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    valid_graphs = [g for g in sampling_results['sampled_graphs'] if g is not None]
    scale_info = sampling_results['scale_info']
    canvas_size = scale_info['canvas_size_mm']

    for i, graph_result in enumerate(valid_graphs[:6]):
        if i >= 6:
            break

        ax = axes[i]

        # Import consistent colors (moved outside loop for efficiency)
        pass

        # Plot canvas boundaries
        ax.plot([0, canvas_size, canvas_size, 0, 0],
               [0, 0, canvas_size, canvas_size, 0],
               'k--', alpha=0.3, linewidth=1, label='Canvas (16x16cm)')

        # Plot each semantic phase
        for phase_name in ['boundary', 'internal', 'detail']:
            phase_strokes = graph_result[phase_name]

            for stroke in phase_strokes:
                if stroke['success'] and len(stroke['sampled_points_mm']) > 1:
                    points_mm = stroke['sampled_points_mm']
                    color = SEMANTIC_COLORS[phase_name]
                    alpha = SEMANTIC_ALPHAS[phase_name]

                    # Plot stroke path
                    ax.plot(points_mm[:, 0], points_mm[:, 1],
                           color=color, linewidth=2, alpha=alpha)

                    # Mark sample points
                    ax.scatter(points_mm[:, 0], points_mm[:, 1],
                             c=color, s=4, alpha=alpha*0.7, zorder=5)

                    # Mark start/end for open strokes
                    if not stroke['is_closed_loop']:
                        ax.scatter(points_mm[0, 0], points_mm[0, 1],
                                 c='green', s=20, marker='o', zorder=6, edgecolor='white')
                        ax.scatter(points_mm[-1, 0], points_mm[-1, 1],
                                 c='red', s=20, marker='s', zorder=6, edgecolor='white')

        # Set title with statistics
        meta = graph_result['metadata']
        title = f'Graph {i+1}: Sampled (1mm spacing)\n'
        title += f'{meta["successful_samplings"]}/{meta["total_strokes"]} strokes, {meta["total_length_mm"]:.1f}mm\n'
        title += f'{meta["closed_loops"]} closed loops'

        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, canvas_size)
        ax.set_ylim(0, canvas_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.invert_yaxis()  # Match image coordinates

    # Hide unused subplots
    for i in range(len(valid_graphs), 6):
        axes[i].axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=2, label='Boundary (1mm spacing)'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Internal (1mm spacing)'),
        plt.Line2D([0], [0], color='green', lw=2, label='Detail (1mm spacing)'),
        plt.Line2D([0], [0], color='black', lw=1, linestyle='--', label='Canvas bounds'),
        plt.scatter([], [], c='green', s=20, marker='o', label='Start point'),
        plt.scatter([], [], c='red', s=20, marker='s', label='End point')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)

    plt.suptitle(f'Step 6: Millimeter Sampling Results', fontsize=16)
    plt.tight_layout()

    # Determine output path and create directory
    results_dir = Path(output_dir)
    if not results_dir.exists():
        results_dir = Path("results/step6_sampling")
    if not results_dir.exists():
        results_dir = Path("../results/step6_sampling")
    if not results_dir.exists():
        results_dir = Path(".")  # Fallback to current directory

    # Create directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"sampling_results_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   💾 Saved sampling visualization: {output_path}")

    return output_path