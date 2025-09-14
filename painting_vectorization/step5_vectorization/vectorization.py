#!/usr/bin/env python3
"""
Step 5: Vectorization & Simplification

Convert pixel sequences from stroke graphs into compact polylines or smooth curves
suitable for robot motion and vector graphics.

Features:
- RDP (Ramer-Douglas-Peucker) simplification
- Spline fitting for smooth curves
- Adaptive parameter tuning
- Quality metrics and error analysis
- Semantic-aware vectorization
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    from rdp import rdp
    RDP_AVAILABLE = True
except ImportError:
    RDP_AVAILABLE = False
    print("Warning: rdp not available. Install with: pip install rdp")

try:
    from scipy.interpolate import splprep, splev
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")

def vectorize_outline_stroke(stroke, max_error=1.0):
    """
    Vectorize a single outline stroke using Douglas-Peucker simplification
    
    Args:
        stroke: Stroke dictionary with 'path' key
        max_error: Maximum simplification error in pixels
    
    Returns:
        dict: Vectorized stroke with simplified path
    """
    path = stroke['path']
    
    if len(path) < 3:
        # Too few points to simplify
        return {
            **stroke,
            'original_path': path,
            'simplified_path': path,
            'compression_ratio': 1.0,
            'simplification_error': 0.0,
            'num_points_original': len(path),
            'num_points_simplified': len(path)
        }
    
    # Apply Douglas-Peucker simplification
    if RDP_AVAILABLE:
        try:
            simplified_path = rdp(path, epsilon=max_error)
        except:
            # Fallback to OpenCV if rdp fails
            simplified_path = cv2.approxPolyDP(
                path.astype(np.int32), max_error, closed=stroke.get('is_closed', True)
            ).reshape(-1, 2)
    else:
        # Use OpenCV Douglas-Peucker
        simplified_path = cv2.approxPolyDP(
            path.astype(np.int32), max_error, closed=stroke.get('is_closed', True)
        ).reshape(-1, 2)
    
    # Calculate compression metrics
    original_points = len(path)
    simplified_points = len(simplified_path)
    compression_ratio = original_points / simplified_points if simplified_points > 0 else 1.0
    
    # Calculate average error (simplified)
    avg_error = max_error  # Approximation since exact calculation is expensive
    
    return {
        **stroke,
        'original_path': path,
        'simplified_path': simplified_path.astype(np.float32),
        'compression_ratio': compression_ratio,
        'simplification_error': avg_error,
        'num_points_original': original_points,
        'num_points_simplified': simplified_points,
        'method': 'douglas_peucker'
    }

def vectorize_outline_stroke_plan(stroke_plan, max_error=1.0):
    """
    Vectorize all strokes in an outline stroke plan
    
    Args:
        stroke_plan: Stroke plan from create_contour_stroke_plan()
        max_error: Maximum simplification error in pixels
    
    Returns:
        dict: Vectorized stroke plan
    """
    if stroke_plan is None:
        return None
    
    vectorized_strokes = []
    total_original_points = 0
    total_simplified_points = 0
    total_error = 0
    
    for stroke in stroke_plan['strokes']:
        vectorized_stroke = vectorize_outline_stroke(stroke, max_error)
        vectorized_strokes.append(vectorized_stroke)
        
        total_original_points += vectorized_stroke['num_points_original']
        total_simplified_points += vectorized_stroke['num_points_simplified']
        total_error += vectorized_stroke['simplification_error']
    
    # Calculate overall statistics
    avg_compression = total_original_points / total_simplified_points if total_simplified_points > 0 else 1.0
    avg_error = total_error / len(vectorized_strokes) if vectorized_strokes else 0.0
    
    # Create compatible structure for downstream processing
    # Convert outline strokes to semantic phase format expected by sampling/color detection
    boundary_strokes = []
    for stroke in vectorized_strokes:
        # Convert to format expected by later steps
        compatible_stroke = {
            'original_points': stroke['original_path'],
            'simplified_points': stroke['simplified_path'],
            'vectorized_points': stroke['simplified_path'],  # Required by sampling step
            'length': stroke['length'],
            'is_closed': stroke.get('is_closed', True),
            'compression_ratio': stroke['compression_ratio'],
            'simplification_error': stroke['simplification_error'],
            'success': True,
            'phase': 'boundary',  # All outline strokes are boundary strokes
            'stroke_id': stroke['id'],
            'method': 'outline_tracing'
        }
        boundary_strokes.append(compatible_stroke)
    
    # Return in semantic phase structure expected by sampling step
    return {
        'boundary': boundary_strokes,
        'internal': [],  # No internal strokes in outline-only mode
        'detail': [],    # No detail strokes in outline-only mode
        'method': 'outline_vectorization',
        'metadata': {
            'total_strokes': len(vectorized_strokes),
            'total_points_original': total_original_points,
            'total_points_simplified': total_simplified_points,
            'average_compression': avg_compression,
            'average_error': avg_error,
            'max_error_threshold': max_error
        },
        'stats': stroke_plan['stats']  # Preserve original stats
    }

def process_all_outline_vectorizations(stroke_plans, max_error=1.0):
    """
    Process all outline stroke plans for vectorization
    
    Args:
        stroke_plans: List of outline stroke plans
        max_error: Maximum simplification error in pixels
    
    Returns:
        list: List of vectorized stroke plans
    """
    print(f"   🔧 Vectorizing {len(stroke_plans)} outline stroke plans (max error: {max_error}px)...")
    
    vectorized_results = []
    total_strokes = 0
    total_original_points = 0
    total_simplified_points = 0
    
    for i, stroke_plan in enumerate(stroke_plans):
        print(f"   🔧 Vectorizing stroke plan {i+1}/{len(stroke_plans)}...")
        
        vectorized_plan = vectorize_outline_stroke_plan(stroke_plan, max_error)
        
        if vectorized_plan is not None:
            vectorized_results.append(vectorized_plan)
            metadata = vectorized_plan['metadata']
            stroke_count = metadata['total_strokes']
            total_strokes += stroke_count
            total_original_points += metadata['total_points_original']
            total_simplified_points += metadata['total_points_simplified']
            
            print(f"      📊 Strokes: {stroke_count}")
            print(f"      📉 Compression: {metadata['average_compression']:.2f}x ({metadata['total_points_original']} → {metadata['total_points_simplified']} points)")
            print(f"      📏 Avg error: {metadata['average_error']:.2f} pixels")
        else:
            print(f"      ⚠️  No vectorization created, skipping...")
            vectorized_results.append(None)
    
    successful_vectorizations = len([r for r in vectorized_results if r is not None])
    overall_compression = total_original_points / total_simplified_points if total_simplified_points > 0 else 1.0
    
    print(f"   ✅ Success: {successful_vectorizations}/{len(stroke_plans)} vectorizations")
    print(f"   🔧 Total strokes: {total_strokes}")
    print(f"   📉 Overall compression: {overall_compression:.2f}x ({total_original_points:,} → {total_simplified_points:,} points)")
    
    return vectorized_results

def simplify_polyline_rdp(pts, epsilon=2.0):
    """
    Simplify polyline using RDP (Ramer-Douglas-Peucker) algorithm

    Args:
        pts: numpy.ndarray - Array of (x, y) points
        epsilon: float - Simplification tolerance (higher = more aggressive)

    Returns:
        numpy.ndarray: Simplified polyline points
    """
    if len(pts) < 3:
        return pts

    # Convert to float coordinates for precision
    pts_float = pts.astype(np.float64)

    if RDP_AVAILABLE:
        # Use optimized RDP library
        simplified = rdp(pts_float, epsilon=epsilon)
    else:
        # Fallback: basic RDP implementation
        simplified = rdp_fallback(pts_float, epsilon)

    return simplified

def rdp_fallback(pts, epsilon):
    """
    Basic RDP implementation when rdp library not available

    Args:
        pts: numpy.ndarray - Array of points
        epsilon: float - Tolerance threshold

    Returns:
        numpy.ndarray: Simplified points
    """
    if len(pts) <= 2:
        return pts

    # Find the point with maximum distance from line segment
    dmax = 0
    index = 0
    end = len(pts) - 1

    for i in range(1, end):
        d = point_line_distance(pts[i], pts[0], pts[end])
        if d > dmax:
            index = i
            dmax = d

    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        # Recursive call
        rec_results1 = rdp_fallback(pts[:index+1], epsilon)
        rec_results2 = rdp_fallback(pts[index:], epsilon)

        # Build the result list
        result = np.vstack([rec_results1[:-1], rec_results2])
    else:
        result = np.array([pts[0], pts[end]])

    return result

def point_line_distance(point, line_start, line_end):
    """
    Calculate perpendicular distance from point to line segment

    Args:
        point: numpy.ndarray - Point coordinates
        line_start: numpy.ndarray - Line start point
        line_end: numpy.ndarray - Line end point

    Returns:
        float: Distance from point to line
    """
    if np.array_equal(line_start, line_end):
        return np.linalg.norm(point - line_start)

    # Calculate distance using cross product formula
    numerator = abs(np.cross(line_end - line_start, line_start - point))
    denominator = np.linalg.norm(line_end - line_start)

    return numerator / denominator

def fit_spline(pts, smoothing_factor=0.1, num_points=None):
    """
    Fit smooth spline to points for robot-friendly trajectories

    Args:
        pts: numpy.ndarray - Array of (x, y) points
        smoothing_factor: float - Spline smoothing parameter (0=interpolation, >0=approximation)
        num_points: int - Number of output points (default: 2x input)

    Returns:
        numpy.ndarray: Smooth spline points
    """
    if not SCIPY_AVAILABLE:
        print("Warning: SciPy not available, using linear interpolation")
        return linear_interpolation(pts, num_points or len(pts) * 2)

    if len(pts) < 4:
        return pts

    try:
        # Remove duplicate points
        pts_clean = remove_duplicates(pts)
        if len(pts_clean) < 4:
            return pts_clean

        # Fit parametric spline
        tck, u = splprep([pts_clean[:, 0], pts_clean[:, 1]],
                        s=smoothing_factor * len(pts_clean),
                        per=False)

        # Evaluate spline at desired number of points
        if num_points is None:
            num_points = len(pts) * 2

        u_new = np.linspace(0, 1, num_points)
        spline_x, spline_y = splev(u_new, tck)

        return np.column_stack([spline_x, spline_y])

    except Exception as e:
        print(f"Warning: Spline fitting failed ({e}), using original points")
        return pts

def linear_interpolation(pts, num_points):
    """
    Simple linear interpolation fallback when SciPy unavailable

    Args:
        pts: numpy.ndarray - Input points
        num_points: int - Desired number of output points

    Returns:
        numpy.ndarray: Interpolated points
    """
    if len(pts) < 2:
        return pts

    # Calculate cumulative distances
    distances = np.cumsum(np.r_[0, np.linalg.norm(np.diff(pts, axis=0), axis=1)])
    total_distance = distances[-1]

    if total_distance == 0:
        return pts

    # Create new parameter values
    new_distances = np.linspace(0, total_distance, num_points)

    # Interpolate x and y coordinates
    new_x = np.interp(new_distances, distances, pts[:, 0])
    new_y = np.interp(new_distances, distances, pts[:, 1])

    return np.column_stack([new_x, new_y])

def remove_duplicates(pts, tolerance=1e-6):
    """
    Remove duplicate consecutive points

    Args:
        pts: numpy.ndarray - Input points
        tolerance: float - Distance threshold for duplicate detection

    Returns:
        numpy.ndarray: Points with duplicates removed
    """
    if len(pts) <= 1:
        return pts

    unique_pts = [pts[0]]

    for i in range(1, len(pts)):
        distance = np.linalg.norm(pts[i] - unique_pts[-1])
        if distance > tolerance:
            unique_pts.append(pts[i])

    return np.array(unique_pts)

def adaptive_simplification(pts, max_error=1.0, method='rdp'):
    """
    Adaptively choose simplification parameters to meet error tolerance

    Args:
        pts: numpy.ndarray - Input points
        max_error: float - Maximum allowable approximation error
        method: str - Simplification method ('rdp' or 'spline')

    Returns:
        dict: Simplified points and metadata
    """
    if len(pts) < 3:
        return {
            'points': pts,
            'method': method,
            'parameter': 0.0,
            'error': 0.0,
            'compression_ratio': 1.0,
            'success': True
        }

    original_count = len(pts)

    if method == 'rdp':
        # Try different epsilon values for RDP
        epsilon = 0.5
        max_epsilon = 5.0

        while epsilon <= max_epsilon:
            simplified = simplify_polyline_rdp(pts, epsilon=epsilon)
            error = calculate_approximation_error(pts, simplified)

            if error <= max_error:
                compression_ratio = len(simplified) / original_count
                return {
                    'points': simplified,
                    'method': 'rdp',
                    'parameter': epsilon,
                    'error': error,
                    'compression_ratio': compression_ratio,
                    'success': True
                }

            epsilon += 0.5

        # If no solution found, return best attempt
        simplified = simplify_polyline_rdp(pts, epsilon=max_epsilon)
        error = calculate_approximation_error(pts, simplified)
        compression_ratio = len(simplified) / original_count

        return {
            'points': simplified,
            'method': 'rdp',
            'parameter': max_epsilon,
            'error': error,
            'compression_ratio': compression_ratio,
            'success': False
        }

    elif method == 'spline':
        # Try different smoothing factors for splines
        smoothing_factors = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

        for smoothing in smoothing_factors:
            simplified = fit_spline(pts, smoothing_factor=smoothing,
                                  num_points=max(10, original_count // 2))
            error = calculate_approximation_error(pts, simplified)

            if error <= max_error:
                compression_ratio = len(simplified) / original_count
                return {
                    'points': simplified,
                    'method': 'spline',
                    'parameter': smoothing,
                    'error': error,
                    'compression_ratio': compression_ratio,
                    'success': True
                }

        # Return best attempt
        simplified = fit_spline(pts, smoothing_factor=5.0,
                              num_points=max(10, original_count // 2))
        error = calculate_approximation_error(pts, simplified)
        compression_ratio = len(simplified) / original_count

        return {
            'points': simplified,
            'method': 'spline',
            'parameter': 5.0,
            'error': error,
            'compression_ratio': compression_ratio,
            'success': False
        }

def calculate_approximation_error(original, simplified):
    """
    Calculate maximum deviation between original and simplified curves

    Args:
        original: numpy.ndarray - Original points
        simplified: numpy.ndarray - Simplified points

    Returns:
        float: Maximum approximation error
    """
    if not SCIPY_AVAILABLE:
        # Simple fallback: average distance to nearest simplified point
        if len(simplified) == 0:
            return float('inf')

        errors = []
        for orig_pt in original:
            distances = [np.linalg.norm(orig_pt - simp_pt) for simp_pt in simplified]
            errors.append(min(distances))

        return max(errors) if errors else 0.0

    # Use scipy for accurate distance calculation
    if len(simplified) == 0:
        return float('inf')

    distances = cdist(original, simplified)
    min_distances = np.min(distances, axis=1)

    return np.max(min_distances)

def vectorize_stroke_sequences(stroke_graph, method='adaptive', max_error=1.0):
    """
    Vectorize all stroke sequences from a stroke graph using semantic ordering

    Args:
        stroke_graph: StrokeGraph object from Step 4
        method: str - Vectorization method ('rdp', 'spline', 'adaptive')
        max_error: float - Maximum approximation error for adaptive method

    Returns:
        dict: Vectorized strokes organized by semantic phase
    """
    # Get semantic drawing order
    semantic_order = stroke_graph.semantic_order

    vectorized_phases = {
        'boundary': [],
        'internal': [],
        'detail': [],
        'metadata': {
            'total_strokes': 0,
            'total_points_original': 0,
            'total_points_simplified': 0,
            'average_compression': 0.0,
            'average_error': 0.0
        }
    }

    # Process each semantic phase
    for phase_name, edges in semantic_order['drawing_phases'].items():
        phase_results = []

        for edge in edges:
            # Get original stroke points
            points = stroke_graph.get_edge_points(edge)

            if len(points) < 2:
                continue

            # Apply vectorization method
            if method == 'adaptive':
                # Choose best method adaptively
                rdp_result = adaptive_simplification(points, max_error, method='rdp')
                spline_result = adaptive_simplification(points, max_error, method='spline')

                # Choose result with better compression while meeting error tolerance
                if rdp_result['success'] and spline_result['success']:
                    if rdp_result['compression_ratio'] <= spline_result['compression_ratio']:
                        result = rdp_result
                    else:
                        result = spline_result
                elif rdp_result['success']:
                    result = rdp_result
                elif spline_result['success']:
                    result = spline_result
                else:
                    # Neither succeeded, choose RDP as fallback
                    result = rdp_result

            elif method == 'rdp':
                result = adaptive_simplification(points, max_error, method='rdp')

            elif method == 'spline':
                result = adaptive_simplification(points, max_error, method='spline')

            else:
                raise ValueError(f"Unknown vectorization method: {method}")

            # Add stroke metadata
            stroke_result = {
                'edge': edge,
                'original_points': points,
                'vectorized_points': result['points'],
                'method': result['method'],
                'parameter': result['parameter'],
                'error': result['error'],
                'compression_ratio': result['compression_ratio'],
                'success': result['success'],
                'phase': phase_name
            }

            phase_results.append(stroke_result)

        vectorized_phases[phase_name] = phase_results

    # Calculate overall statistics
    all_results = []
    for phase in ['boundary', 'internal', 'detail']:
        all_results.extend(vectorized_phases[phase])

    if all_results:
        total_original = sum(len(r['original_points']) for r in all_results)
        total_simplified = sum(len(r['vectorized_points']) for r in all_results)
        avg_compression = np.mean([r['compression_ratio'] for r in all_results])
        avg_error = np.mean([r['error'] for r in all_results])

        vectorized_phases['metadata'] = {
            'total_strokes': len(all_results),
            'total_points_original': total_original,
            'total_points_simplified': total_simplified,
            'average_compression': avg_compression,
            'average_error': avg_error,
            'successful_vectorizations': sum(1 for r in all_results if r['success'])
        }

    return vectorized_phases

def process_all_vectorizations(stroke_graphs, method='adaptive', max_error=1.0):
    """
    Process all stroke graphs through vectorization

    Args:
        stroke_graphs: List of StrokeGraph objects from Step 4
        method: str - Vectorization method
        max_error: float - Maximum approximation error

    Returns:
        list: List of vectorized results for each graph
    """
    vectorized_results = []

    for i, graph in enumerate(stroke_graphs):
        if graph is not None:
            print(f"   🔧 Vectorizing graph {i+1}...")

            result = vectorize_stroke_sequences(graph, method=method, max_error=max_error)

            # Print statistics
            meta = result['metadata']
            print(f"      📊 Strokes: {meta['total_strokes']}")
            print(f"      📉 Compression: {meta['average_compression']:.2f}x "
                  f"({meta['total_points_original']} → {meta['total_points_simplified']} points)")
            print(f"      📏 Avg error: {meta['average_error']:.2f} pixels")
            print(f"      ✅ Success: {meta['successful_vectorizations']}/{meta['total_strokes']}")

            vectorized_results.append(result)
        else:
            print(f"   ⚠️  Skipping empty graph {i+1}")
            vectorized_results.append(None)

    return vectorized_results

def save_vectorization_results(vectorized_results, output_dir="results/step5_vectorization"):
    """
    Save vectorization visualization and results with consistent scaling

    Args:
        vectorized_results: List of vectorization results
        output_dir: Directory to save results

    Returns:
        str: Path to saved visualization
    """
    import sys
    from pathlib import Path
    import matplotlib.pyplot as plt
    sys.path.append(str(Path(__file__).parent.parent))
    
    from visualization_utils import (
        get_consistent_figure_layout, set_consistent_axis_properties,
        save_visualization_with_timestamp, hide_unused_subplots,
        normalize_coordinates_to_display, SEMANTIC_COLORS, SEMANTIC_ALPHAS,
        STANDARD_TARGET_SIZE
    )

    # Create consistent figure layout
    fig, axes = get_consistent_figure_layout()

    valid_results = [r for r in vectorized_results if r is not None]

    for i, result in enumerate(valid_results[:6]):
        if i >= 6:
            break

        ax = axes[i]

        # Determine original shape from first available stroke points
        original_shape = None
        for phase_name in ['boundary', 'internal', 'detail']:
            phase_strokes = result[phase_name]
            for stroke in phase_strokes:
                if len(stroke['original_points']) > 0:
                    # Estimate original shape from coordinate ranges
                    all_points = stroke['original_points']
                    max_y = np.max(all_points[:, 0]) if len(all_points) > 0 else STANDARD_TARGET_SIZE
                    max_x = np.max(all_points[:, 1]) if len(all_points) > 0 else STANDARD_TARGET_SIZE
                    original_shape = (int(max_y) + 50, int(max_x) + 50)  # Add padding
                    break
            if original_shape:
                break
        
        if original_shape is None:
            original_shape = (STANDARD_TARGET_SIZE, STANDARD_TARGET_SIZE)

        # Plot each semantic phase with consistent scaling
        for phase_name in ['boundary', 'internal', 'detail']:
            phase_strokes = result[phase_name]

            for stroke in phase_strokes:
                original_pts = stroke['original_points']
                vectorized_pts = stroke['vectorized_points']

                if len(original_pts) > 0 and len(vectorized_pts) > 0:
                    color = SEMANTIC_COLORS[phase_name]
                    alpha = SEMANTIC_ALPHAS[phase_name]

                    # Scale coordinates to consistent size
                    scaled_original = normalize_coordinates_to_display(
                        original_pts, original_shape, STANDARD_TARGET_SIZE
                    )
                    scaled_vectorized = normalize_coordinates_to_display(
                        vectorized_pts, original_shape, STANDARD_TARGET_SIZE
                    )

                    if len(scaled_original) > 0 and len(scaled_vectorized) > 0:
                        # Plot original points as thin gray line
                        ax.plot(scaled_original[:, 1], scaled_original[:, 0],
                               color='lightgray', linewidth=0.5, alpha=0.5)

                        # Plot vectorized result as colored line
                        ax.plot(scaled_vectorized[:, 1], scaled_vectorized[:, 0],
                               color=color, linewidth=2, alpha=alpha)

                        # Mark control points
                        ax.scatter(scaled_vectorized[:, 1], scaled_vectorized[:, 0],
                                 c=color, s=8, alpha=alpha, zorder=5)

        # Set title with statistics
        meta = result['metadata']
        title = (f'Graph {i+1}: Vectorized\n'
                f'{meta["total_strokes"]} strokes, {meta["average_compression"]:.2f}x compression\n'
                f'Error: {meta["average_error"]:.1f}px, Success: {meta["successful_vectorizations"]}/{meta["total_strokes"]}')

        # Set consistent axis properties
        set_consistent_axis_properties(ax, title, STANDARD_TARGET_SIZE)

    # Hide unused subplots
    hide_unused_subplots(axes, len(valid_results))

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=2, label='Boundary strokes'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Internal strokes'),
        plt.Line2D([0], [0], color='green', lw=2, label='Detail strokes'),
        plt.Line2D([0], [0], color='lightgray', lw=1, label='Original pixels')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4)

    plt.suptitle(f'Step 5: Vectorization Results', fontsize=16)
    plt.tight_layout()

    # Save with consistent timestamp and path handling
    return save_visualization_with_timestamp(fig, output_dir, 'vectorization_results')