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
    Save vectorization visualization and results

    Args:
        vectorized_results: List of vectorization results
        output_dir: Directory to save results

    Returns:
        str: Path to saved visualization
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    valid_results = [r for r in vectorized_results if r is not None]

    for i, result in enumerate(valid_results[:6]):
        if i >= 6:
            break

        ax = axes[i]

        # Color scheme for semantic phases
        colors = {'boundary': 'red', 'internal': 'blue', 'detail': 'green'}
        alphas = {'boundary': 0.8, 'internal': 0.6, 'detail': 0.4}

        # Plot each semantic phase
        for phase_name in ['boundary', 'internal', 'detail']:
            phase_strokes = result[phase_name]

            for stroke in phase_strokes:
                original_pts = stroke['original_points']
                vectorized_pts = stroke['vectorized_points']

                if len(original_pts) > 0 and len(vectorized_pts) > 0:
                    color = colors[phase_name]
                    alpha = alphas[phase_name]

                    # Plot original points as thin gray line
                    ax.plot(original_pts[:, 1], original_pts[:, 0],
                           color='lightgray', linewidth=0.5, alpha=0.5)

                    # Plot vectorized result as colored line
                    ax.plot(vectorized_pts[:, 1], vectorized_pts[:, 0],
                           color=color, linewidth=2, alpha=alpha)

                    # Mark control points
                    ax.scatter(vectorized_pts[:, 1], vectorized_pts[:, 0],
                             c=color, s=8, alpha=alpha, zorder=5)

        # Set title with statistics
        meta = result['metadata']
        title = f'Graph {i+1}: Vectorized\n'
        title += f'{meta["total_strokes"]} strokes, {meta["average_compression"]:.2f}x compression\n'
        title += f'Error: {meta["average_error"]:.1f}px, Success: {meta["successful_vectorizations"]}/{meta["total_strokes"]}'

        ax.set_title(title, fontsize=10)
        ax.axis('off')
        ax.invert_yaxis()  # Match image coordinates

    # Hide unused subplots
    for i in range(len(valid_results), 6):
        axes[i].axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=2, label='Boundary strokes'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Internal strokes'),
        plt.Line2D([0], [0], color='green', lw=2, label='Detail strokes'),
        plt.Line2D([0], [0], color='lightgray', lw=1, label='Original pixels')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4)

    plt.suptitle(f'Step 5: Vectorization Results - {timestamp}', fontsize=16)
    plt.tight_layout()

    # Determine output path and create directory
    results_dir = Path(output_dir)
    if not results_dir.exists():
        results_dir = Path("results/step5_vectorization")
    if not results_dir.exists():
        results_dir = Path("../results/step5_vectorization")
    if not results_dir.exists():
        results_dir = Path(".")  # Fallback to current directory

    # Create directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"vectorization_results_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   💾 Saved vectorization visualization: {output_path}")

    return output_path