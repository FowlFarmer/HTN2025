#!/usr/bin/env python3
"""
Step 8: Flattening & Stroke Ordering (Travel Optimization)

Produce a single ordered array of strokes with coordinates, color temperature,
and timing metadata for robotic orchestration. This is the final step that
prepares data for hardware execution.

Features:
- Complete stroke flattening from hierarchical structure
- Travel distance optimization using TSP algorithms
- Timing calculation based on curvature analysis
- Pen lift planning for safe robot movement
- Multiple export formats (JSON, G-code, CSV)
- Comprehensive metadata for hardware integration
"""

import numpy as np
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import copy
import sys
import os

# Add project root to path for constants import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import (
    CANVAS_SIZE_MM, CANVAS_WIDTH_MM, CANVAS_HEIGHT_MM,
    ROBOT_CONFIG
)

class StrokeData:
    """
    Complete stroke data structure for robotic painting execution

    This class contains all information needed by the hardware engineer
    to execute the painting on a robotic system.
    """

    def __init__(self, canvas_mm: List[float] = [CANVAS_WIDTH_MM, CANVAS_HEIGHT_MM]):
        self.meta = {
            'canvas_mm': canvas_mm,  # Canvas dimensions in mm
            'scale_mm_per_px': 0.0,  # Will be set from sampling results
            'total_strokes': 0,
            'total_length_mm': 0.0,
            'total_pen_lifts': 0,
            'estimated_travel_distance_mm': 0.0,
            'processing_timestamp': datetime.now().isoformat(),
            'optimization_method': ''
        }
        self.strokes = []
        self.pen_lifts = []

    def add_stroke(self, stroke_id: int, element_id: int, warmth: int,
                   points: List[List[float]], length_mm: float,
                   start_pos: List[float], end_pos: List[float], 
                   phase: str = "unknown", **kwargs):
        """
        Add a stroke to the data structure

        Args:
            stroke_id: Unique identifier for this stroke
            element_id: ID of the image element this stroke belongs to
            warmth: Color temperature (0=warm, 1=cool)
            points: List of [x_mm, y_mm] coordinates
            length_mm: Total stroke length in millimeters
            start_pos: Starting position [x_mm, y_mm]
            end_pos: Ending position [x_mm, y_mm]
            phase: Semantic phase ('boundary', 'internal', 'detail')
            **kwargs: Additional stroke metadata
        """
        stroke = {
            'stroke_id': stroke_id,
            'element_id': element_id,
            'warmth': warmth,  # 0=warm, 1=cool
            'warmth_name': 'warm' if warmth == 0 else 'cool',
            'phase': phase,
            'points': points,  # List of [x_mm, y_mm] coordinates
            'length_mm': length_mm,
            'start_pos': start_pos,  # [x_mm, y_mm]
            'end_pos': end_pos,      # [x_mm, y_mm]
            'order_index': len(self.strokes)  # Original order before optimization
        }

        self.strokes.append(stroke)
        self.meta['total_strokes'] += 1
        self.meta['total_length_mm'] += length_mm

    def add_pen_lift(self, from_stroke_id: int, to_stroke_id: int,
                     travel_distance_mm: float, height_mm: float = ROBOT_CONFIG['pen_lift_height_mm']):
        """
        Add a pen lift instruction between strokes

        Args:
            from_stroke_id: ID of stroke being left
            to_stroke_id: ID of stroke being approached
            travel_distance_mm: Distance traveled with pen up
            height_mm: Height to lift pen (safety clearance)
        """
        pen_lift = {
            'type': 'pen_lift',
            'from_stroke_id': from_stroke_id,
            'to_stroke_id': to_stroke_id,
            'travel_distance_mm': travel_distance_mm,
            'height_mm': height_mm,
            'duration_s': travel_distance_mm / 50.0  # Assume 50mm/s travel speed
        }

        self.pen_lifts.append(pen_lift)
        self.meta['total_pen_lifts'] += 1
        self.meta['estimated_travel_distance_mm'] += travel_distance_mm
        self.meta['estimated_duration_s'] += pen_lift['duration_s']

def flatten_strokes_from_results(color_detection_results: Dict) -> StrokeData:
    """
    Flatten all strokes from hierarchical structure into single ordered array

    Args:
        color_detection_results: Results from Step 7 with color classifications

    Returns:
        StrokeData: Flattened stroke data ready for optimization
    """
    enhanced_sampling = color_detection_results['enhanced_sampling_results']
    stroke_data = StrokeData(canvas_mm=[160.0, 160.0])

    # Set scale from sampling results
    stroke_data.meta['scale_mm_per_px'] = enhanced_sampling['scale_info']['scale_mm_per_px']

    stroke_id = 0

    # Process each graph (image element)
    for element_id, graph_result in enumerate(enhanced_sampling['sampled_graphs']):
        if graph_result is None:
            continue

        # Get color information for this element
        color_info = graph_result.get('color_info', {})
        warmth = color_info.get('warmth_class', 1)  # Default to cool if unknown

        # Process each semantic phase in order
        for phase_name in ['boundary', 'internal', 'detail']:
            phase_strokes = graph_result.get(phase_name, [])

            for stroke in phase_strokes:
                if not stroke.get('success', False) or len(stroke.get('sampled_points_mm', [])) < 2:
                    continue

                points_mm = stroke['sampled_points_mm']
                if hasattr(points_mm, 'tolist'):
                    points = points_mm.tolist()
                else:
                    points = points_mm
                start_pos = points[0]
                end_pos = points[-1]
                length_mm = stroke.get('total_length_mm', 0.0)

                # Stroke calculations removed - handled by firmware engineer
                stroke_data.add_stroke(
                    stroke_id=stroke_id,
                    element_id=element_id,
                    warmth=warmth,
                    points=points,
                    length_mm=length_mm,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    phase=phase_name
                )

                stroke_id += 1

    print(f"   📊 Flattened {stroke_data.meta['total_strokes']} strokes from {len([g for g in enhanced_sampling['sampled_graphs'] if g is not None])} elements")
    print(f"   📏 Total drawing length: {stroke_data.meta['total_length_mm']:.1f}mm")

    return stroke_data

def optimize_stroke_sequence(stroke_data: StrokeData, strategy: str = 'hybrid') -> StrokeData:
    """
    Optimize order of strokes across all elements

    Args:
        stroke_data: Flattened stroke data
        strategy: Optimization strategy
            - 'salience_first': Draw most salient elements first
            - 'travel_minimization': Minimize travel distance (TSP-based)
            - 'semantic_preserve': Maintain semantic ordering (boundary→internal→detail)
            - 'hybrid': Combine semantic ordering with travel optimization

    Returns:
        StrokeData: Optimized stroke data
    """
    stroke_data.meta['optimization_method'] = strategy

    if strategy == 'travel_minimization':
        return optimize_by_travel_distance(stroke_data)
    elif strategy == 'semantic_preserve':
        return optimize_by_semantic_order(stroke_data)
    elif strategy == 'hybrid':
        return optimize_hybrid(stroke_data)
    else:
        # Default: maintain original order
        return stroke_data

def optimize_by_travel_distance(stroke_data: StrokeData) -> StrokeData:
    """
    Fast travel distance optimization using greedy nearest-neighbor

    Uses a greedy nearest-neighbor approach without expensive 2-opt improvements
    to minimize the total pen-up travel distance between strokes.
    """
    print("   🎯 Optimizing stroke order for minimum travel distance (fast mode)...")

    if len(stroke_data.strokes) <= 1:
        return stroke_data

    # For very large stroke sets, use spatial sorting instead of TSP
    if len(stroke_data.strokes) > 100:
        print(f"   ⚡ Using spatial sorting for {len(stroke_data.strokes)} strokes (too many for TSP)")
        # Simple spatial sorting by diagonal distance from origin
        sorted_strokes = sorted(stroke_data.strokes, 
                              key=lambda s: (s['start_pos'][0] + s['start_pos'][1]))
        stroke_data.strokes = sorted_strokes
    else:
        # Extract stroke positions for TSP
        stroke_positions = [(i, stroke['start_pos']) for i, stroke in enumerate(stroke_data.strokes)]

        # Solve using greedy nearest neighbor only (skip expensive 2-opt)
        ordered_indices = solve_tsp_greedy(stroke_positions)

        # Reorder strokes
        original_strokes = copy.deepcopy(stroke_data.strokes)
        stroke_data.strokes = [original_strokes[i] for i in ordered_indices]

    # Calculate pen lifts for optimized order
    calculate_pen_lifts(stroke_data)

    travel_distance = sum([lift['travel_distance_mm'] for lift in stroke_data.pen_lifts])
    print(f"   ✅ Optimized order: {travel_distance:.1f}mm total travel distance")

    return stroke_data

def optimize_hybrid(stroke_data: StrokeData) -> StrokeData:
    """
    Fast hybrid optimization: Group by semantic phase with minimal travel optimization

    This preserves the semantic meaning (boundary→internal→detail) while
    doing lightweight travel optimization within each phase.
    """
    print("   🎯 Optimizing stroke order with fast hybrid strategy...")

    # Group strokes by phase
    phase_groups = {'boundary': [], 'internal': [], 'detail': []}
    for i, stroke in enumerate(stroke_data.strokes):
        phase = stroke.get('phase', 'detail')
        phase_groups[phase].append((i, stroke))

    optimized_strokes = []

    # Optimize each phase separately with performance limits
    for phase_name in ['boundary', 'internal', 'detail']:
        if not phase_groups[phase_name]:
            continue

        # For performance, only optimize small groups with expensive TSP
        if len(phase_groups[phase_name]) <= 20:
            # Extract positions for this phase
            phase_positions = [(i, stroke['start_pos']) for i, stroke in phase_groups[phase_name]]

            if len(phase_positions) > 1:
                # Light TSP optimization for small groups only
                ordered_indices = solve_tsp_greedy(phase_positions)
                # Skip expensive 2-opt for performance
                
                # Add optimized strokes from this phase
                for tsp_idx in ordered_indices:
                    if tsp_idx < len(phase_groups[phase_name]):
                        original_idx, stroke = phase_groups[phase_name][tsp_idx]
                        optimized_strokes.append(stroke)
            else:
                # Single stroke in phase
                optimized_strokes.extend([stroke for _, stroke in phase_groups[phase_name]])
        else:
            # For large groups, use simple spatial sorting instead of expensive TSP
            print(f"   ⚡ Using fast spatial sorting for {len(phase_groups[phase_name])} {phase_name} strokes")
            sorted_strokes = sorted(phase_groups[phase_name], 
                                  key=lambda x: (x[1]['start_pos'][0] + x[1]['start_pos'][1]))
            optimized_strokes.extend([stroke for _, stroke in sorted_strokes])

    stroke_data.strokes = optimized_strokes

    # Calculate pen lifts for optimized order
    calculate_pen_lifts(stroke_data)

    travel_distance = sum([lift['travel_distance_mm'] for lift in stroke_data.pen_lifts])
    print(f"   ✅ Hybrid optimization: {travel_distance:.1f}mm total travel distance")

    return stroke_data

def solve_tsp_greedy(stroke_positions: List[Tuple[int, List[float]]]) -> List[int]:
    """
    Greedy TSP solution starting from canvas center

    Args:
        stroke_positions: List of (index, [x, y]) tuples

    Returns:
        List of indices in optimized order
    """
    if len(stroke_positions) <= 1:
        return [pos[0] for pos in stroke_positions]

    canvas_center = [CANVAS_WIDTH_MM/2, CANVAS_HEIGHT_MM/2]  # Center of canvas
    unvisited = set(range(len(stroke_positions)))
    path = []
    current_pos = canvas_center

    while unvisited:
        # Find nearest unvisited stroke
        nearest_idx = min(unvisited,
                         key=lambda i: np.linalg.norm(
                             np.array(stroke_positions[i][1]) - np.array(current_pos)
                         ))

        path.append(stroke_positions[nearest_idx][0])  # Add original index
        unvisited.remove(nearest_idx)
        current_pos = stroke_positions[nearest_idx][1]

    return path

def improve_with_2opt(path: List[int], stroke_positions: List[Tuple[int, List[float]]]) -> List[int]:
    """
    Improve TSP solution using 2-opt local search

    Args:
        path: Current path as list of indices
        stroke_positions: List of (index, [x, y]) tuples

    Returns:
        Improved path
    """
    if len(path) <= 3:
        return path

    # Create position lookup
    pos_lookup = {stroke_positions[i][0]: stroke_positions[i][1] for i in range(len(stroke_positions))}

    improved = True
    iterations = 0
    max_iterations = 50  # Prevent infinite loops

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        for i in range(1, len(path) - 1):
            for j in range(i + 1, len(path)):
                # Try 2-opt swap
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]

                if calculate_total_distance(new_path, pos_lookup) < calculate_total_distance(path, pos_lookup):
                    path = new_path
                    improved = True
                    break
            if improved:
                break

    return path

def calculate_total_distance(path: List[int], pos_lookup: Dict[int, List[float]]) -> float:
    """Calculate total travel distance for a given path"""
    if len(path) <= 1:
        return 0.0

    total_distance = 0.0
    canvas_center = [CANVAS_WIDTH_MM/2, CANVAS_HEIGHT_MM/2]
    current_pos = canvas_center

    for stroke_id in path:
        stroke_pos = pos_lookup[stroke_id]
        distance = np.linalg.norm(np.array(stroke_pos) - np.array(current_pos))
        total_distance += distance
        current_pos = stroke_pos

    return total_distance

def calculate_pen_lifts(stroke_data: StrokeData, min_travel_distance_mm: float = ROBOT_CONFIG['min_travel_distance_mm']):
    """
    Calculate pen lifts between consecutive strokes

    Args:
        stroke_data: Stroke data to analyze
        min_travel_distance_mm: Minimum distance requiring pen lift
    """
    stroke_data.pen_lifts = []  # Reset pen lifts
    stroke_data.meta['total_pen_lifts'] = 0
    stroke_data.meta['estimated_travel_distance_mm'] = 0.0

    for i in range(1, len(stroke_data.strokes)):
        current_stroke = stroke_data.strokes[i-1]
        next_stroke = stroke_data.strokes[i]

        # Calculate travel distance
        travel_distance = np.linalg.norm(
            np.array(next_stroke['start_pos']) - np.array(current_stroke['end_pos'])
        )

        if travel_distance > min_travel_distance_mm:
            stroke_data.add_pen_lift(
                from_stroke_id=current_stroke['stroke_id'],
                to_stroke_id=next_stroke['stroke_id'],
                travel_distance_mm=travel_distance
            )

# Timing and curvature calculation functions removed - handled by firmware engineer

def optimize_by_semantic_order(stroke_data: StrokeData) -> StrokeData:
    """
    Maintain semantic ordering: boundary → internal → detail strokes

    This preserves artistic meaning while minimizing travel within phases.
    """
    print("   🎯 Optimizing stroke order with semantic preservation...")

    # Group by phase and element
    groups = {}
    for stroke in stroke_data.strokes:
        key = (stroke.get('phase', 'detail'), stroke['element_id'])
        if key not in groups:
            groups[key] = []
        groups[key].append(stroke)

    # Order: boundary first, then internal, then detail
    phase_order = ['boundary', 'internal', 'detail']
    optimized_strokes = []

    for phase in phase_order:
        phase_groups = {k: v for k, v in groups.items() if k[0] == phase}

        # Add strokes from this phase, grouped by element
        for element_id in sorted(set(k[1] for k in phase_groups.keys())):
            key = (phase, element_id)
            if key in phase_groups:
                optimized_strokes.extend(phase_groups[key])

    stroke_data.strokes = optimized_strokes

    # Calculate pen lifts
    calculate_pen_lifts(stroke_data)

    travel_distance = sum([lift['travel_distance_mm'] for lift in stroke_data.pen_lifts])
    print(f"   ✅ Semantic ordering: {travel_distance:.1f}mm total travel distance")

    return stroke_data

def group_strokes_by_mask(stroke_data: StrokeData) -> List[Dict]:
    """
    Group optimized strokes by mask for narration timing
    
    Returns an array of arrays where each inner array represents strokes for one mask.
    This structure allows narration to be timed by the current mask being drawn.

    Args:
        stroke_data: Optimized stroke data

    Returns:
        List of mask arrays, each containing:
        {
            'mask_id': int,
            'warmth_class': int,
            'warmth_name': str,
            'strokes': List[stroke_objects],  # Array of strokes for this mask
            'total_length_mm': float,
            'total_duration_s': float,
            'stroke_count': int
        }
    """
    # Group strokes by mask ID (element_id corresponds to mask)
    mask_groups = {}

    for stroke in stroke_data.strokes:
        mask_id = stroke['element_id']  # element_id corresponds to mask

        if mask_id not in mask_groups:
            mask_groups[mask_id] = {
                'mask_id': mask_id,
                'warmth_class': stroke['warmth'],
                'warmth_name': stroke['warmth_name'],
                'strokes': []  # This is the inner array for this mask
            }

        mask_groups[mask_id]['strokes'].append(stroke)

    # Convert to list sorted by mask ID for consistent ordering
    mask_stroke_arrays = []
    for mask_id in sorted(mask_groups.keys()):
        mask_group = mask_groups[mask_id]

        # Add statistics for the entire mask
        total_length = sum(s['length_mm'] for s in mask_group['strokes'])
        stroke_count = len(mask_group['strokes'])

        mask_group.update({
            'total_length_mm': total_length,
            'stroke_count': stroke_count
        })

        mask_stroke_arrays.append(mask_group)

    print(f"   📱 Grouped strokes into {len(mask_stroke_arrays)} mask arrays for narration")
    for i, mask_array in enumerate(mask_stroke_arrays):
        print(f"     • Array[{i}] - Mask {mask_array['mask_id']}: {mask_array['stroke_count']} strokes ({mask_array['warmth_name']})")

    return mask_stroke_arrays

# Timing functions removed - handled by firmware engineer

def process_all_stroke_ordering(color_detection_results: Dict, strategy: str = 'hybrid') -> Dict:
    """
    Complete stroke ordering process: flatten, optimize, and group by masks for narration

    Args:
        color_detection_results: Results from Step 7
        strategy: Optimization strategy

    Returns:
        Dict containing mask-grouped stroke arrays and statistics
    """
    print(f"   📋 Flattening stroke hierarchy...")

    # Step 1: Flatten all strokes from hierarchical structure
    stroke_data = flatten_strokes_from_results(color_detection_results)

    if stroke_data.meta['total_strokes'] == 0:
        print("   ⚠️  No valid strokes found for ordering")
        return {
            'mask_stroke_arrays': [],
            'statistics': {
                'total_strokes': 0,
                'total_masks': 0,
                'optimization_method': strategy,
                'success': False
            }
        }

    print(f"   🎯 Optimizing stroke order using '{strategy}' strategy...")

    # Step 2: Optimize stroke order
    optimized_data = optimize_stroke_sequence(stroke_data, strategy)

    # Step 3: Group strokes by mask for narration timing
    mask_stroke_arrays = group_strokes_by_mask(optimized_data)

    # Step 4: Generate comprehensive statistics
    statistics = {
        'total_strokes': optimized_data.meta['total_strokes'],
        'total_masks': len(mask_stroke_arrays),
        'total_length_mm': optimized_data.meta['total_length_mm'],
        'total_pen_lifts': optimized_data.meta['total_pen_lifts'],
        'estimated_travel_distance_mm': optimized_data.meta['estimated_travel_distance_mm'],
        'optimization_method': strategy,
        'success': True,
        'mask_distribution': [
            {
                'mask_id': mask_array['mask_id'],
                'stroke_count': len(mask_array['strokes']),
                'warmth_name': mask_array['warmth_name'],
                'total_length_mm': sum(s['length_mm'] for s in mask_array['strokes'])
            }
            for mask_array in mask_stroke_arrays
        ]
    }

    print(f"   ✅ Optimization complete!")
    print(f"   📊 {statistics['total_strokes']} strokes across {statistics['total_masks']} masks")
    print(f"   📏 Drawing: {statistics['total_length_mm']:.1f}mm, Travel: {statistics['estimated_travel_distance_mm']:.1f}mm")
    print(f"   🚀 Ready for firmware timing calculations")

    return {
        'mask_stroke_arrays': mask_stroke_arrays,
        'statistics': statistics
    }

def save_stroke_ordering_results(ordering_results: Dict, output_dir: str = "results/step8_stroke_ordering",
                                create_visualization: bool = False) -> str:
    """
    Save stroke ordering results with optimized performance

    Args:
        ordering_results: Results from stroke ordering process
        output_dir: Directory to save results
        create_visualization: Whether to create visualization (slow)

    Returns:
        Path to main results file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure results directory exists
    results_dir = Path(output_dir)
    if not results_dir.is_absolute():
        # Make relative paths relative to current working directory
        results_dir = Path.cwd() / results_dir
        
    print(f"   📁 Using output directory: {results_dir}")  # Debug output

    # Create directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save simplified JSON format with mask arrays only
    json_path = results_dir / f"stroke_ordering_results_{timestamp}.json"
    export_enhanced_json(ordering_results, str(json_path))

    print(f"   💾 Saved stroke ordering results:")
    print(f"   📊 JSON: {json_path}")

    return str(json_path)

def export_enhanced_json(ordering_results: Dict, filepath: str):
    """Export stroke data with mask arrays for narration"""
    output_data = {
        'mask_stroke_arrays': ordering_results['mask_stroke_arrays'],
        'statistics': ordering_results['statistics'],
        'format_version': '2.0',
        'description': 'Robotic painting stroke data grouped by mask for narration timing'
    }

    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

# Visualization functions removed - no longer needed for performance

def export_json(stroke_data: StrokeData, filepath: str):
    """Export stroke data as JSON (primary format for hardware integration)"""
    output_data = {
        'metadata': stroke_data.meta,
        'strokes': stroke_data.strokes,
        'pen_lifts': stroke_data.pen_lifts,
        'format_version': '1.0',
        'description': 'Robotic painting stroke data - optimized for execution'
    }

    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

def export_gcode(stroke_data: StrokeData, filepath: str):
    """Export stroke data as G-code for direct robot execution"""
    lines = [
        "; Robotic Painting G-code",
        f"; Generated: {stroke_data.meta['processing_timestamp']}",
        f"; Total strokes: {stroke_data.meta['total_strokes']}",
        f"; Estimated time: {stroke_data.meta['estimated_duration_s']:.1f}s",
        "",
        "G21 ; Set units to millimeters",
        "G90 ; Absolute positioning",
        "M3 ; Prepare pen/brush system",
        "G0 Z15 ; Lift pen to safe height",
        "G0 X80 Y80 ; Move to canvas center",
        ""
    ]

    for i, stroke in enumerate(stroke_data.strokes):
        lines.append(f"; Stroke {stroke['stroke_id']}: {stroke['phase']} ({stroke['warmth_name']})")
        lines.append(f"; Length: {stroke['length_mm']:.1f}mm, Speed: {stroke['speed_mm_s']:.1f}mm/s")

        # Move to start position
        start_x, start_y = stroke['start_pos']
        lines.append(f"G0 X{start_x:.2f} Y{start_y:.2f} ; Move to start")
        lines.append("G0 Z0 ; Lower pen")

        # Draw stroke
        feed_rate = stroke['speed_mm_s'] * 60  # Convert to mm/min for G-code
        for point in stroke['points']:
            x, y = point
            lines.append(f"G1 X{x:.2f} Y{y:.2f} F{feed_rate:.0f}")

        lines.append("G0 Z15 ; Lift pen")
        lines.append("")

    lines.extend([
        "G0 X0 Y0 ; Return to origin",
        "M5 ; Turn off pen/brush system",
        "; End of program"
    ])

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

def export_csv(stroke_data: StrokeData, filepath: str):
    """Export stroke data as CSV for analysis and debugging"""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'stroke_id', 'element_id', 'phase', 'warmth', 'warmth_name',
            'length_mm', 'duration_s', 'speed_mm_s', 'curvature_avg',
            'start_x', 'start_y', 'end_x', 'end_y', 'is_closed_loop',
            'point_count', 'order_index'
        ])

        # Write stroke data
        for stroke in stroke_data.strokes:
            writer.writerow([
                stroke['stroke_id'],
                stroke['element_id'],
                stroke['phase'],
                stroke['warmth'],
                stroke['warmth_name'],
                stroke['length_mm'],
                stroke['duration_s'],
                stroke['speed_mm_s'],
                stroke['curvature_avg'],
                stroke['start_pos'][0],
                stroke['start_pos'][1],
                stroke['end_pos'][0],
                stroke['end_pos'][1],
                stroke['is_closed_loop'],
                len(stroke['points']),
                stroke['order_index']
            ])

def plot_stroke_order_visualization(stroke_data: StrokeData, ax):
    """Plot stroke order with travel paths"""
    canvas_size = stroke_data.meta['canvas_mm'][0]

    # Plot strokes in order
    colors = {'boundary': 'red', 'internal': 'blue', 'detail': 'green'}

    for i, stroke in enumerate(stroke_data.strokes):
        points = np.array(stroke['points'])
        color = colors.get(stroke['phase'], 'gray')
        alpha = 0.8 if stroke['warmth'] == 0 else 0.6  # Warm strokes more opaque

        # Plot stroke
        ax.plot(points[:, 0], points[:, 1], color=color, alpha=alpha, linewidth=1.5)

        # Plot travel path to next stroke
        if i < len(stroke_data.strokes) - 1:
            next_stroke = stroke_data.strokes[i + 1]
            travel_path = [stroke['end_pos'], next_stroke['start_pos']]
            travel_points = np.array(travel_path)
            ax.plot(travel_points[:, 0], travel_points[:, 1],
                   'k--', alpha=0.3, linewidth=0.5)

    ax.set_xlim(0, canvas_size)
    ax.set_ylim(0, canvas_size)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

def plot_speed_profile(stroke_data: StrokeData, ax):
    """Plot speed profile across strokes"""
    stroke_indices = range(len(stroke_data.strokes))
    speeds = [stroke['speed_mm_s'] for stroke in stroke_data.strokes]
    phases = [stroke['phase'] for stroke in stroke_data.strokes]

    colors = {'boundary': 'red', 'internal': 'blue', 'detail': 'green'}

    for phase in ['boundary', 'internal', 'detail']:
        phase_indices = [i for i, p in enumerate(phases) if p == phase]
        phase_speeds = [speeds[i] for i in phase_indices]
        if phase_speeds:
            ax.scatter(phase_indices, phase_speeds,
                      c=colors[phase], alpha=0.7, label=phase.title())

    ax.set_xlabel('Stroke Index')
    ax.set_ylabel('Speed (mm/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_statistics(statistics: Dict, ax):
    """Plot execution statistics"""
    stats_text = f"""Total Strokes: {statistics['total_strokes']}
Drawing Length: {statistics['total_length_mm']:.1f} mm
Travel Distance: {statistics['estimated_travel_distance_mm']:.1f} mm
Pen Lifts: {statistics['total_pen_lifts']}
Estimated Time: {statistics['estimated_duration_s']:.1f} s ({statistics['estimated_duration_s']/60:.1f} min)
Optimization: {statistics['optimization_method']}"""

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontfamily='monospace', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def plot_distribution(statistics: Dict, ax):
    """Plot stroke distribution by color and phase"""
    dist = statistics['stroke_distribution']

    # Color distribution
    colors_data = [dist['warm_strokes'], dist['cool_strokes']]
    colors_labels = ['Warm', 'Cool']

    # Phase distribution
    phases_data = [dist['boundary_strokes'], dist['internal_strokes'], dist['detail_strokes']]
    phases_labels = ['Boundary', 'Internal', 'Detail']

    # Create pie charts
    ax.pie(colors_data, labels=colors_labels, autopct='%1.1f%%', startangle=90,
          colors=['#FF6B6B', '#4ECDC4'])
    ax.set_title('Color Temperature Distribution')