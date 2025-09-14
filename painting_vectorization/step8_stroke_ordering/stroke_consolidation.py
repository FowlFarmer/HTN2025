#!/usr/bin/env python3
"""
Stroke Consolidation - Combine Fragmented Strokes for Tactile Feedback

This module addresses the problem of having too many tiny strokes (1495 individual segments)
by consolidating connected stroke segments into longer, more meaningful drawing paths.

The goal is to create fewer, longer strokes that feel recognizable when drawn on someone's hand,
rather than thousands of tiny scratches that feel like chaos.
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import math

def calculate_stroke_length(points: List[List[float]]) -> float:
    """
    Calculate the actual length of a stroke in millimeters
    
    Args:
        points: List of [x, y] coordinates in mm
        
    Returns:
        Total length in millimeters
    """
    if len(points) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        total_length += math.sqrt(dx*dx + dy*dy)
    
    return total_length

def are_strokes_connected(stroke1: Dict, stroke2: Dict, max_gap_mm: float = 2.0) -> bool:
    """
    Check if two strokes can be connected (endpoints are close)
    
    Args:
        stroke1, stroke2: Stroke dictionaries with 'points'
        max_gap_mm: Maximum gap distance to consider connected
        
    Returns:
        True if strokes can be connected
    """
    if not stroke1['points'] or not stroke2['points']:
        return False
    
    # Check all possible connections between endpoints
    s1_start, s1_end = stroke1['points'][0], stroke1['points'][-1]
    s2_start, s2_end = stroke2['points'][0], stroke2['points'][-1]
    
    connections = [
        (s1_end, s2_start),  # s1 → s2
        (s1_end, s2_end),    # s1 → reverse s2
        (s1_start, s2_start), # reverse s1 → reverse s2
        (s1_start, s2_end)   # reverse s1 → s2
    ]
    
    for p1, p2 in connections:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        distance = math.sqrt(dx*dx + dy*dy)
        if distance <= max_gap_mm:
            return True
    
    return False

def connect_strokes(stroke1: Dict, stroke2: Dict) -> Dict:
    """
    Connect two strokes into a single longer stroke
    
    Args:
        stroke1, stroke2: Stroke dictionaries to connect
        
    Returns:
        New consolidated stroke dictionary
    """
    s1_points = stroke1['points']
    s2_points = stroke2['points']
    
    if not s1_points or not s2_points:
        return stroke1 if s1_points else stroke2
    
    # Find best connection
    s1_start, s1_end = s1_points[0], s1_points[-1]
    s2_start, s2_end = s2_points[0], s2_points[-1]
    
    # Calculate all possible connection distances
    connections = [
        (s1_end, s2_start, s1_points + s2_points),  # s1 → s2
        (s1_end, s2_end, s1_points + s2_points[::-1]),  # s1 → reverse s2
        (s1_start, s2_start, s1_points[::-1] + s2_points[::-1]),  # reverse s1 → reverse s2
        (s1_start, s2_end, s1_points[::-1] + s2_points)  # reverse s1 → s2
    ]
    
    # Choose connection with minimum gap
    best_connection = min(connections, key=lambda x: math.sqrt((x[0][0] - x[1][0])**2 + (x[0][1] - x[1][1])**2))
    
    # Create consolidated stroke
    consolidated = {
        'stroke_id': stroke1['stroke_id'],  # Keep first ID
        'element_id': stroke1['element_id'],
        'warmth': stroke1['warmth'],
        'warmth_name': stroke1['warmth_name'],
        'phase': stroke1['phase'],
        'points': best_connection[2],
        'length_mm': calculate_stroke_length(best_connection[2]),
        'start_pos': best_connection[2][0],
        'end_pos': best_connection[2][-1],
        'order_index': min(stroke1.get('order_index', 0), stroke2.get('order_index', 0)),
        'consolidated_from': [stroke1['stroke_id'], stroke2['stroke_id']]
    }
    
    return consolidated

def consolidate_strokes_by_phase(strokes: List[Dict], max_gap_mm: float = 2.0, 
                                min_consolidated_length_mm: float = 5.0) -> List[Dict]:
    """
    Consolidate fragmented strokes into longer, more meaningful drawing paths
    
    Args:
        strokes: List of stroke dictionaries
        max_gap_mm: Maximum gap to bridge between strokes
        min_consolidated_length_mm: Minimum length for consolidated strokes
        
    Returns:
        List of consolidated strokes with fewer, longer paths
    """
    if not strokes:
        return []
    
    # Group strokes by phase and element for better consolidation
    phase_groups = defaultdict(list)
    for stroke in strokes:
        key = (stroke['phase'], stroke['element_id'])
        phase_groups[key].append(stroke.copy())
    
    consolidated_strokes = []
    
    for (phase, element_id), group_strokes in phase_groups.items():
        print(f"   🔗 Consolidating {len(group_strokes)} {phase} strokes for element {element_id}")
        
        # Calculate actual lengths first
        for stroke in group_strokes:
            if stroke['length_mm'] == 0.0:  # Fix the 0.0mm bug
                stroke['length_mm'] = calculate_stroke_length(stroke['points'])
        
        # Keep track of which strokes have been consumed
        remaining = list(range(len(group_strokes)))
        
        while remaining:
            # Start with the longest remaining stroke
            best_idx = max(remaining, key=lambda i: group_strokes[i]['length_mm'])
            current_stroke = group_strokes[best_idx]
            remaining.remove(best_idx)
            
            # Try to connect more strokes to this one
            changed = True
            while changed and remaining:
                changed = False
                
                # Find the best stroke to connect to current
                best_connection = None
                best_distance = float('inf')
                
                for idx in remaining:
                    candidate = group_strokes[idx]
                    if are_strokes_connected(current_stroke, candidate, max_gap_mm):
                        # Calculate connection quality (prefer shorter gaps, longer strokes)
                        gap_dist = min(
                            math.sqrt((current_stroke['points'][-1][0] - candidate['points'][0][0])**2 + 
                                    (current_stroke['points'][-1][1] - candidate['points'][0][1])**2),
                            math.sqrt((current_stroke['points'][0][0] - candidate['points'][-1][0])**2 + 
                                    (current_stroke['points'][0][1] - candidate['points'][-1][1])**2)
                        )
                        
                        quality_score = gap_dist - candidate['length_mm'] * 0.1  # Prefer longer strokes
                        
                        if quality_score < best_distance:
                            best_distance = quality_score
                            best_connection = idx
                
                # Connect the best candidate
                if best_connection is not None:
                    candidate = group_strokes[best_connection]
                    current_stroke = connect_strokes(current_stroke, candidate)
                    remaining.remove(best_connection)
                    changed = True
            
            # Only keep strokes that meet minimum length requirement
            if current_stroke['length_mm'] >= min_consolidated_length_mm:
                consolidated_strokes.append(current_stroke)
            else:
                # Keep very short strokes if they're detail strokes (might be important)
                if current_stroke['phase'] == 'detail' or current_stroke['length_mm'] >= 1.0:
                    consolidated_strokes.append(current_stroke)
    
    return consolidated_strokes

def consolidate_all_mask_strokes(mask_stroke_arrays: List[Dict], 
                                max_gap_mm: float = 2.0,
                                min_consolidated_length_mm: float = 5.0) -> List[Dict]:
    """
    Consolidate strokes across all masks for better tactile feedback
    
    Args:
        mask_stroke_arrays: List of mask dictionaries from Step 8
        max_gap_mm: Maximum gap to bridge between strokes
        min_consolidated_length_mm: Minimum length for consolidated strokes
        
    Returns:
        Updated mask stroke arrays with consolidated strokes
    """
    print("🔗 Consolidating fragmented strokes for better tactile feedback...")
    
    original_total = sum(len(mask['strokes']) for mask in mask_stroke_arrays)
    
    for mask_idx, mask_array in enumerate(mask_stroke_arrays):
        original_count = len(mask_array['strokes'])
        
        # Consolidate strokes within this mask
        consolidated = consolidate_strokes_by_phase(
            mask_array['strokes'], 
            max_gap_mm=max_gap_mm,
            min_consolidated_length_mm=min_consolidated_length_mm
        )
        
        # Update mask array
        mask_array['strokes'] = consolidated
        
        # Recalculate statistics
        total_length = sum(stroke['length_mm'] for stroke in consolidated)
        mask_array['total_length_mm'] = total_length
        mask_array['stroke_count'] = len(consolidated)
        
        print(f"   📊 Mask {mask_array['mask_id']}: {original_count} → {len(consolidated)} strokes ({total_length:.1f}mm)")
    
    final_total = sum(len(mask['strokes']) for mask in mask_stroke_arrays)
    total_length = sum(mask['total_length_mm'] for mask in mask_stroke_arrays)
    
    print(f"✅ Consolidation complete: {original_total} → {final_total} strokes ({total_length:.1f}mm total)")
    print(f"   🎯 Reduction: {((original_total - final_total) / original_total * 100):.1f}% fewer strokes")
    
    return mask_stroke_arrays

def analyze_stroke_distribution(mask_stroke_arrays: List[Dict]) -> Dict:
    """
    Analyze the distribution of consolidated strokes
    
    Returns:
        Analysis statistics for tactile feedback suitability
    """
    all_strokes = []
    for mask in mask_stroke_arrays:
        all_strokes.extend(mask['strokes'])
    
    if not all_strokes:
        return {}
    
    lengths = [stroke['length_mm'] for stroke in all_strokes]
    
    # Analyze length distribution for tactile feedback
    analysis = {
        'total_strokes': len(all_strokes),
        'total_length_mm': sum(lengths),
        'avg_length_mm': sum(lengths) / len(lengths),
        'min_length_mm': min(lengths),
        'max_length_mm': max(lengths),
        'length_distribution': {
            'very_short': len([l for l in lengths if l <= 2.0]),  # Likely noise
            'short': len([l for l in lengths if 2.0 < l <= 5.0]),      # Small details
            'medium': len([l for l in lengths if 5.0 < l <= 15.0]),    # Good tactile strokes
            'long': len([l for l in lengths if 15.0 < l <= 50.0]),     # Major features
            'very_long': len([l for l in lengths if l > 50.0])         # Main outlines
        },
        'phase_distribution': {}
    }
    
    # Analyze by phase
    for phase in ['boundary', 'internal', 'detail']:
        phase_strokes = [s for s in all_strokes if s['phase'] == phase]
        if phase_strokes:
            phase_lengths = [s['length_mm'] for s in phase_strokes]
            analysis['phase_distribution'][phase] = {
                'count': len(phase_strokes),
                'avg_length_mm': sum(phase_lengths) / len(phase_lengths),
                'total_length_mm': sum(phase_lengths)
            }
    
    return analysis

