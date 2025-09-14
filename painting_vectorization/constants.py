#!/usr/bin/env python3
"""
Painting Vectorization - Global Constants

This file contains all global constants used throughout the painting vectorization pipeline.
Change values here to update the entire system.

Canvas Configuration:
- Change CANVAS_SIZE_MM to scale the entire system
- All modules will automatically use the new dimensions
- Maintains aspect ratio and coordinate systems
"""

# =============================================================================
# CANVAS DIMENSIONS
# =============================================================================

# Physical canvas dimensions in millimeters
# Change this value to scale the entire system (e.g., 150 for 150x150mm)
CANVAS_SIZE_MM = 160  # 16cm x 16cm canvas

# Derived canvas properties (automatically calculated)
CANVAS_WIDTH_MM = CANVAS_SIZE_MM
CANVAS_HEIGHT_MM = CANVAS_SIZE_MM

# Canvas bounds for validation
CANVAS_MIN_X = 0.0
CANVAS_MAX_X = CANVAS_WIDTH_MM
CANVAS_MIN_Y = 0.0
CANVAS_MAX_Y = CANVAS_HEIGHT_MM

# =============================================================================
# COORDINATE SYSTEM
# =============================================================================

# Coordinate system configuration
COORDINATE_ORIGIN = "top-left"  # Origin (0,0) position
COORDINATE_SYSTEM = {
    "origin": "top-left",
    "x_direction": "left-to-right",
    "y_direction": "top-to-bottom",
    "units": "millimeters"
}

# =============================================================================
# SAMPLING & SPACING
# =============================================================================

# Default point spacing for robot motion
DEFAULT_SPACING_MM = 1.0  # 1mm spacing between sampled points

# Minimum segment length to preserve
MIN_SEGMENT_LENGTH_MM = 0.5

# =============================================================================
# ROBOT CONFIGURATION
# =============================================================================

# Robot motion parameters (handled by firmware engineer)
ROBOT_CONFIG = {
    "pen_lift_height_mm": 15.0,
    "min_travel_distance_mm": 2.0
}

# =============================================================================
# DEMO TIMING
# =============================================================================

# Demo constraints (timing handled by firmware engineer)
# These are kept for reference but not used in calculations
DEMO_DURATION_LIMIT_S = 120.0  # 2 minutes target
STROKE_COUNT_ESTIMATE = "handled_by_firmware"  # Timing calculations removed

# =============================================================================
# FILE PATHS
# =============================================================================

# Default output directories
OUTPUT_DIRS = {
    "step2_segmentation": "results/step2_segmentation",
    "step3_edge_extraction": "results/step3_edge_extraction", 
    "step4_stroke_graphs": "results/step4_stroke_graphs",
    "step5_vectorization": "results/step5_vectorization",
    "step6_sampling": "results/step6_sampling",
    "step7_color_detection": "results/step7_color_detection",
    "step8_stroke_ordering": "results/step8_stroke_ordering"
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_canvas_bounds(x, y):
    """
    Validate that coordinates are within canvas bounds
    
    Args:
        x: X coordinate in mm
        y: Y coordinate in mm
        
    Returns:
        bool: True if within bounds, False otherwise
    """
    return (CANVAS_MIN_X <= x <= CANVAS_MAX_X and 
            CANVAS_MIN_Y <= y <= CANVAS_MAX_Y)

def get_canvas_info():
    """
    Get comprehensive canvas information
    
    Returns:
        dict: Canvas configuration details
    """
    return {
        "size_mm": CANVAS_SIZE_MM,
        "width_mm": CANVAS_WIDTH_MM,
        "height_mm": CANVAS_HEIGHT_MM,
        "bounds": {
            "min_x": CANVAS_MIN_X,
            "max_x": CANVAS_MAX_X,
            "min_y": CANVAS_MIN_Y,
            "max_y": CANVAS_MAX_Y
        },
        "coordinate_system": COORDINATE_SYSTEM,
        "area_mm2": CANVAS_WIDTH_MM * CANVAS_HEIGHT_MM,
        "diagonal_mm": (CANVAS_WIDTH_MM**2 + CANVAS_HEIGHT_MM**2)**0.5
    }

def scale_to_new_canvas_size(new_size_mm):
    """
    Calculate scaling factor for a new canvas size
    
    Args:
        new_size_mm: New canvas size in mm
        
    Returns:
        float: Scaling factor to apply to existing coordinates
    """
    return new_size_mm / CANVAS_SIZE_MM

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("🎨 Painting Vectorization Constants")
    print("=" * 40)
    
    canvas_info = get_canvas_info()
    print(f"Canvas Size: {canvas_info['size_mm']}mm x {canvas_info['size_mm']}mm")
    print(f"Canvas Area: {canvas_info['area_mm2']:.0f} mm²")
    print(f"Coordinate System: {canvas_info['coordinate_system']['origin']}")
    print(f"Default Spacing: {DEFAULT_SPACING_MM}mm")
    print(f"Demo Duration: {DEMO_DURATION_LIMIT_S}s")
    
    print("\n📐 Scaling Example:")
    new_size = 150
    scale_factor = scale_to_new_canvas_size(new_size)
    print(f"To change to {new_size}x{new_size}mm:")
    print(f"  1. Set CANVAS_SIZE_MM = {new_size}")
    print(f"  2. Existing coordinates scale by {scale_factor:.3f}")
    
    print("\n✅ To update canvas size:")
    print("  1. Edit CANVAS_SIZE_MM in this file")
    print("  2. All modules automatically use new dimensions")
    print("  3. No other code changes needed")
