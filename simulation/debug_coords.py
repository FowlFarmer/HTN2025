#!/usr/bin/env python3
"""
Debug coordinate transformation to understand the disconnect
"""

import sys
from pathlib import Path
import json

# Add project root to path for constants
sys.path.append(str(Path(__file__).parent.parent))
from painting_vectorization.constants import CANVAS_WIDTH_MM, CANVAS_HEIGHT_MM

def test_coordinate_conversion():
    """Test the coordinate conversion logic"""
    
    # Canvas setup (matching simulator)
    window_size = (1000, 800)
    canvas_padding = 50
    canvas_pixel_width = min(window_size[0] - 2 * canvas_padding, window_size[1] - 200)
    canvas_pixel_height = canvas_pixel_width  # Square canvas
    
    # Canvas position on screen
    canvas_x = (window_size[0] - canvas_pixel_width) // 2
    canvas_y = canvas_padding
    
    # Scale factor: mm to pixels
    mm_to_pixel = canvas_pixel_width / CANVAS_WIDTH_MM
    
    print(f"📐 Canvas Setup:")
    print(f"   Window: {window_size}")
    print(f"   Canvas pixels: {canvas_pixel_width} x {canvas_pixel_height}")
    print(f"   Canvas position: ({canvas_x}, {canvas_y})")
    print(f"   MM to pixel ratio: {mm_to_pixel:.3f}")
    print(f"   Canvas MM: {CANVAS_WIDTH_MM} x {CANVAS_HEIGHT_MM}")
    print()
    
    def mm_to_screen(pos_mm):
        """Convert millimeter coordinates to screen pixels"""
        screen_x = int(canvas_x + pos_mm[0] * mm_to_pixel)
        screen_y = int(canvas_y + pos_mm[1] * mm_to_pixel)
        return (screen_x, screen_y)
    
    # Test some coordinate conversions
    test_points = [
        [0, 0],           # Top-left corner
        [80, 80],         # Center
        [160, 160],       # Bottom-right corner
        [26.2, 157.3],    # First stroke start
        [29.9, 157.7],    # First stroke end
    ]
    
    print("🎯 Coordinate Conversions:")
    for point_mm in test_points:
        screen_pos = mm_to_screen(point_mm)
        print(f"   MM ({point_mm[0]:6.1f}, {point_mm[1]:6.1f}) -> Screen {screen_pos}")
    
    # Load actual stroke data and test first stroke
    stroke_file = "../painting_vectorization/results/step8_stroke_ordering/stroke_ordering_results_20250913_215941.json"
    
    if Path(stroke_file).exists():
        with open(stroke_file, 'r') as f:
            data = json.load(f)
        
        first_stroke = data['mask_stroke_arrays'][0]['strokes'][0]
        points = first_stroke['points']
        
        print(f"\n🔵 First Stroke Analysis:")
        print(f"   Stroke ID: {first_stroke['stroke_id']}")
        print(f"   Points: {len(points)}")
        print(f"   Warmth: {first_stroke['warmth_name']}")
        
        for i, point in enumerate(points):
            screen_pos = mm_to_screen(point)
            print(f"   Point {i}: MM ({point[0]:6.1f}, {point[1]:6.1f}) -> Screen {screen_pos}")

if __name__ == "__main__":
    test_coordinate_conversion()

