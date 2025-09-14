# Step 8: Flattening & Stroke Ordering - Hardware Integration Guide

## Overview

Step 8 is the final stage of the painting vectorization pipeline that produces optimized robot instructions for hardware execution. This module flattens the hierarchical stroke structure into a single ordered array and optimizes the drawing sequence to minimize travel time and pen lifts.

**🎯 Primary Output for Hardware Engineers**: JSON files containing complete robot instructions with timing, coordinates, and metadata ready for robotic painting systems.

## Output Data Structure

### JSON Format (Primary)

The main output is a simplified JSON file optimized for narration timing:

```json
{
  "mask_stroke_arrays": [
    {
      "mask_id": 0,
      "warmth_class": 0,
      "warmth_name": "warm",
      "total_length_mm": 45.2,
      "total_duration_s": 2.1,
      "stroke_count": 3,
      "strokes": [
        {
          "stroke_id": 0,
          "element_id": 0,
          "warmth": 0,
          "warmth_name": "warm",
          "phase": "boundary",
          "points": [[45.2, 78.9], [46.1, 79.2], [47.0, 79.8]],
          "length_mm": 12.7,
          "duration_s": 0.85,
          "start_pos": [45.2, 78.9],
          "end_pos": [47.0, 79.8],
          "speed_mm_s": 15.2
        }
      ]
    }
  ],
  "statistics": {
    "total_strokes": 700,
    "total_masks": 12,
    "total_length_mm": 2687.3,
    "estimated_duration_s": 156.2,
    "optimization_method": "hybrid"
  },
  "format_version": "2.0",
  "description": "Robotic painting stroke data grouped by mask for narration timing"
}
```

## Key Data Fields for Hardware Integration

### Primary Output: Mask-Grouped Stroke Arrays
Step 8 now returns strokes grouped by mask for narration timing:

```python
{
    'mask_stroke_arrays': [
        {
            'mask_id': 0,
            'warmth_class': 0,
            'warmth_name': 'warm',
            'strokes': [stroke1, stroke2, stroke3],  # Array of strokes for mask 0
            'total_length_mm': 45.2,
            'total_duration_s': 2.1,
            'stroke_count': 3
        },
        {
            'mask_id': 1,
            'warmth_class': 1,
            'warmth_name': 'cool',
            'strokes': [stroke4, stroke5],  # Array of strokes for mask 1
            'total_length_mm': 32.8,
            'total_duration_s': 1.6,
            'stroke_count': 2
        }
    ],
    'statistics': {...}
}
```

### Stroke Data
- **`stroke_id`**: Unique identifier for execution order tracking
- **`points`**: Array of [x, y] coordinates in millimeters (absolute positioning)
- **`start_pos`** / **`end_pos`**: Critical for path planning and pen lift decisions
- **`speed_mm_s`**: Recommended drawing speed based on curvature analysis
- **`duration_s`**: Expected execution time for this stroke
- **`is_closed_loop`**: Boolean indicating if stroke forms a closed shape

### Movement Planning
- **`pen_lifts`**: Array of required pen lift operations between strokes
- **`travel_distance_mm`**: Distance to travel with pen raised
- **`height_mm`**: Safety clearance height (default: 15mm)

### Canvas & Scaling
- **`canvas_mm`**: Physical canvas dimensions [width, height] in millimeters
- **`scale_mm_per_px`**: Conversion factor from pixel coordinates to millimeters

## Optimization Strategies

The system provides three optimization strategies for different use cases:

### 1. Hybrid Strategy (Recommended)
```python
strategy='hybrid'
```
- **Use Case**: General robotic painting with artistic preservation
- **Behavior**: Maintains semantic order (boundary→internal→detail) while optimizing travel within each phase
- **Benefits**: Preserves artistic meaning, moderate travel optimization
- **Typical Travel Reduction**: 30-50%

### 2. Travel Minimization Strategy
```python
strategy='travel_minimization'
```
- **Use Case**: Maximum efficiency, speed-critical applications
- **Behavior**: Pure TSP-based optimization ignoring artistic semantics
- **Benefits**: Minimum total travel distance and pen lifts
- **Typical Travel Reduction**: 60-80%

### 3. Semantic Preservation Strategy
```python
strategy='semantic_preserve'
```
- **Use Case**: Artistic fidelity is critical, educational demonstrations
- **Behavior**: Strict semantic ordering with minimal travel optimization
- **Benefits**: Maximum artistic accuracy, predictable drawing order
- **Typical Travel Reduction**: 10-20%

## Hardware Integration Specifications

### Coordinate System
- **Origin**: Bottom-left corner (0, 0)
- **Units**: Millimeters (mm)
- **Coordinate Range**: 0-160mm x 0-160mm (configurable)
- **Precision**: 0.01mm (2 decimal places in output)

### Motion Profile Recommendations

#### Drawing Speeds
- **Straight Lines**: 20-30 mm/s
- **Curved Sections**: 10-20 mm/s (adjusted by curvature factor)
- **Closed Loops**: 15-25 mm/s (with deceleration at closure)

#### Travel Speeds
- **Pen-up Movement**: 50 mm/s (default assumption)
- **Safety Height**: 15mm minimum clearance
- **Minimum Travel Distance**: 2mm (below this, no pen lift)

#### Acceleration Profiles
- **Drawing Acceleration**: 100-200 mm/s²
- **Travel Acceleration**: 500-1000 mm/s²

### Timing Estimates
The system provides comprehensive timing estimates:

```python
{
  "estimated_duration_s": 156.2,    # Total drawing time
  "drawing_time_s": 134.5,          # Pen-down time only
  "travel_time_s": 21.7,            # Pen-up movement time
  "pen_lift_operations": 245        # Number of pen lifts
}
```

## Quality Assurance & Error Handling

### Data Validation
- **Point Validation**: All coordinates within canvas bounds
- **Continuity Check**: End point of stroke N matches start of stroke N+1 (accounting for pen lifts)
- **Speed Limits**: All speeds within specified min/max bounds
- **Loop Closure**: Closed loop start/end points match within 0.1mm tolerance

### Error Recovery
```python
{
  "validation_errors": [
    {
      "type": "coordinate_bounds",
      "stroke_id": 45,
      "message": "Point [165.2, 78.9] exceeds canvas bounds",
      "severity": "error"
    }
  ],
  "warnings": [
    {
      "type": "high_curvature",
      "stroke_id": 23,
      "message": "Curvature 0.85 may require slower speed",
      "severity": "warning"
    }
  ]
}
```

## Performance Metrics

### Typical Pipeline Output
- **Processing Time**: 0.1-0.5 seconds for Step 8 optimization
- **Stroke Count**: 200-1000 strokes per image
- **Drawing Length**: 1000-5000mm total
- **Execution Time**: 2-10 minutes estimated
- **Travel Distance**: 500-2000mm (depends on optimization)

### Optimization Effectiveness
- **Pen Lifts Reduced**: 20-60% compared to naive ordering
- **Travel Distance Reduced**: 30-80% depending on strategy
- **Execution Time Saved**: 15-45% overall time reduction

## Integration Examples

### Python Integration with Narration Timing
```python
from step8_stroke_ordering.stroke_ordering import process_all_stroke_ordering

# Process stroke ordering (returns mask-grouped arrays)
results = process_all_stroke_ordering(color_detection_results, strategy='hybrid')
mask_stroke_arrays = results['mask_stroke_arrays']

# Execute drawing with narration timing
for i, mask_array in enumerate(mask_stroke_arrays):
    print(f"Drawing mask {mask_array['mask_id']} ({mask_array['warmth_name']})")
    
    # Start narration for this mask
    narration.start_mask_narration(
        mask_id=mask_array['mask_id'],
        duration_s=mask_array['total_duration_s'],
        warmth=mask_array['warmth_name']
    )
    
    # Draw all strokes for this mask
    for stroke in mask_array['strokes']:
        print(f"  Drawing stroke {stroke['stroke_id']} at {stroke['speed_mm_s']}mm/s")
        robot.draw_stroke(stroke['points'], stroke['speed_mm_s'])
    
    # Finish narration for this mask
    narration.finish_mask_narration(mask_array['mask_id'])

print(f"Completed drawing {len(mask_stroke_arrays)} masks")
```

### Direct Stroke Access
```python
# Access individual strokes across all masks
all_strokes = []
for mask_array in mask_stroke_arrays:
    all_strokes.extend(mask_array['strokes'])

# Process strokes sequentially if needed
for stroke in all_strokes:
    robot.draw_stroke(stroke['points'], stroke['speed_mm_s'])
```

### Robot Control Workflow
```python
# 1. Load JSON instructions
with open('stroke_ordering_results_20250913_191530.json') as f:
    data = json.load(f)

# 2. Initialize robot at canvas center
robot.home()
robot.move_to(80, 80)  # Canvas center

# 3. Execute strokes in order
for stroke in data['strokes']:
    # Move to start position
    robot.move_to(*stroke['start_pos'])
    robot.lower_pen()

    # Draw stroke at recommended speed
    robot.set_speed(stroke['speed_mm_s'])
    for point in stroke['points']:
        robot.draw_to(*point)

    robot.lift_pen()

# 4. Return to origin
robot.move_to(0, 0)
```

## File Outputs

Each processing run generates multiple output files:

1. **`stroke_ordering_results_YYYYMMDD_HHMMSS.json`** - Primary robot instructions
2. **`stroke_ordering_gcode_YYYYMMDD_HHMMSS.gcode`** - G-code for CNC systems
3. **`stroke_ordering_data_YYYYMMDD_HHMMSS.csv`** - Data analysis format
4. **`stroke_ordering_visualization_YYYYMMDD_HHMMSS.png`** - Visual verification

## Troubleshooting

### Common Issues

**Issue**: Coordinates outside canvas bounds
**Solution**: Check `canvas_mm` parameter matches physical setup

**Issue**: Excessive pen lifts
**Solution**: Adjust `min_travel_distance_mm` parameter (default: 2mm)

**Issue**: Drawing too slow/fast
**Solution**: Modify speed profile in `StrokeData.meta['speed_profile']`

**Issue**: Path not optimized
**Solution**: Try different optimization strategy (`hybrid`, `travel_minimization`)

### Validation Commands
```bash
# Test complete pipeline
python process_image.py examples/test_image.jpg

# Check output files
ls -la results/step8_stroke_ordering/

# Validate JSON structure
python -m json.tool stroke_ordering_results_*.json
```

## Hardware Requirements

### Minimum System Specifications
- **Positioning Accuracy**: ±0.1mm
- **Speed Range**: 5-50 mm/s variable
- **Working Area**: 160mm x 160mm minimum
- **Z-axis Travel**: 20mm minimum
- **Control System**: JSON parsing capability OR G-code interpreter

### Recommended Hardware
- **Positioning Accuracy**: ±0.05mm
- **Speed Range**: 1-100 mm/s variable
- **Acceleration**: 1000 mm/s² capability
- **Working Area**: 200mm x 200mm (allows margins)
- **Z-axis Travel**: 50mm (improved safety)

The Step 8 output provides everything needed for professional robotic painting execution with optimized efficiency and artistic preservation.