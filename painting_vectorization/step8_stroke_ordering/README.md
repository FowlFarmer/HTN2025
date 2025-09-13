# Step 8: Flattening & Stroke Ordering - Hardware Integration Guide

## Overview

Step 8 is the final stage of the painting vectorization pipeline that produces optimized robot instructions for hardware execution. This module flattens the hierarchical stroke structure into a single ordered array and optimizes the drawing sequence to minimize travel time and pen lifts.

**🎯 Primary Output for Hardware Engineers**: JSON files containing complete robot instructions with timing, coordinates, and metadata ready for robotic painting systems.

## Output Data Structure

### 1. JSON Format (Primary)

The main output is a comprehensive JSON file with the following structure:

```json
{
  "metadata": {
    "canvas_mm": [160.0, 160.0],
    "scale_mm_per_px": 0.156,
    "total_strokes": 700,
    "total_length_mm": 2687.3,
    "total_pen_lifts": 245,
    "estimated_duration_s": 156.2,
    "estimated_travel_distance_mm": 1240.8,
    "processing_timestamp": "2025-09-13T19:15:30.123456",
    "optimization_method": "hybrid",
    "speed_profile": {
      "default_speed_mm_s": 20.0,
      "min_speed_mm_s": 10.0,
      "max_speed_mm_s": 30.0
    }
  },
  "strokes": [
    {
      "stroke_id": 0,
      "element_id": 2,
      "warmth": 0,
      "warmth_name": "warm",
      "phase": "boundary",
      "points": [[45.2, 78.9], [46.1, 79.2], [47.0, 79.8]],
      "length_mm": 12.7,
      "duration_s": 0.85,
      "start_pos": [45.2, 78.9],
      "end_pos": [47.0, 79.8],
      "pen_lifts": 0,
      "curvature_avg": 0.124,
      "speed_mm_s": 15.2,
      "is_closed_loop": false,
      "order_index": 45
    }
  ],
  "pen_lifts": [
    {
      "type": "pen_lift",
      "from_stroke_id": 0,
      "to_stroke_id": 1,
      "travel_distance_mm": 8.5,
      "height_mm": 15.0,
      "duration_s": 0.17
    }
  ],
  "format_version": "1.0",
  "description": "Robotic painting stroke data - optimized for execution"
}
```

### 2. G-code Format (CNC/Robot Compatible)

```gcode
; Robotic Painting G-code
; Generated: 2025-09-13T19:15:30.123456
; Total strokes: 700
; Estimated time: 156.2s

G21 ; Set units to millimeters
G90 ; Absolute positioning
M3 ; Prepare pen/brush system
G0 Z15 ; Lift pen to safe height
G0 X80 Y80 ; Move to canvas center

; Stroke 0: boundary (warm)
; Length: 12.7mm, Speed: 15.2mm/s
G0 X45.20 Y78.90 ; Move to start
G0 Z0 ; Lower pen
G1 X46.10 Y79.20 F912
G1 X47.00 Y79.80 F912
G0 Z15 ; Lift pen
```

### 3. CSV Format (Analysis & Debugging)

```csv
stroke_id,element_id,phase,warmth,warmth_name,length_mm,duration_s,speed_mm_s,curvature_avg,start_x,start_y,end_x,end_y,is_closed_loop,point_count,order_index
0,2,boundary,0,warm,12.7,0.85,15.2,0.124,45.2,78.9,47.0,79.8,false,3,45
1,2,internal,0,warm,8.3,0.42,19.8,0.089,47.0,79.8,50.1,82.4,false,2,46
```

## Key Data Fields for Hardware Integration

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

### Python Integration
```python
from step8_stroke_ordering.stroke_ordering import process_all_stroke_ordering

# Process stroke ordering
results = process_all_stroke_ordering(color_detection_results, strategy='hybrid')
stroke_data = results['stroke_data']

# Access robot instructions
for stroke in stroke_data.strokes:
    print(f"Draw stroke {stroke['stroke_id']} at speed {stroke['speed_mm_s']}mm/s")
    for point in stroke['points']:
        x, y = point
        robot.move_to(x, y)

# Handle pen lifts
for lift in stroke_data.pen_lifts:
    robot.lift_pen(lift['height_mm'])
    robot.travel_to_next_stroke()
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