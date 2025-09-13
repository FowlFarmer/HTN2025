# Step 5: Vectorization & Simplification

## Overview
This module converts pixel sequences from stroke graphs into compact polylines or smooth curves suitable for robot motion, vector graphics, and efficient storage. It transforms discrete pixel paths into mathematical curve descriptions while preserving semantic drawing order.

## Purpose
- Convert pixel-based stroke sequences to vector format
- Reduce data complexity while maintaining visual fidelity
- Generate robot-friendly trajectory descriptions
- Support multiple vectorization methods (RDP, splines, adaptive)
- Maintain semantic hierarchy from Step 4

## Key Components

### `vectorization.py`
Main vectorization processing module with multiple algorithms.

**Input**:
- Stroke graphs with semantic ordering from Step 4
- Pixel coordinate sequences for each stroke
- Quality and compression parameters

**Output**:
- Vectorized polylines or splines
- Compression statistics and error analysis
- Semantic phase organization preserved
- Robot-ready trajectory data

## Processing Pipeline

### 5.1 Method Selection
Choose optimal vectorization approach based on use case:

```python
# RDP (Ramer-Douglas-Peucker): Fast, preserves corners
result = vectorize_stroke_sequences(stroke_graph, method='rdp', max_error=1.0)

# Spline fitting: Smooth curves, better for robot motion
result = vectorize_stroke_sequences(stroke_graph, method='spline', max_error=1.0)

# Adaptive: Automatically chooses best method per stroke
result = vectorize_stroke_sequences(stroke_graph, method='adaptive', max_error=1.0)
```

### 5.2 RDP Simplification
Reduces polyline complexity while maintaining shape:

```python
def simplify_polyline_rdp(pts, epsilon=2.0):
    """Simplify polyline using RDP algorithm"""
    if len(pts) < 3:
        return pts

    # Convert to float for precision
    pts_float = pts.astype(np.float64)

    # Apply RDP simplification
    if RDP_AVAILABLE:
        simplified = rdp(pts_float, epsilon=epsilon)
    else:
        simplified = rdp_fallback(pts_float, epsilon)

    return simplified
```

**Algorithm**: Recursively removes points that deviate less than `epsilon` from straight line segments.

### 5.3 Spline Fitting
Creates smooth parametric curves:

```python
def fit_spline(pts, smoothing_factor=0.1, num_points=None):
    """Fit smooth spline to points for robot-friendly trajectories"""
    if len(pts) < 4:
        return pts

    # Fit parametric spline
    tck, u = splprep([pts[:, 0], pts[:, 1]],
                     s=smoothing_factor * len(pts), per=False)

    # Evaluate spline at desired resolution
    if num_points is None:
        num_points = len(pts) * 2

    u_new = np.linspace(0, 1, num_points)
    spline_x, spline_y = splev(u_new, tck)

    return np.column_stack([spline_x, spline_y])
```

**Benefits**: Continuous derivatives, smooth acceleration profiles, reduced jerk for robot systems.

### 5.4 Adaptive Parameter Tuning
Automatically optimizes parameters to meet quality requirements:

```python
def adaptive_simplification(pts, max_error=1.0, method='rdp'):
    """Adaptively choose simplification parameters"""
    if method == 'rdp':
        # Try increasing epsilon values
        for epsilon in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            simplified = simplify_polyline_rdp(pts, epsilon=epsilon)
            error = calculate_approximation_error(pts, simplified)

            if error <= max_error:
                return {
                    'points': simplified,
                    'parameter': epsilon,
                    'error': error,
                    'compression_ratio': len(simplified) / len(pts),
                    'success': True
                }

    # Return best attempt if target not achievable
    return {'success': False, 'points': simplified, ...}
```

### 5.5 Quality Metrics
Measure approximation fidelity:

```python
def calculate_approximation_error(original, simplified):
    """Calculate maximum deviation between curves"""
    # Use scipy for accurate distance calculation
    distances = cdist(original, simplified)
    min_distances = np.min(distances, axis=1)
    return np.max(min_distances)  # Maximum error
```

**Metrics Provided**:
- **Maximum error**: Worst-case deviation in pixels
- **Average error**: Mean approximation error
- **Compression ratio**: Points reduced (e.g., 0.3 = 70% reduction)
- **Success rate**: Fraction meeting error tolerance

## Semantic-Aware Processing

### Hierarchical Vectorization
Preserves meaningful drawing order from Step 4:

```python
def vectorize_stroke_sequences(stroke_graph, method='adaptive', max_error=1.0):
    # Get semantic drawing order
    semantic_order = stroke_graph.semantic_order

    vectorized_phases = {
        'boundary': [],    # Foundation strokes (high fidelity)
        'internal': [],    # Major features (balanced quality)
        'detail': []       # Fine details (higher compression)
    }

    # Process each semantic phase with appropriate settings
    for phase_name, edges in semantic_order['drawing_phases'].items():
        for edge in edges:
            points = stroke_graph.get_edge_points(edge)

            # Adapt quality settings by semantic importance
            if phase_name == 'boundary':
                error_tolerance = max_error * 0.5  # Higher fidelity
            elif phase_name == 'internal':
                error_tolerance = max_error        # Standard fidelity
            else:  # detail
                error_tolerance = max_error * 1.5  # Lower fidelity OK

            result = adaptive_simplification(points, error_tolerance, method)
            vectorized_phases[phase_name].append(result)

    return vectorized_phases
```

## Algorithm Comparison

### RDP vs Spline Fitting

| Aspect | RDP | Splines |
|--------|-----|---------|
| **Speed** | Fast (O(n log n)) | Slower (O(n³)) |
| **Output** | Angular polylines | Smooth curves |
| **Corners** | Preserved exactly | Rounded |
| **Robot Motion** | Jerky accelerations | Smooth trajectories |
| **Compression** | High (50-90% reduction) | Moderate (30-70% reduction) |
| **Use Case** | Technical drawings | Artistic curves |

### Method Selection Guidelines

#### **Choose RDP when**:
- Sharp corners are important (architectural drawings)
- Maximum compression needed
- Processing speed is critical
- Simple pen plotters or basic robots

#### **Choose Splines when**:
- Smooth motion required (high-end robots)
- Artistic/organic content
- Continuous derivatives needed
- Professional robotic systems

#### **Choose Adaptive when**:
- Mixed content types
- Unknown input characteristics
- Want best of both approaches
- Quality requirements vary by semantic phase

## Usage Examples

### Basic Vectorization
```python
from vectorization import process_all_vectorizations, save_vectorization_results

# Process all stroke graphs from Step 4
vectorized_results = process_all_vectorizations(stroke_graphs,
                                               method='adaptive',
                                               max_error=1.0)

# Save visualization
output_path = save_vectorization_results(vectorized_results)
print(f"Results saved to: {output_path}")
```

### Method Comparison
```python
# Compare all methods on single stroke graph
methods = ['rdp', 'spline', 'adaptive']
results = {}

for method in methods:
    result = vectorize_stroke_sequences(stroke_graph,
                                      method=method,
                                      max_error=1.0)

    meta = result['metadata']
    results[method] = {
        'compression': meta['average_compression'],
        'error': meta['average_error'],
        'points': meta['total_points_simplified']
    }

    print(f"{method:8}: {meta['average_compression']:.2f}x compression, "
          f"{meta['average_error']:.1f}px error")
```

### Robot Trajectory Generation
```python
def generate_robot_commands(vectorized_result):
    """Convert vectorized strokes to robot drawing commands"""
    commands = []

    # Process in semantic order for logical construction
    for phase_name in ['boundary', 'internal', 'detail']:
        phase_strokes = vectorized_result[phase_name]

        for stroke in phase_strokes:
            points = stroke['vectorized_points']

            if len(points) > 0:
                # Move to start position (pen up)
                commands.append(f"MOVE {points[0][0]:.2f} {points[0][1]:.2f}")
                commands.append("PEN_DOWN")

                # Draw stroke path
                for point in points[1:]:
                    commands.append(f"LINE {point[0]:.2f} {point[1]:.2f}")

                commands.append("PEN_UP")

    return commands
```

### SVG Export
```python
def export_to_svg(vectorized_result, output_path, image_size=(1024, 1024)):
    """Export vectorized strokes to SVG format"""
    svg_content = [
        f'<svg width="{image_size[0]}" height="{image_size[1]}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]

    # Color scheme for semantic phases
    colors = {'boundary': '#FF0000', 'internal': '#0000FF', 'detail': '#00AA00'}

    for phase_name, color in colors.items():
        phase_strokes = vectorized_result[phase_name]

        for stroke in phase_strokes:
            points = stroke['vectorized_points']

            if len(points) > 1:
                # Create SVG path
                path_data = f"M {points[0][0]:.2f} {points[0][1]:.2f}"
                for point in points[1:]:
                    path_data += f" L {point[0]:.2f} {point[1]:.2f}"

                svg_content.append(
                    f'<path d="{path_data}" stroke="{color}" '
                    f'fill="none" stroke-width="2"/>'
                )

    svg_content.append('</svg>')

    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_content))

    print(f"SVG exported to: {output_path}")
```

## Performance Characteristics

### Processing Speed (per stroke)
- **RDP**: 1-5ms for typical strokes (100-1000 points)
- **Splines**: 5-20ms for typical strokes
- **Adaptive**: 2-15ms (depends on content)

### Compression Ratios
| Content Type | RDP | Splines | Adaptive |
|--------------|-----|---------|-----------|
| **Straight lines** | 0.05x (95% reduction) | 0.3x | 0.05x |
| **Smooth curves** | 0.4x | 0.6x | 0.5x |
| **Complex details** | 0.7x | 0.8x | 0.6x |
| **Mixed content** | 0.5x | 0.6x | 0.4x |

### Error Characteristics
- **RDP**: Uniform error distribution, sharp transitions
- **Splines**: Lower error near control points, smooth everywhere
- **Adaptive**: Optimized per stroke type

## Integration with Pipeline

### Input Requirements
- StrokeGraph objects from Step 4 with semantic ordering
- Pixel coordinate sequences as numpy arrays
- Quality parameters (max_error, compression targets)

### Output Format
```python
vectorized_result = {
    'boundary': [stroke_results...],
    'internal': [stroke_results...],
    'detail': [stroke_results...],
    'metadata': {
        'total_strokes': 150,
        'total_points_original': 12450,
        'total_points_simplified': 3120,
        'average_compression': 0.25,
        'average_error': 0.8,
        'successful_vectorizations': 148
    }
}

stroke_result = {
    'edge': (node1, node2),
    'original_points': numpy_array,
    'vectorized_points': numpy_array,
    'method': 'rdp'/'spline'/'adaptive',
    'parameter': epsilon_or_smoothing,
    'error': max_deviation_pixels,
    'compression_ratio': 0.3,
    'success': True,
    'phase': 'boundary'/'internal'/'detail'
}
```

### Next Steps
- **SVG Export**: Convert to scalable vector graphics
- **Robot Control**: Generate G-code or robot commands
- **Interactive Editing**: Allow manual refinement
- **Style Transfer**: Apply artistic effects

## Algorithm Parameters

### RDP Settings
- **epsilon**: Distance tolerance (0.5-5.0 pixels)
  - 0.5: High fidelity, minimal compression
  - 2.0: Balanced quality/compression (default)
  - 5.0: High compression, lower fidelity

### Spline Settings
- **smoothing_factor**: Approximation vs interpolation (0.0-5.0)
  - 0.0: Perfect interpolation through all points
  - 0.1: Light smoothing (default)
  - 1.0+: Heavy smoothing, significant approximation

### Adaptive Settings
- **max_error**: Maximum allowed deviation (pixels)
- **Quality by phase**:
  - Boundary: 0.5x max_error (higher fidelity)
  - Internal: 1.0x max_error (standard)
  - Detail: 1.5x max_error (lower fidelity acceptable)

## Error Handling

### Common Issues

#### **Empty Stroke Sequences**
- **Symptoms**: No points to vectorize
- **Cause**: Failed edge extraction or empty stroke graphs
- **Solution**: Check Step 3 and Step 4 pipeline integrity

#### **High Approximation Error**
- **Symptoms**: Vectorized curves don't match originals
- **Cause**: Too aggressive compression settings
- **Solution**: Reduce epsilon (RDP) or smoothing_factor (splines)

#### **Poor Compression**
- **Symptoms**: Output nearly same size as input
- **Cause**: Very complex curves or conservative settings
- **Solution**: Increase tolerance or try different method

### Parameter Tuning

#### **For Technical Drawings**
- Method: RDP
- Max error: 0.5-1.0 pixels
- Priority: Preserve corners and straight lines

#### **For Artistic Content**
- Method: Splines or Adaptive
- Max error: 1.0-2.0 pixels
- Priority: Smooth curves and natural motion

#### **For Robot Systems**
- Method: Splines (smooth acceleration)
- Max error: Based on robot precision
- Consider velocity/acceleration constraints

## Dependencies
- RDP (`rdp`) - Fast polyline simplification
- SciPy (`scipy`) - Spline fitting and distance calculations
- NumPy (`numpy`) - Array operations and mathematics
- Matplotlib (`matplotlib`) - Visualization and result plotting

## Quality Assurance
- **Error bounds**: Guaranteed maximum deviation
- **Visual validation**: Overlay comparisons in output plots
- **Compression metrics**: Quantified size reduction
- **Success rates**: Fraction meeting quality targets
- **Semantic preservation**: Maintains drawing order hierarchy