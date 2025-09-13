# Step 7: Color Detection - Warm vs Cool Classification

## Overview

Step 7 performs perceptual color analysis to classify painting strokes as warm or cool colors. This binary classification (0 = warm, 1 = cool) enables robots to understand the emotional temperature of different regions in a painting for intelligent drawing decisions.

## Technical Implementation

### Color Space Analysis

**Primary Method: CIELAB Color Space**
- Uses perceptually uniform CIELAB color space for accurate color temperature analysis
- Analyzes `a*` and `b*` channels to determine color warmth
- `a*` channel: green (-) to red (+)
- `b*` channel: blue (-) to yellow (+)

**Fallback Method: HSV Color Space**
- Used when scikit-image is unavailable
- Analyzes hue values to classify warm vs cool regions
- Warm hues: 0-60° (reds, oranges, yellows) and 300-360° (magentas)
- Cool hues: 120-240° (greens, cyans, blues)

### Classification Algorithm

```python
# CIELAB method
warm_votes = ((a_values > a_thresh) | (b_values > b_thresh))
warm_ratio = np.sum(warm_votes) / len(a_values)
warmth_class = 0 if warm_ratio >= warm_ratio_thresh else 1

# HSV method
warm_hues = ((hues <= 60) | (hues >= 300))
warm_ratio = np.sum(warm_hues) / len(hues)
warmth_class = 0 if warm_ratio >= warm_ratio_thresh else 1
```

### Parameters

- `a_thresh`: 5.0 - Threshold for CIELAB a* channel warmth detection
- `b_thresh`: 5.0 - Threshold for CIELAB b* channel warmth detection
- `warm_ratio_thresh`: 0.6 - Minimum ratio of warm pixels to classify as warm
- `chroma_thresh`: 6.0 - Minimum chroma (color saturation) for reliable classification

### Quality Assurance

- **Chroma filtering**: Ignores achromatic (gray) regions with low color saturation
- **Pixel count validation**: Requires minimum 10 pixels per region for reliable classification
- **Confidence scoring**: Based on color distribution uniformity and saturation levels
- **Edge case handling**: Graceful degradation for problematic regions

## Integration

Step 7 integrates seamlessly into the pipeline between Step 6 (Sampling) and the final output:

1. **Input**: Processed image, segmentation masks, and sampled stroke sequences
2. **Processing**: Color analysis of each mask region using stroke coordinate sampling
3. **Output**: Binary classification (0=warm, 1=cool) associated with each stroke sequence
4. **Visualization**: Color-coded visualization showing warm (red) and cool (blue) classifications

## Output Format

```python
{
    'results': [
        {
            'mask_id': 0,
            'warmth_class': 0,  # 0 = warm, 1 = cool
            'confidence': 0.856,
            'method': 'CIELAB',
            'statistics': {
                'warm_ratio': 0.743,
                'avg_chroma': 12.4,
                'pixel_count': 1247
            }
        }
    ],
    'overall_statistics': {
        'successful_classifications': 12,
        'warm_strokes': 8,
        'cool_strokes': 4,
        'warm_percentage': 66.7,
        'cool_percentage': 33.3,
        'average_confidence': 0.821,
        'method_used': 'CIELAB'
    }
}
```

## Usage Example

```python
from step7_color_detection.color_detection import process_all_color_detection

# Process color detection
color_results = process_all_color_detection(
    processed_img=image,
    masks=segmentation_masks,
    sampling_results=stroke_sequences
)

# Access classifications
for result in color_results['results']:
    warmth = "warm" if result['warmth_class'] == 0 else "cool"
    confidence = result['confidence']
    print(f"Stroke {result['mask_id']}: {warmth} ({confidence:.3f} confidence)")
```

## Performance

- **Processing Speed**: ~0.05-0.2s per mask region
- **Memory Usage**: Minimal - processes regions individually
- **Accuracy**: High precision with perceptual color space analysis
- **Robustness**: Handles various lighting conditions and color palettes

## Applications

- **Emotional Drawing Ordering**: Draw warm regions first for emotional impact
- **Color-Aware Path Planning**: Optimize drawing sequence based on color temperature
- **Artistic Style Analysis**: Understand warm/cool balance in compositions
- **Robot Drawing Intelligence**: Make color-aware decisions during automated painting

## Files

- `color_detection.py`: Main implementation with CIELAB and HSV methods
- `README.md`: This documentation file
- Generated visualizations: `examples/color_detection_results_YYYYMMDD_HHMMSS.png`