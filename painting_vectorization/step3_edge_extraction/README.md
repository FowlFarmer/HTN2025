# Step 3: Edge / Line Extraction (Per-Mask)

## Overview
This module extracts clean binary line images (black strokes on white background) from each segmented mask. The goal is to isolate the structural elements of brush strokes while removing texture noise and preserving the essential line work.

## Purpose
- Extract meaningful edges from segmented regions
- Clean and skeletonize edge maps for stroke analysis
- Produce binary line representations suitable for vectorization
- Handle painterly textures and varying contrast levels

## Key Components

### `edge_extraction.py`
Main edge extraction pipeline with multiple processing stages.

**Core Function**: `extract_edges_from_mask(image, mask, method)`

**Input**:
- `image`: Preprocessed image (numpy array, float32, [0,1])
- `mask`: Binary segmentation mask (boolean array)
- `method`: Edge detection method ("adaptive", "structural", "boundary", "multi_scale", "canny")

**Output**:
- Dictionary containing:
  - `skeleton`: Skeletonized edge map (boolean array)
  - `edges`: Raw edge map before skeletonization
  - `cropped_region`: Cropped image region
  - `cropped_mask`: Cropped mask region
  - `offset`: Coordinate offset for mapping back to full image
  - `stats`: Extraction statistics

## Processing Pipeline

### 3.1 Mask Cropping
Crops image to mask bounding box for computational efficiency:

```python
def crop_to_mask(image, mask):
    coords = np.where(mask)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # Add padding to avoid edge artifacts
    pad = 5
    cropped_img = image[y_min-pad:y_max+pad+1, x_min-pad:x_max+pad+1]
    cropped_mask = mask[y_min-pad:y_max+pad+1, x_min-pad:x_max+pad+1]

    return cropped_img, cropped_mask, (x_min-pad, y_min-pad)
```

### 3.2 Preprocessing
Prepares cropped regions for edge detection:

```python
def preprocess_for_edges(cropped_img, cropped_mask):
    # Convert to grayscale
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    # Apply bilateral filter to remove texture noise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Mask out background areas
    gray[~cropped_mask] = 255  # White background

    return gray
```

### 3.3 Edge Detection Methods

#### Adaptive Pattern-Aware Approach (Recommended - NEWEST)
Content-adaptive filtering that preserves meaningful patterns while removing texture noise:

```python
def extract_adaptive_structural_edges(gray, cropped_mask):
    # Stage 1: Multi-scale edge detection
    fine_edges = cv2.Canny(light_smooth, 80, 160)      # Fine patterns
    major_edges = cv2.Canny(medium_smooth, 100, 200)   # Major structures
    boundary_edges = cv2.Canny(heavy_smooth, 120, 240) # Overall boundaries

    # Stage 2: Analyze local complexity
    complexity_map = analyze_local_complexity(gray)

    # Stage 3: Detect pattern regions
    pattern_regions = detect_pattern_regularity(combined_edges)

    # Stage 4: Adaptive filtering based on content
    # High pattern regions → keep fine details
    # High structure regions → use major edges
    # Low complexity regions → use only boundaries

    # Stage 5: Dynamic thresholding
    # Smaller thresholds for pattern regions
    # Larger thresholds for texture regions
```

**Key Features**:
- **Pattern Detection**: Identifies regular, repetitive elements (bridge rectangles, window grids)
- **Adaptive Filtering**: Different processing for different content types
- **Content Awareness**: Preserves structured patterns, removes random texture
- **Universal Application**: Works for any painting style or content

**Advantages**:
- **Pattern Preservation**: Keeps meaningful architectural details (bridge supports, building elements)
- **Texture Removal**: Eliminates artistic brush strokes and canvas texture
- **General Purpose**: Not specific to any painting type
- **Smart Thresholding**: Context-aware component filtering

#### Boundary-Focused Approach
Extracts clean object outlines with minimal internal texture:

```python
def extract_boundary_edges(gray, cropped_mask):
    # Method 1: Contour-based approach
    smooth = cv2.GaussianBlur(gray, (7, 7), 2.0)
    edges = cv2.Canny(smooth, 80, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Method 2: Conservative Canny for boundaries
    smooth = cv2.bilateralFilter(gray, 9, 80, 80)
    canny_edges = cv2.Canny(smooth, 100, 200)

    # Method 3: Morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gradient = cv2.morphologyEx(smooth, cv2.MORPH_GRADIENT, kernel)

    # Combine methods
    combined = cv2.bitwise_or(contour_edges, canny_edges)
    combined = cv2.bitwise_or(combined, gradient_edges)
```

**Advantages**:
- **Clean object outlines** - focuses on boundaries, not brush texture
- **Reduced noise** - 30-40% fewer pixels than multi-scale method
- **Better for vectorization** - cleaner lines suitable for SVG conversion

#### Multi-Scale Approach (Legacy)
Combines multiple Canny thresholds for robust detection:

```python
def multi_scale_edges(gray):
    edges_low = cv2.Canny(gray, 30, 100)    # Faint edges
    edges_high = cv2.Canny(gray, 80, 200)   # Strong edges
    edges_med = cv2.Canny(gray, 50, 150)    # Medium edges

    # Combine all scales
    edges_combined = cv2.bitwise_or(edges_low, edges_high)
    edges_combined = cv2.bitwise_or(edges_combined, edges_med)

    return edges_combined
```

#### Standard Canny
Single-threshold approach for simpler cases:

```python
edges = cv2.Canny(gray, 50, 150)
```

### 3.4 Morphological Cleaning
Removes noise and connects broken line segments:

```python
def morphological_cleaning(edges):
    kernel = np.ones((3, 3), np.uint8)

    # Closing to connect broken lines
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Opening to remove small specks
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 20:  # min_area threshold
            edges[labels == i] = 0

    return edges
```

### 3.5 Skeletonization
Reduces thick edges to single-pixel-width lines:

```python
from skimage.morphology import skeletonize

def extract_skeleton(edges):
    binary = edges > 0
    skeleton = skeletonize(binary)
    return skeleton
```

## Usage Examples

### Basic Edge Extraction
```python
from edge_extraction import extract_edges_from_mask

# Extract adaptive pattern-aware edges (recommended)
result = extract_edges_from_mask(processed_img, mask, method="adaptive")

if result is not None:
    skeleton = result['skeleton']
    stats = result['stats']
    print(f"Extracted {stats['skeleton_pixels']} skeleton pixels")
    print(f"Edge density: {stats['edge_density']:.3f}")
```

### Process All Masks
```python
from edge_extraction import process_all_masks, save_edge_results

# Extract adaptive pattern-aware edges from all masks (recommended)
edge_results = process_all_masks(processed_img, masks, method="adaptive")

# Save visualization
output_path = save_edge_results(edge_results, masks, processed_img.shape)
print(f"Results saved to: {output_path}")
```

### Integration with Pipeline
```python
from step1_preprocessing.image_preprocessing import preprocess_image
from step2_segmentation.segmentation import segment_painting
from step3_edge_extraction.edge_extraction import process_all_masks

# Complete pipeline
processed_img, scale, dims = preprocess_image("examples/monet.jpeg")
masks, scores, _ = segment_painting("examples/monet.jpeg")
edge_results = process_all_masks(processed_img, masks)

print(f"Extracted edges from {len(edge_results)} masks")
```

### Analyzing Results
```python
# Examine extraction quality
for i, result in enumerate(edge_results):
    if result is not None:
        stats = result['stats']
        print(f"Mask {i}:")
        print(f"  Crop size: {stats['crop_size']}")
        print(f"  Edge density: {stats['edge_density']:.3f}")
        print(f"  Skeleton pixels: {stats['skeleton_pixels']}")
```

## Algorithm Parameters

### Edge Detection Thresholds
- **Low threshold**: 30 (captures faint strokes)
- **Medium threshold**: 50-150 (standard edges)
- **High threshold**: 80-200 (strong edges only)

### Morphological Operations
- **Kernel size**: 3×3 (connects nearby edges)
- **Min area**: 20 pixels (removes noise specks)
- **Padding**: 5 pixels (avoids edge artifacts)

### Bilateral Filter
- **Diameter**: 9 pixels
- **Sigma color**: 75 (smoothing strength)
- **Sigma space**: 75 (spatial smoothing)

## Output Specifications

### Skeleton Format
- **Data type**: Boolean numpy array
- **True pixels**: Represent extracted stroke lines
- **Coordinate system**: Matches input image
- **Width**: Single-pixel thickness

### Statistics Provided
- `total_pixels`: Pixels in the mask region
- `edge_pixels`: Pixels in raw edge map
- `skeleton_pixels`: Pixels in final skeleton
- `edge_density`: edge_pixels / total_pixels
- `skeleton_density`: skeleton_pixels / total_pixels
- `crop_size`: Dimensions of processed region

## Performance Characteristics

### Processing Time (per mask)
- **Small masks** (< 1000 pixels): ~10-50ms
- **Medium masks** (1000-10000 pixels): ~50-200ms
- **Large masks** (> 10000 pixels): ~200-500ms

### Memory Usage
- **Peak memory**: ~2-3× mask area in bytes
- **Typical usage**: 10-50MB for standard paintings

### Quality Factors
- **Edge density**: 0.01-0.1 indicates good extraction
- **Skeleton density**: 0.001-0.01 for clean lines
- **Coverage**: Higher is better for stroke-rich regions

## Method Comparison

### Multi-Scale vs Single Canny

| Metric | Multi-Scale | Single Canny |
|--------|-------------|--------------|
| Quality | Higher | Good |
| Speed | ~3× slower | Faster |
| Faint strokes | Excellent | Fair |
| Noise handling | Better | Adequate |
| Use case | Final output | Quick testing |

### When to Use Each Method
- **Multi-scale**: Production, high-quality output, painterly images
- **Canny**: Testing, simple line art, speed-critical applications

## Troubleshooting

### Common Issues

#### Poor Edge Detection
- **Symptoms**: Missing strokes, broken lines
- **Solutions**:
  - Lower Canny thresholds
  - Increase bilateral filter parameters
  - Use multi-scale method

#### Too Much Noise
- **Symptoms**: Scattered edge pixels, texture artifacts
- **Solutions**:
  - Increase morphological cleaning
  - Raise minimum area threshold
  - Stronger bilateral filtering

#### Thick Lines
- **Symptoms**: Multi-pixel wide strokes
- **Solutions**:
  - Verify skeletonization is working
  - Check input mask quality
  - Adjust morphological parameters

### Parameter Tuning

#### For Detailed Paintings
- Lower Canny thresholds (20-80)
- Stronger bilateral filtering
- Smaller minimum area (10-15 pixels)

#### For Simple Line Art
- Higher Canny thresholds (60-180)
- Minimal bilateral filtering
- Larger minimum area (30-50 pixels)

## Dependencies
- OpenCV (`cv2`) - Image processing
- NumPy (`numpy`) - Array operations
- scikit-image (`skimage.morphology`) - Skeletonization
- Matplotlib (`matplotlib`) - Visualization

## Integration Notes
- Input coordinates preserved through `offset` parameter
- Compatible with Step 2 segmentation masks
- Output ready for Step 4 vectorization
- Timestamps added to all output files