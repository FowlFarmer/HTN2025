# Step 1: Image Preprocessing

## Overview
This module handles the initial preprocessing of painting images to prepare them for ML-based segmentation and stroke extraction. The preprocessing pipeline ensures images are in the optimal format, size, and quality for downstream analysis.

## Purpose
- Convert images to standardized format (RGB, float32, [0,1] range)
- Resize large images while preserving aspect ratio
- Enhance contrast for better edge detection
- Prepare data for ML model compatibility

## Key Components

### `image_preprocessing.py`
Main preprocessing pipeline implementation.

**Core Function**: `preprocess_image(image_path)`

**Input**:
- `image_path` (str): Path to painting image file

**Output**:
- `processed_image`: numpy.ndarray with shape (height, width, 3), dtype float32, values in [0.0, 1.0], RGB color order
- `scale_factor`: float, resize ratio applied to original image
- `original_dimensions`: tuple (original_height, original_width) in pixels

## Processing Steps

### 1.1 Image Loading and Format Conversion
```python
img = cv2.imread(image_path)[:, :, ::-1]  # BGR -> RGB
if img.shape[2] == 4:  # RGBA
    img = img[:, :, :3]  # Remove alpha channel
```

### 1.2 Resizing Strategy
- **Target**: Long edge ≤ 1024 pixels
- **Method**: Preserve aspect ratio using `cv2.INTER_AREA`
- **Rationale**: Keeps ML models and processing time reasonable

### 1.3 Contrast Enhancement
- **Method**: CLAHE on luminance channel
- **Parameters**: `clipLimit=2.0`, `tileGridSize=(8,8)`
- **Purpose**: Helps edge detectors pick up subtle strokes

### 1.4 Data Type Conversion
- Convert to float32 in [0,1] range for ML compatibility

## Usage Examples

### Basic Usage
```python
from image_preprocessing import preprocess_image

# Process a painting
processed_img, scale_factor, original_dims = preprocess_image("../examples/monet.jpeg")

print(f"Original: {original_dims}, Scale: {scale_factor}")
print(f"Processed shape: {processed_img.shape}, dtype: {processed_img.dtype}")
```

### Integration with Pipeline
```python
# Use preprocessed image for segmentation
from step2_segmentation.segmentation import segment_painting

masks, scores, processed_img = segment_painting("../examples/monet.jpeg")
```

## Dependencies
- OpenCV (`cv2`)
- NumPy (`numpy`)

## Output Specifications
- **Image format**: RGB color space
- **Data type**: float32
- **Value range**: [0.0, 1.0]
- **Coordinate system**: Standard image coordinates (y, x, channel)
- **Size constraint**: Long edge ≤ 1024 pixels (preserves aspect ratio)

## Performance Notes
- Processing time: ~50-200ms for typical paintings
- Memory usage: ~12MB for 1024x1024 RGB float32 image
- No GPU required for this step