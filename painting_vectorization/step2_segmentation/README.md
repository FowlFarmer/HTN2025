# Step 2: Segmentation / Element Discovery

## Overview
This module splits paintings into meaningful visual elements (sky, buildings, trees, focal areas, etc.) using the Segment Anything Model (SAM). Each mask isolates a major element for separate stroke extraction in later pipeline steps.

## Purpose
- Generate comprehensive mask proposals using SAM
- Score masks based on visual importance and structure
- Select 3-6 optimal masks for hackathon demo
- Merge overlapping masks to avoid redundancy

## Key Components

### `segmentation.py`
Main segmentation pipeline using SAM for automatic mask generation.

**Core Function**: `segment_painting(image_path, model_path)`

**Input**:
- `image_path` (str): Path to painting image
- `model_path` (str): Path to SAM model checkpoint (default: "../models/sam_vit_h_4b8939.pth")

**Output**:
- `selected_masks`: List of boolean numpy arrays representing segmentation masks
- `mask_scores`: List of dictionaries with scoring metrics for each mask
- `processed_image`: Preprocessed input image (numpy array, float32, [0,1])

## Algorithm Details

### 2.1 SAM Integration
Uses Facebook's Segment Anything Model for zero-shot segmentation:
```python
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)
```

### 2.2 Mask Scoring System
Each mask is evaluated using multiple criteria:

```python
def score_mask(mask, image, edge_map):
    # Area score (larger elements are more important)
    area_score = np.sum(mask) / (image.shape[0] * image.shape[1])

    # Edge strength (more structured strokes)
    edge_strength = np.mean(edge_map[mask])

    # Color distinctiveness
    mask_pixels = image[mask]
    outside_pixels = image[~mask]
    color_distinctiveness = np.std(mask_pixels) / (np.std(outside_pixels) + 1e-6)

    # Combined salience score
    salience = area_score * edge_strength * color_distinctiveness

    return {
        'area_score': area_score,
        'edge_strength': edge_strength,
        'color_distinctiveness': color_distinctiveness,
        'salience': salience
    }
```

### 2.3 Mask Selection Parameters
- **Target count**: 3-6 masks for demo
- **Minimum area**: >1% of image area
- **Maximum area**: <80% of image area
- **IoU threshold**: 0.6 for mask merging

### 2.4 Mask Merging Strategy
Overlapping masks with IoU > 0.6 are merged to reduce redundancy:
```python
def merge_masks(masks, iou_threshold=0.6):
    # Merge highly overlapping regions
    # Returns consolidated mask list
```

## Usage Examples

### Basic Segmentation
```python
from segmentation import segment_painting

# Segment a painting
masks, scores, processed_img = segment_painting("../examples/monet.jpeg")

print(f"Generated {len(masks)} masks")
for i, score in enumerate(scores):
    print(f"Mask {i}: salience={score['salience']:.3f}")
```

### Accessing Individual Masks
```python
# Work with individual masks
for i, mask in enumerate(masks):
    print(f"Mask {i} covers {np.sum(mask)} pixels")

    # Extract region
    masked_image = processed_img.copy()
    masked_image[~mask] = 0  # Zero out non-mask areas
```

### Visualization
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, (mask, score) in enumerate(zip(masks, scores)):
    ax = axes[i // 3, i % 3]
    ax.imshow(processed_img)
    ax.imshow(mask, alpha=0.5, cmap='viridis')
    ax.set_title(f'Mask {i} (salience: {score["salience"]:.3f})')
    ax.axis('off')
```

## Dependencies
- PyTorch (`torch`, `torchvision`)
- Segment Anything (`segment-anything`)
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`) for visualization

## Model Requirements
- **SAM Model**: ViT-H checkpoint (~2.4GB)
- **Download**: Automatic via `segment_painting()` or manual from Facebook Research
- **GPU**: Optional but recommended for faster inference
- **Memory**: ~4-6GB GPU memory for ViT-H model

## Output Specifications
- **Mask format**: Boolean numpy arrays matching image dimensions
- **Mask count**: 3-6 optimal masks (configurable)
- **Coordinate system**: Same as input image
- **Quality metrics**: Salience scores combining area, edge strength, and color distinctiveness

## Performance Notes
- **Processing time**:
  - CPU: 30-60 seconds for 1024x1024 image
  - GPU: 5-15 seconds for 1024x1024 image
- **Memory usage**:
  - CPU: ~2-3GB RAM
  - GPU: ~4-6GB VRAM
- **Model loading**: ~5-10 seconds (first run only)

## Troubleshooting
- **CUDA out of memory**: Use CPU device or reduce image size
- **Too many masks**: Increase `pred_iou_thresh` and `stability_score_thresh`
- **Too few masks**: Decrease thresholds or increase `points_per_side`