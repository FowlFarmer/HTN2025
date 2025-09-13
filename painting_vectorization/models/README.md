# Models Directory

## Overview
This directory contains pre-trained models used in the painting vectorization pipeline. Models are automatically downloaded when needed or can be manually placed here.

## Current Models

### SAM (Segment Anything Model)
- **File**: `sam_vit_h_4b8939.pth`
- **Size**: ~2.4GB
- **Architecture**: Vision Transformer Huge (ViT-H)
- **Purpose**: Zero-shot segmentation for Step 2
- **Source**: [Facebook Research](https://github.com/facebookresearch/segment-anything)

**Download Command**:
```bash
curl -L -o sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

**Model Specifications**:
- **Input**: RGB images (any size)
- **Output**: Segmentation masks with confidence scores
- **Parameters**: 630M parameters
- **License**: Apache 2.0

## Usage in Pipeline

### Step 2 - Segmentation
```python
from step2_segmentation.segmentation import segment_painting

# Uses models/sam_vit_h_4b8939.pth automatically
masks, scores, processed_img = segment_painting(
    image_path="../examples/monet.jpeg",
    model_path="../models/sam_vit_h_4b8939.pth"
)
```

## Model Performance

### SAM ViT-H
- **Inference Speed**:
  - CPU: 30-60s per image (1024x1024)
  - GPU: 5-15s per image (1024x1024)
- **Memory Requirements**:
  - CPU: ~2-3GB RAM
  - GPU: ~4-6GB VRAM
- **Quality**: State-of-the-art zero-shot segmentation

## Alternative Models

### SAM ViT-L (Lighter Alternative)
- **File**: `sam_vit_l_0b3195.pth` (not included)
- **Size**: ~1.2GB
- **Speed**: ~2x faster than ViT-H
- **Quality**: Slightly lower than ViT-H

**Download**:
```bash
curl -L -o sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

### SAM ViT-B (Fastest)
- **File**: `sam_vit_b_01ec64.pth` (not included)
- **Size**: ~375MB
- **Speed**: ~4x faster than ViT-H
- **Quality**: Good for real-time applications

**Download**:
```bash
curl -L -o sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Model Selection Guide

### For Development/Testing
- **Recommended**: SAM ViT-B
- **Reason**: Fast iteration, good quality

### For Production/Demo
- **Recommended**: SAM ViT-H
- **Reason**: Highest quality segmentation

### For Real-time Applications
- **Recommended**: SAM ViT-L
- **Reason**: Best speed/quality balance

## Storage Notes
- Models are large files (375MB - 2.4GB)
- Consider using Git LFS for version control
- Models can be shared across multiple projects
- Download once, use everywhere

## Troubleshooting
- **Download fails**: Check internet connection, use alternative mirrors
- **Out of disk space**: SAM ViT-H requires ~2.5GB free space
- **Model loading errors**: Verify file integrity with checksum validation