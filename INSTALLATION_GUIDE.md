# Installation Guide - Painting Vectorization Pipeline

This guide will help you set up the painting vectorization pipeline to run `python process_image.py` successfully.

## Prerequisites

- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **Git** (for cloning the repository)
- **Internet connection** (for downloading dependencies and SAM model)
- **At least 4GB RAM** (8GB+ recommended for large images)
- **3GB free disk space** (for SAM model and dependencies)

## Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd HTN2025
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### 4. Download SAM Model
The SAM (Segment Anything Model) is required for image segmentation but is too large for Git. Download it manually:

```bash
# Create models directory
mkdir -p painting_vectorization/models

# Download SAM ViT-H model (2.4GB) - Highest quality
curl -L -o painting_vectorization/models/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

**Alternative SAM Models** (if you want faster processing):
```bash
# SAM ViT-L (1.2GB) - Good balance of speed/quality
curl -L -o painting_vectorization/models/sam_vit_l_0b3195.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# SAM ViT-B (375MB) - Fastest, good for testing
curl -L -o painting_vectorization/models/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 5. Test Installation
```bash
# Navigate to painting vectorization directory
cd painting_vectorization

# Test with interactive mode
python process_image.py

# Or test with a specific image
python process_image.py examples/monet.jpeg
```

## Detailed Installation Steps

### Python Environment Setup

#### Option A: Using Virtual Environment (Recommended)
```bash
# Create isolated environment
python -m venv venv

# Activate environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip
```

#### Option B: Using Conda
```bash
# Create conda environment
conda create -n painting_vectorization python=3.9
conda activate painting_vectorization

# Install PyTorch with CUDA support (if you have GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Dependency Installation

#### Core Dependencies
```bash
# Install from requirements.txt (recommended)
pip install -r requirements.txt

# Or install individually:
pip install numpy>=1.21.0
pip install opencv-python>=4.5.0
pip install torch>=1.9.0
pip install torchvision>=0.10.0
pip install segment-anything>=1.0
pip install scikit-image>=0.18.0
pip install networkx>=2.6.0
pip install matplotlib>=3.5.0
pip install scipy>=1.7.0
pip install sknw>=0.11.0
```

#### Optional Dependencies
```bash
# For drawing robot communication
pip install pyserial>=3.5

# For GPU acceleration (if you have CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### SAM Model Setup

The SAM model is essential for Step 2 (segmentation). Choose based on your needs:

#### High Quality (Recommended for Production)
```bash
# SAM ViT-H - Best quality, slower processing
curl -L -o painting_vectorization/models/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
- **Size**: 2.4GB
- **Speed**: 5-15s per image (GPU), 30-60s (CPU)
- **Quality**: Highest segmentation accuracy

#### Balanced Performance
```bash
# SAM ViT-L - Good balance
curl -L -o painting_vectorization/models/sam_vit_l_0b3195.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```
- **Size**: 1.2GB
- **Speed**: ~2x faster than ViT-H
- **Quality**: Slightly lower than ViT-H

#### Fast Testing
```bash
# SAM ViT-B - Fastest
curl -L -o painting_vectorization/models/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
- **Size**: 375MB
- **Speed**: ~4x faster than ViT-H
- **Quality**: Good for real-time applications

## Verification

### Test Basic Functionality
```bash
cd painting_vectorization

# List available images
python process_image.py --list

# Show recent results
python process_image.py --show-results

# Process an example image
python process_image.py examples/monet.jpeg
```

### Expected Output
You should see:
1. **Step 1**: Image preprocessing with scale factor
2. **Step 2**: Segmentation with mask count and coverage
3. **Step 3**: Edge extraction with skeleton statistics
4. **Step 4**: Stroke graph construction with node/edge counts
5. **Output files**: Generated in `examples/` directory

## Troubleshooting

### Common Issues

#### 1. SAM Model Not Found
```
FileNotFoundError: SAM model not found
```
**Solution**: Download the SAM model as described above.

#### 2. CUDA/GPU Issues
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Use CPU-only PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- Use smaller SAM model (ViT-B instead of ViT-H)
- Process smaller images

#### 3. Missing Dependencies
```
ModuleNotFoundError: No module named 'sknw'
```
**Solution**: Install missing package:
```bash
pip install sknw
```

#### 4. Memory Issues
```
MemoryError: Unable to allocate array
```
**Solutions**:
- Close other applications
- Use smaller images
- Process images in smaller batches

#### 5. Download Failures
```
curl: (7) Failed to connect to dl.fbaipublicfiles.com
```
**Solutions**:
- Check internet connection
- Try alternative download methods:
  ```bash
  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  ```
- Use browser to download and place in `painting_vectorization/models/`

### Performance Optimization

#### For Faster Processing
1. **Use GPU**: Install CUDA-enabled PyTorch
2. **Smaller model**: Use SAM ViT-B instead of ViT-H
3. **Smaller images**: Resize input images to 1024x1024 or smaller

#### For Better Quality
1. **Larger model**: Use SAM ViT-H
2. **Higher resolution**: Process images at original size
3. **GPU acceleration**: Reduces processing time significantly

## System Requirements

### Minimum Requirements
- **RAM**: 4GB
- **Storage**: 3GB free space
- **CPU**: Any modern processor
- **Python**: 3.8+

### Recommended Requirements
- **RAM**: 8GB+
- **Storage**: 5GB free space
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **CPU**: Multi-core processor
- **Python**: 3.9 or 3.10

## Next Steps

After successful installation:

1. **Add your images** to `painting_vectorization/examples/`
2. **Run the pipeline**: `python process_image.py`
3. **View results** in the `examples/` directory
4. **Customize parameters** in individual step modules
5. **Integrate with drawing robots** using the generated stroke graphs

## Getting Help

If you encounter issues:

1. **Check this guide** for common solutions
2. **Verify all dependencies** are installed correctly
3. **Ensure SAM model** is downloaded and in correct location
4. **Check system requirements** match your setup
5. **Review error messages** for specific guidance

## File Structure After Installation

```
HTN2025/
├── painting_vectorization/
│   ├── models/
│   │   └── sam_vit_h_4b8939.pth  # Downloaded SAM model
│   ├── examples/
│   │   ├── monet.jpeg            # Example images
│   │   └── *.png                 # Generated results
│   ├── step1_preprocessing/
│   ├── step2_segmentation/
│   ├── step3_edge_extraction/
│   ├── step4_stroke_graph/
│   └── process_image.py          # Main script
├── requirements.txt              # Dependencies
└── INSTALLATION_GUIDE.md         # This file
```

You're now ready to vectorize paintings! 🎨
