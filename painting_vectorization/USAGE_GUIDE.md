# Painting Vectorization - Usage Guide

## Quick Start: Processing Any Image

You can now easily process any image through the complete pipeline (Steps 1-3) using the `process_image.py` script.

## 🚀 Running the Pipeline

### Method 1: Interactive Mode (Recommended)
```bash
python process_image.py
```

This will:
1. Show all available images in the `examples/` directory
2. Let you select which image to process
3. Run the complete pipeline
4. Save timestamped results

**Example Output:**
```
📂 Available images in examples/ directory:
==================================================
   1. monet.jpeg (0.1 MB)
   2. starry_night.jpg (0.1 MB)
   3. your_painting.png (2.3 MB)

Select image (1-3) or 'q' to quit: 2
```

### Method 2: Direct Image Processing
```bash
# Process specific image by name
python process_image.py monet.jpeg

# Process with full path
python process_image.py examples/starry_night.jpg

# Process image from current directory
python process_image.py /path/to/your/painting.jpg
```

### Method 3: Utility Commands
```bash
# List all available images
python process_image.py --list

# Show recent processing results
python process_image.py --show-results
```

## 📁 File Organization

### Input Images
Place your images in the `examples/` directory:
```
examples/
├── monet.jpeg           # Original test image
├── starry_night.jpg     # Van Gogh example
├── your_painting.png    # Your custom images
└── ...
```

**Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`

### Output Files
Results are automatically saved with timestamps:

```
examples/
├── segmentation_results_20250913_134146.png     # Step 2 output
├── edge_extraction_results_20250913_134146.png  # Step 3 output
└── ...
```

## 🎯 Understanding the Results

### Step 1: Preprocessing
- **What it does**: Normalizes image format, size, contrast
- **Output**: Preprocessed image ready for ML processing
- **Time**: ~50-100ms

### Step 2: Segmentation
- **What it does**: Splits painting into meaningful regions using SAM
- **Output**: 3-6 segmentation masks + visualization
- **Key metrics**:
  - **Coverage**: % of image covered by all masks (aim for 80%+)
  - **Salience**: Quality score for each mask (higher = better)
- **Time**: 10-60s (CPU), 3-15s (GPU)

### Step 3: Edge Extraction
- **What it does**: Extracts clean line drawings from each mask
- **Output**: Skeletonized edge maps + visualization
- **Key metrics**:
  - **Edge density**: Raw edge pixels / total pixels (0.1-0.5 typical)
  - **Skeleton density**: Final line pixels / total pixels (0.01-0.1 typical)
  - **Skeleton pixels**: Total extracted line work
- **Time**: 100-500ms

## 📊 Example Processing Session

```bash
python process_image.py starry_night.jpg
```

**Expected Output:**
```
🎨 Processing: starry_night.jpg
============================================================
📐 Step 1: Image Preprocessing
------------------------------
   📏 Original: (606, 480)
   📐 Scale: 1.000
   🖼️  Shape: (606, 480, 3)
   ⏱️  Time: 0.074s
   ✅ Complete!

🎯 Step 2: Segmentation
------------------------------
   🎭 Masks: 4
   • Mask 1: 52.3% area, salience=0.189
   • Mask 2: 28.7% area, salience=0.081
   • Mask 3: 15.2% area, salience=0.045
   • Mask 4: 3.8% area, salience=0.012
   📊 Coverage: 91.2%
   ⏱️  Time: 87.3s
   ✅ Complete!

🖋️  Step 3: Edge Extraction
------------------------------
   ✅ Success: 4/4 masks
   📏 Avg edge density: 0.421
   🖋️  Avg skeleton density: 0.089
   📊 Skeleton pixels: 23,147
   ⏱️  Time: 0.156s
   ✅ Complete!

📋 Processing Summary
------------------------------
   🖼️  Input: starry_night.jpg
   📐 Resolution: (606, 480) -> (606, 480)
   🎭 Segments: 4
   🖋️  Extractions: 4
   ⏱️  Total time: 87.8s

📁 Output Files:
   🎯 Segmentation: examples/segmentation_results_20250913_140245.png
   🖋️  Edge extraction: examples/edge_extraction_results_20250913_140245.png
```

## 🔍 Viewing Results

### Recent Results
```bash
python process_image.py --show-results
```

**Output:**
```
📊 Recent Results
========================================
🖋️  Edge Extraction Results:
   • edge_extraction_results_20250913_140245.png (14:02:45)
   • edge_extraction_results_20250913_134146.png (13:41:46)
   • edge_extraction_results_20250913_133555.png (13:35:55)
```

### Opening Result Files
The output PNG files can be opened with any image viewer:

1. **Segmentation Results**: Shows original image with colored mask overlays
2. **Edge Extraction Results**: Shows extracted line drawings (black lines on white background)

## ⚙️ Advanced Usage

### Processing Multiple Images
```bash
# Process all images in examples/
for img in examples/*.jpg examples/*.png; do
    python process_image.py "$img"
done
```

### Custom Model Path
```bash
# Use different SAM model
python process_image.py your_image.jpg --model-path /path/to/sam_model.pth
```

### Integration with Python Code
```python
from process_image import process_painting_pipeline

# Process an image programmatically
result = process_painting_pipeline("examples/monet.jpeg")

if result['success']:
    print(f"Processed {result['masks']} segments")
    print(f"Output: {result['edge_output_path']}")
```

## 🛠 Troubleshooting

### Common Issues

#### "Image not found"
```
❌ Image not found: my_image.jpg
💡 Try:
   python process_image.py --list
```
**Solution**: Check image is in `examples/` directory or use full path

#### "SAM model not found"
```
❌ Error: SAM model not found
```
**Solution**: Download the model to `models/` directory:
```bash
curl -L -o models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

#### "Out of memory"
**Solution**: Use smaller images or add more RAM/GPU memory

#### Poor segmentation quality
- **Low coverage** (<80%): Try different images or adjust SAM parameters
- **Too many small masks**: Increase minimum area threshold
- **Missing details**: Use higher resolution input images

#### Poor edge extraction
- **Too much noise**: Increase morphological cleaning parameters
- **Missing lines**: Lower Canny thresholds
- **Thick lines**: Verify skeletonization is working

### Performance Tips

1. **Use GPU acceleration**: Install CUDA-compatible PyTorch for 5-10x speedup
2. **Resize large images**: Images >2000px will be slow
3. **Close other applications**: Free up RAM for processing
4. **Use SSD storage**: Faster file I/O improves performance

## 📋 Quality Assessment

### Good Results Indicators
- **Coverage**: 80%+ of image covered by masks
- **Segment count**: 3-6 meaningful regions
- **Edge density**: 0.1-0.5 (good edge detection)
- **Skeleton density**: 0.01-0.1 (clean lines)
- **Processing time**: <2 minutes total

### When to Adjust Parameters
- **Too few masks**: Lower SAM thresholds
- **Too many masks**: Raise SAM thresholds
- **Noisy edges**: Increase bilateral filtering
- **Missing edges**: Lower Canny thresholds
- **Thick lines**: Check skeletonization settings

## 🎯 Next Steps

After processing images through Steps 1-3:

1. **Analyze Results**: Open the generated PNG files to evaluate quality
2. **Iterate**: Adjust parameters and reprocess if needed
3. **Vector Reconstruction**: Implement Step 4 to create SVG output
4. **Batch Processing**: Process multiple paintings for comparison
5. **Custom Models**: Train specialized models for your art style

---

*For technical details, see the README files in each step directory.*