# 🏗️ Structural Segmentation for Tactile Drawing

## 🎯 **Problem Solved**

**Previous Issue**: Color-based clustering created too many small segments, leading to:
- ❌ Fragmented drawing with many pen lifts
- ❌ Small color patches instead of coherent objects  
- ❌ Poor tactile experience for blind users
- ❌ No logical drawing sequence

**New Solution**: Structure-based segmentation focuses on major shapes and regions:
- ✅ **3-6 large, coherent regions** instead of 8+ color fragments
- ✅ **Minimal pen lifts** with smooth, continuous strokes
- ✅ **Logical drawing order** (large objects first)
- ✅ **Better tactile recognition** of distinct shapes
- ✅ **Optimized for blind users** to feel object boundaries

## 🔄 **How to Switch to Structural Segmentation**

### **1. Automatic (Recommended)**
The preprocessing now defaults to structural segmentation:

```python
# This now uses structural segmentation by default
processed_img, scale, dims = preprocess_image("examples/tower.jpg")
```

### **2. Explicit Configuration**
```python
# Structural segmentation (recommended for tactile drawing)
processed_img, scale, dims = preprocess_image(
    "examples/tower.jpg",
    segmentation_method='structural',
    segmentation_params={
        'method': 'contour_hierarchy',  # Focus on major shapes
        'max_regions': 5,               # Limit to 5 major regions
        'min_area_ratio': 0.03,         # Ignore tiny details (< 3% of image)
        'smoothing': 5                  # Smooth boundaries
    }
)

# Fallback to color clustering if needed
processed_img, scale, dims = preprocess_image(
    "examples/tower.jpg", 
    segmentation_method='color_clustering'
)
```

## 🏗️ **Structural Segmentation Methods**

### **1. Contour Hierarchy (Default)**
- **Best for**: Buildings, objects with clear boundaries
- **Focus**: Major shapes and structural elements
- **Parameters**:
  - `max_regions`: 3-7 (fewer = larger regions)
  - `min_area_ratio`: 0.02-0.05 (higher = ignore smaller details)
  - `smoothing`: 3-7 (higher = smoother boundaries)

### **2. Watershed Regions**
- **Best for**: Natural scenes, organic shapes
- **Focus**: Natural region boundaries
- **Parameters**:
  - `num_markers`: 4-8 (number of seed regions)
  - `compactness`: 0.1-0.5 (higher = more compact regions)

### **3. Gradient Regions**
- **Best for**: High-contrast images
- **Focus**: Texture and pattern differences
- **Parameters**:
  - `gradient_threshold`: 20-40 (higher = fewer regions)
  - `min_region_size`: 0.02-0.05 (minimum region size)

## 🎨 **Testing and Comparison**

### **Run Comparison Test**
```bash
cd painting_vectorization
python test_structural_segmentation.py
```

This will:
- Compare structural vs color-based approaches
- Test different parameter settings
- Save detailed visualizations to `results/step1_preprocessing/`
- Show tactile drawing benefits analysis

### **Manual Testing**
```bash
cd step1_preprocessing
python structural_segmentation.py
```

## 📊 **Expected Results**

### **Before (Color Clustering)**
```
🎨 Found 8-12 color regions:
  - Sky patches (3 different blue shades)
  - Building fragments (5 different stone colors)  
  - Tree pieces (4 different green variations)
  → 12+ pen lifts, fragmented drawing
```

### **After (Structural Segmentation)**
```
🏗️ Found 5 major structural regions:
  Region 1: 35.2% area, priority=0.847 (Sky)
  Region 2: 28.1% area, priority=0.723 (Main Building)  
  Region 3: 18.5% area, priority=0.612 (Foreground)
  Region 4: 12.3% area, priority=0.445 (Trees)
  Region 5: 5.9% area, priority=0.234 (Details)
  → 5 pen lifts, smooth continuous drawing
```

## 🤚 **Tactile Drawing Benefits**

### **For Blind Users**:
1. **Logical Sequence**: Draw major objects first (sky → building → foreground)
2. **Smooth Motions**: Each region is one continuous stroke
3. **Shape Recognition**: Feel complete object boundaries, not color fragments
4. **Minimal Interruption**: Fewer pen lifts = better tactile flow
5. **Intuitive Learning**: Objects drawn as coherent wholes

### **Drawing Order Optimization**:
- **Priority 1**: Large background elements (sky, walls)
- **Priority 2**: Major foreground objects (buildings, trees)  
- **Priority 3**: Medium details (windows, branches)
- **Priority 4**: Fine details (texture, small elements)

## 🔧 **Parameter Tuning Guide**

### **For Fewer, Larger Regions** (Recommended for tactile):
```python
segmentation_params = {
    'max_regions': 3,        # Only 3 major regions
    'min_area_ratio': 0.05,  # Ignore details < 5% of image
    'smoothing': 7           # Very smooth boundaries
}
```

### **For More Detail** (If needed):
```python
segmentation_params = {
    'max_regions': 7,        # Up to 7 regions
    'min_area_ratio': 0.02,  # Include smaller details
    'smoothing': 3           # Preserve more detail
}
```

### **For Different Image Types**:

**Architectural/Geometric**:
```python
segmentation_params = {
    'method': 'contour_hierarchy',
    'max_regions': 5,
    'min_area_ratio': 0.03
}
```

**Natural/Organic**:
```python
segmentation_params = {
    'method': 'watershed_regions', 
    'num_markers': 6,
    'compactness': 0.2
}
```

**High Contrast**:
```python
segmentation_params = {
    'method': 'gradient_regions',
    'gradient_threshold': 30,
    'min_region_size': 0.03
}
```

## 📁 **Output Files**

All results saved to `results/step1_preprocessing/`:

### **Structural Segmentation Results**:
- `structural_segmentation_[image]_[time].png` - Detailed 4-panel analysis
- `approach_comparison_[image]_[time].png` - Structural vs color comparison
- `parameter_tests_[time].png` - Different parameter settings

### **Analysis Information**:
- Region priorities for drawing order
- Area percentages for each region
- Tactile drawing recommendations

## 🚀 **Integration with Pipeline**

The structural segmentation automatically integrates with the rest of the pipeline:

1. **Step 1**: Structural segmentation creates major regions
2. **Step 2**: SAM segments each major region (better results!)
3. **Step 3-8**: Edge extraction, vectorization, etc. work on coherent regions
4. **Result**: Smooth, continuous strokes optimized for tactile drawing

## 🎯 **Key Benefits Summary**

| Aspect | Color Clustering | Structural Segmentation |
|--------|------------------|------------------------|
| **Regions** | 8-12 color patches | 3-6 major shapes |
| **Pen Lifts** | 8-12+ interruptions | 3-6 smooth transitions |
| **Tactile Experience** | Fragmented | Coherent objects |
| **Drawing Logic** | Random color order | Size/importance priority |
| **Boundary Quality** | Color-based edges | Structural boundaries |
| **User Experience** | Confusing fragments | Intuitive shapes |

**→ Result: Much better tactile drawing experience with minimal pen lifts and logical drawing sequence!**
