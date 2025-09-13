# Painting Vectorization Pipeline Specification

## Overview

This document outlines an 8-step pipeline for converting digital paintings into vectorized stroke data suitable for robotic drawing systems. The pipeline processes input images through segmentation, edge extraction, graph construction, vectorization, and optimization to produce ordered stroke sequences with color temperature classification.

## Pipeline Architecture

```
Input Image → Preprocessing → Segmentation → Edge Extraction → Graph Construction → Vectorization → Sampling → Color Analysis → Stroke Ordering → Output
```

## Step 1: Image Preprocessing

### Goal
Make input robust for segmentation and edge extraction while maintaining computational efficiency.

### Implementation Details

#### 1.1 Image Loading and Format Conversion
```python
# Load image and convert RGBA to RGB
img = cv2.imread("painting.jpg")[:, :, ::-1]  # BGR -> RGB
if img.shape[2] == 4:  # RGBA
    img = img[:, :, :3]  # Remove alpha channel
```

#### 1.2 Resizing Strategy
- **Target**: Long edge ≤ 1024 pixels
- **Method**: Preserve aspect ratio using `cv2.INTER_AREA` interpolation
- **Rationale**: Keeps ML models and processing time reasonable
- **Fallback**: Store original dimensions for final coordinate mapping

```python
h, w = img.shape[:2]
scale = 1024 / max(h, w) if max(h, w) > 1024 else 1.0
img_small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
```

#### 1.3 Denoising (Optional)
- **Method**: `cv2.fastNlMeansDenoisingColored` or mild Gaussian blur (σ=0.5–1 px)
- **When to apply**: Only if image is visibly noisy
- **Caution**: Avoid over-smoothing that removes fine stroke details

#### 1.4 Contrast Enhancement
- **Method**: CLAHE (Contrast Limited Adaptive Histogram Equalization) on luminance channel
- **Parameters**: `clipLimit=2.0`, `tileGridSize=(8,8)`
- **Purpose**: Helps edge detectors pick up subtle strokes

```python
lab = cv2.cvtColor(img_small, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l = clahe.apply(l)
img_small = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)
```

#### 1.5 Data Type Conversion
- Convert to float in [0,1] range for ML model compatibility
- Store original scale factor for coordinate mapping

### Pitfalls and Mitigations
- **Upscaling small images**: Reduces quality; request higher resolution input
- **Over-aggressive CLAHE**: Can amplify brush texture; tune parameters carefully
- **Memory usage**: Monitor for large images; implement streaming if needed

---

## Step 2: Segmentation / Element Discovery

### Goal
Split the painting into n meaningful elements (sky, building, tree, focal swirl, etc.) where each mask isolates a single major element for separate stroke extraction.

### Implementation Strategy

#### 2.1 Segmentation Method Selection
**Primary**: SAM (Segment Anything Model)
- **Advantages**: Zero-shot segmentation, automatic mask proposals
- **Output**: Multiple overlapping masks of varying sizes
- **Integration**: Use pre-trained SAM model via `sam2` or `segment-anything` library

**Alternative**: Detectron2/Mask R-CNN + Grounding DINO + CLIP
- **Use case**: When semantic labels are required
- **Complexity**: Higher computational overhead

#### 2.2 Mask Selection and Scoring
Generate comprehensive mask proposals and score each using heuristics:

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

#### 2.3 Mask Selection Parameters
- **Target count**: 3-6 masks for hackathon demo
- **Minimum area**: >1% of image area
- **Maximum area**: <80% of image area (avoid background-only masks)

#### 2.4 Mask Merging Strategy
```python
def merge_masks(masks, iou_threshold=0.6):
    """Merge overlapping masks with IoU > threshold"""
    merged = []
    used = set()
    
    for i, mask1 in enumerate(masks):
        if i in used:
            continue
            
        current_mask = mask1.copy()
        for j, mask2 in enumerate(masks[i+1:], i+1):
            if j in used:
                continue
                
            iou = calculate_iou(mask1, mask2)
            if iou > iou_threshold:
                current_mask = np.logical_or(current_mask, mask2)
                used.add(j)
        
        merged.append(current_mask)
        used.add(i)
    
    return merged
```

#### 2.5 Large Mask Splitting
For masks containing multiple visually distinct elements:
1. Apply edge detection within mask
2. Find connected components of edges
3. Cluster edge components spatially
4. Split mask based on component clusters

### Pitfalls and Mitigations
- **Over-segmentation**: Filter by minimum area threshold
- **Under-segmentation**: Implement mask splitting for large masks
- **Computational cost**: Use GPU acceleration for SAM when available

---

## Step 3: Edge / Line Extraction (Per-Mask)

### Goal
For each selected mask, produce a clean binary line image (black strokes on white background) focusing only on the element.

### Implementation Approaches

#### 3.1 Method Selection
**Deterministic (Fast)**: Canny edge detector → morphological cleaning → skeletonize
**Learning-based (Higher Quality)**: HED/U2-Net/custom line-art model → threshold → clean → skeletonize

#### 3.2 Processing Pipeline

##### 3.2.1 Mask Cropping
```python
def crop_to_mask(image, mask):
    """Crop image to mask bounding box for efficiency"""
    coords = np.where(mask)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    cropped_img = image[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
    
    return cropped_img, cropped_mask, (x_min, y_min)
```

##### 3.2.2 Grayscale Conversion and Preprocessing
```python
# Convert to grayscale
gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

# Apply bilateral filter to remove texture noise while preserving edges
gray = cv2.bilateralFilter(gray, 9, 75, 75)
```

##### 3.2.3 Edge Detection

**Canny Method**:
```python
edges = cv2.Canny(gray, 50, 150)  # Adjust thresholds based on image
```

**HED Method** (for painterly strokes):
```python
# Load pre-trained HED model
hed_model = load_hed_model()
edge_confidence = hed_model.predict(gray)
edges = (edge_confidence > 0.25).astype(np.uint8) * 255
```

##### 3.2.4 Morphological Cleaning
```python
# Closing to connect broken lines
kernel = np.ones((3,3), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Opening to remove small specks
edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

# Remove small connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges)
min_area = 20  # pixels
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] < min_area:
        edges[labels == i] = 0
```

##### 3.2.5 Skeletonization
```python
from skimage.morphology import skeletonize

binary = edges > 0
skeleton = skeletonize(binary)
```

#### 3.3 Multi-Scale Edge Enhancement
For faint strokes, combine multiple edge scales:
```python
def multi_scale_edges(gray):
    # Low threshold for faint edges
    edges_low = cv2.Canny(gray, 30, 100)
    # High threshold for strong edges
    edges_high = cv2.Canny(gray, 80, 200)
    # Combine
    edges_combined = cv2.bitwise_or(edges_low, edges_high)
    return edges_combined
```

### Pitfalls and Mitigations
- **Canvas texture noise**: Use mask restriction and morphological cleaning
- **Over-thinning**: Implement multi-scale edge detection
- **Broken strokes**: Use morphological closing with appropriate kernel size

---

## Step 4: Skeleton → Stroke Graph Construction

### Goal
Convert the skeleton bitmap into a graph structure where nodes represent junctions/endpoints and edges represent continuous pixel sequences.

### Implementation Details

#### 4.1 Graph Construction
```python
import sknw

def build_stroke_graph(skeleton):
    """Convert skeleton to networkx-like graph"""
    G = sknw.build_sknw(skeleton.astype('uint8'))
    return G
```

#### 4.2 Graph Analysis
```python
def analyze_graph(G):
    """Analyze graph structure for traversal planning"""
    nodes = list(G.nodes())
    edges = list(G.edges(data=True))
    
    # Identify node types
    endpoints = [n for n in nodes if G.degree(n) == 1]
    junctions = [n for n in nodes if G.degree(n) > 2]
    
    return {
        'nodes': nodes,
        'edges': edges,
        'endpoints': endpoints,
        'junctions': junctions,
        'is_eulerian': is_eulerian_graph(G)
    }
```

#### 4.3 Stroke Repair
```python
def repair_fragmented_strokes(skeleton, max_gap=6):
    """Connect nearby endpoints to repair broken strokes"""
    # Find endpoints
    endpoints = find_endpoints(skeleton)
    
    # Connect nearby endpoints
    for i, ep1 in enumerate(endpoints):
        for ep2 in endpoints[i+1:]:
            distance = np.linalg.norm(np.array(ep1) - np.array(ep2))
            if distance <= max_gap:
                # Draw line between endpoints
                cv2.line(skeleton, ep1, ep2, 1, 1)
    
    # Re-skeletonize if needed
    return skeletonize(skeleton)
```

#### 4.4 Graph Traversal Strategies

**Eulerian Path** (if graph is Eulerian):
```python
def find_eulerian_path(G):
    """Find path that visits every edge exactly once"""
    if not is_eulerian_graph(G):
        return None
    
    # Use Hierholzer's algorithm
    path = []
    stack = [next(iter(G.nodes()))]
    
    while stack:
        current = stack[-1]
        if G.degree(current) == 0:
            path.append(stack.pop())
        else:
            next_node = next(iter(G.neighbors(current)))
            G.remove_edge(current, next_node)
            stack.append(next_node)
    
    return path
```

**Greedy Traversal** (for non-Eulerian graphs):
```python
def greedy_traversal(G, start_node=None):
    """Greedy traversal minimizing pen lifts"""
    if start_node is None:
        start_node = min(G.nodes(), key=lambda n: G.degree(n))
    
    path = []
    current = start_node
    visited_edges = set()
    
    while len(visited_edges) < G.number_of_edges():
        # Find unvisited edge from current node
        unvisited_neighbors = [
            n for n in G.neighbors(current)
            if (current, n) not in visited_edges and (n, current) not in visited_edges
        ]
        
        if unvisited_neighbors:
            next_node = unvisited_neighbors[0]
            edge = (current, next_node) if (current, next_node) in G.edges() else (next_node, current)
            visited_edges.add(edge)
            path.append(current)
            current = next_node
        else:
            # Need to lift pen - find nearest unvisited edge
            current = find_nearest_unvisited_edge(G, current, visited_edges)
    
    return path
```

### Data Structure
```python
class StrokeGraph:
    def __init__(self, skeleton):
        self.G = sknw.build_sknw(skeleton.astype('uint8'))
        self.nodes = list(self.G.nodes())
        self.edges = list(self.G.edges(data=True))
        self.endpoints = [n for n in self.nodes if self.G.degree(n) == 1]
        self.junctions = [n for n in self.nodes if self.G.degree(n) > 2]
    
    def get_edge_points(self, edge):
        """Get pixel coordinates for an edge"""
        return self.G[edge[0]][edge[1]]['pts']
```

---

## Step 5: Vectorization & Simplification

### Goal
Convert pixel sequences into compact polylines or smooth curves suitable for sampling and robot motion.

### Implementation Options

#### 5.1 Method Selection
- **RDP (Ramer-Douglas-Peucker)**: Fast, preserves corners
- **Potrace/Spline fitting**: Smooth Bézier curves, better for robot motion

#### 5.2 RDP Implementation
```python
from rdp import rdp

def simplify_polyline(pts, epsilon=2.0):
    """Simplify polyline using RDP algorithm"""
    if len(pts) < 3:
        return pts
    
    # Convert to float coordinates
    pts_float = pts.astype(np.float64)
    
    # Apply RDP simplification
    simplified = rdp(pts_float, epsilon=epsilon)
    
    return simplified
```

#### 5.3 Spline Fitting (Alternative)
```python
from scipy.interpolate import splprep, splev

def fit_spline(pts, smoothing_factor=0.1):
    """Fit smooth spline to points"""
    if len(pts) < 4:
        return pts
    
    # Fit parametric spline
    tck, u = splprep([pts[:, 0], pts[:, 1]], s=smoothing_factor, per=False)
    
    # Evaluate spline at more points for smoothness
    u_new = np.linspace(0, 1, len(pts) * 2)
    spline_pts = splev(u_new, tck)
    
    return np.column_stack(spline_pts)
```

#### 5.4 Parameter Tuning
```python
def adaptive_simplification(pts, max_error=1.0):
    """Adaptively choose simplification parameters"""
    # Start with conservative epsilon
    epsilon = 1.0
    
    while epsilon < 5.0:
        simplified = rdp(pts, epsilon=epsilon)
        error = calculate_approximation_error(pts, simplified)
        
        if error <= max_error:
            return simplified
        
        epsilon += 0.5
    
    return simplified
```

#### 5.5 Quality Metrics
```python
def calculate_approximation_error(original, simplified):
    """Calculate maximum deviation between original and simplified"""
    from scipy.spatial.distance import cdist
    
    distances = cdist(original, simplified)
    min_distances = np.min(distances, axis=1)
    
    return np.max(min_distances)
```

### Tradeoffs
- **RDP**: Faster computation, fewer control points, sharp corners
- **Splines**: Smoother trajectories, better for robot motion, more computation

---

## Step 6: Sampling & Scaling to Millimeters

### Goal
Produce continuous sampled points along each polyline with ~1mm spacing in real-world coordinates.

### Implementation Details

#### 6.1 Scale Calculation
```python
def calculate_scale_factor(image_width_px, canvas_width_mm):
    """Calculate mm per pixel scaling factor"""
    return canvas_width_mm / image_width_px
```

#### 6.2 Coordinate Conversion
```python
def convert_to_mm(pts_px, scale_mm_per_px):
    """Convert pixel coordinates to millimeters"""
    return pts_px * scale_mm_per_px
```

#### 6.3 Arc-Length Resampling
```python
def resample_polyline_mm(pts_px, scale_mm_per_px, spacing_mm=1.0):
    """Resample polyline with uniform spacing in mm"""
    # Convert to mm
    pts_mm = pts_px * scale_mm_per_px
    
    # Calculate cumulative distances
    deltas = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
    cum_distances = np.concatenate(([0], np.cumsum(deltas)))
    total_length = cum_distances[-1]
    
    # Generate sample points
    num_samples = int(total_length / spacing_mm) + 1
    sample_distances = np.linspace(0, total_length, num_samples)
    
    # Interpolate coordinates
    x_interp = np.interp(sample_distances, cum_distances, pts_mm[:, 0])
    y_interp = np.interp(sample_distances, cum_distances, pts_mm[:, 1])
    
    return np.column_stack([x_interp, y_interp])
```

#### 6.4 Edge Case Handling
```python
def handle_short_segments(pts_mm, min_length_mm=0.5):
    """Handle segments shorter than minimum length"""
    if len(pts_mm) < 2:
        return pts_mm
    
    total_length = np.sum(np.linalg.norm(np.diff(pts_mm, axis=0), axis=1))
    
    if total_length < min_length_mm:
        # Keep only start and end points
        return np.array([pts_mm[0], pts_mm[-1]])
    
    return pts_mm
```

#### 6.5 Closed Loop Handling
```python
def handle_closed_loops(pts_mm, is_closed=False):
    """Handle closed loops with proper wrapping"""
    if not is_closed or len(pts_mm) < 3:
        return pts_mm
    
    # Ensure loop closure
    if not np.allclose(pts_mm[0], pts_mm[-1]):
        pts_mm = np.vstack([pts_mm, pts_mm[0:1]])
    
    return pts_mm
```

### Quality Assurance
- Verify spacing is within tolerance (±0.1mm)
- Check for coordinate system consistency
- Validate total path length calculations

---

## Step 7: Color Detection - Warm vs Cool Classification

### Goal
For each mask/element, return a binary flag: 0 = warm, 1 = cool, based on perceptual color analysis.

### Implementation Strategy

#### 7.1 Color Space Selection
**CIELAB Color Space**:
- Separates chromatic axes: a* (green↔red), b* (blue↔yellow)
- Perceptually uniform
- Better for color temperature analysis than RGB/HSV

#### 7.2 Core Algorithm
```python
from skimage.color import rgb2lab
import numpy as np

def classify_mask_warmth(rgb_img, mask, a_thresh=5.0, b_thresh=5.0, 
                        warm_ratio_thresh=0.6, chroma_thresh=6.0):
    """
    Classify mask as warm (0) or cool (1) using LAB color space analysis
    
    Parameters:
    - rgb_img: RGB image (0-255 range)
    - mask: Binary mask for the region
    - a_thresh: Threshold for a* channel (green-red axis)
    - b_thresh: Threshold for b* channel (blue-yellow axis)
    - warm_ratio_thresh: Minimum ratio for warm classification
    - chroma_thresh: Minimum chroma for non-neutral classification
    """
    
    # Convert to LAB color space
    lab = rgb2lab(rgb_img / 255.0)  # skimage expects 0-1 range
    
    # Extract LAB values for masked pixels
    a_values = lab[..., 1][mask]
    b_values = lab[..., 2][mask]
    
    # Calculate chroma (saturation)
    chroma = np.sqrt(a_values**2 + b_values**2)
    
    # Per-pixel warm classification
    warm_votes = ((a_values > a_thresh) | (b_values > b_thresh))
    warm_ratio = np.sum(warm_votes) / len(a_values)
    
    # Handle low saturation (neutral/grayscale) regions
    median_chroma = np.median(chroma)
    if median_chroma < chroma_thresh:
        # Neutral region - default to cool (safer choice)
        return 1, {
            'warm_ratio': warm_ratio,
            'median_chroma': median_chroma,
            'classification': 'neutral->cool'
        }
    
    # Classify based on warm ratio
    warmth_class = 0 if warm_ratio >= warm_ratio_thresh else 1
    
    return warmth_class, {
        'warm_ratio': warm_ratio,
        'median_chroma': median_chroma,
        'classification': 'warm' if warmth_class == 0 else 'cool'
    }
```

#### 7.3 Alternative Hue-Based Method
```python
def classify_warmth_hsv(rgb_img, mask, warm_hue_range=(330, 60)):
    """
    Alternative classification using HSV hue analysis
    
    Parameters:
    - warm_hue_range: Tuple of (start_angle, end_angle) for warm colors
    """
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    hue_values = hsv[..., 0][mask]
    
    # Handle hue wraparound (red spans 330-360 and 0-60 degrees)
    start_hue, end_hue = warm_hue_range
    if start_hue > end_hue:  # Wraparound case
        warm_votes = ((hue_values >= start_hue) | (hue_values <= end_hue))
    else:
        warm_votes = ((hue_values >= start_hue) & (hue_values <= end_hue))
    
    warm_ratio = np.sum(warm_votes) / len(hue_values)
    
    return 0 if warm_ratio >= 0.6 else 1, {'warm_ratio': warm_ratio}
```

#### 7.4 Parameter Tuning Framework
```python
def tune_warmth_parameters(validation_set):
    """
    Tune classification parameters using validation data
    
    Parameters:
    - validation_set: List of (image, mask, ground_truth_warmth) tuples
    """
    best_params = None
    best_accuracy = 0
    
    # Parameter search space
    a_thresholds = [3.0, 5.0, 7.0, 10.0]
    b_thresholds = [3.0, 5.0, 7.0, 10.0]
    warm_ratios = [0.5, 0.6, 0.7, 0.8]
    chroma_thresholds = [4.0, 6.0, 8.0, 10.0]
    
    for a_thresh in a_thresholds:
        for b_thresh in b_thresholds:
            for warm_ratio in warm_ratios:
                for chroma_thresh in chroma_thresholds:
                    accuracy = evaluate_parameters(
                        validation_set, a_thresh, b_thresh, warm_ratio, chroma_thresh
                    )
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            'a_thresh': a_thresh,
                            'b_thresh': b_thresh,
                            'warm_ratio_thresh': warm_ratio,
                            'chroma_thresh': chroma_thresh
                        }
    
    return best_params, best_accuracy
```

#### 7.5 Quality Assurance
```python
def validate_warmth_classification(rgb_img, mask, min_pixels=200):
    """
    Validate classification quality and handle edge cases
    """
    pixel_count = np.sum(mask)
    
    if pixel_count < min_pixels:
        # Small mask - use neighbor or global classification
        return classify_small_mask(rgb_img, mask)
    
    # Check for extreme lighting conditions
    if detect_lighting_bias(rgb_img, mask):
        return apply_color_constancy(rgb_img, mask)
    
    return classify_mask_warmth(rgb_img, mask)
```

### Common Failure Modes and Mitigations
- **Mixed masks**: Voting logic handles mixtures but requires threshold tuning
- **Small masks**: Require minimum pixel count or fallback to neighbor classification
- **Lighting bias**: Implement color constancy or gray-world assumption
- **Low saturation**: Default to cool classification for neutral regions

---

## Step 8: Flattening & Stroke Ordering (Travel Optimization)

### Goal
Produce a single ordered array of strokes with coordinates, color temperature, and timing metadata for robotic orchestration.

### Data Structure Design
```python
class StrokeData:
    def __init__(self):
        self.meta = {
            'canvas_mm': [160, 160],  # Canvas dimensions in mm
            'scale_mm_per_px': 0.35,  # Scaling factor
            'total_strokes': 0,
            'total_length_mm': 0.0,
            'estimated_duration_s': 0.0
        }
        self.strokes = []
    
    def add_stroke(self, stroke_id, element_id, warmth, points, 
                   length_mm, duration_s, start_pos, end_pos):
        stroke = {
            'stroke_id': stroke_id,
            'element_id': element_id,
            'warmth': warmth,  # 0=warm, 1=cool
            'points': points,  # List of [x_mm, y_mm] coordinates
            'length_mm': length_mm,
            'duration_s': duration_s,
            'start_pos': start_pos,  # [x_mm, y_mm]
            'end_pos': end_pos,      # [x_mm, y_mm]
            'pen_lifts': 0,  # Number of pen lifts within stroke
            'curvature_avg': 0.0  # Average curvature for speed adjustment
        }
        self.strokes.append(stroke)
        self.meta['total_strokes'] += 1
        self.meta['total_length_mm'] += length_mm
        self.meta['estimated_duration_s'] += duration_s
```

### Stroke Ordering Strategies

#### 8.1 Within-Element Ordering
```python
def optimize_within_element_strokes(graph, start_node=None):
    """
    Optimize stroke order within a single element to minimize pen lifts
    """
    if is_eulerian_graph(graph):
        return find_eulerian_path(graph)
    else:
        return greedy_traversal_with_connectors(graph, start_node)

def greedy_traversal_with_connectors(graph, start_node=None):
    """
    Greedy traversal with intelligent connector placement
    """
    if start_node is None:
        start_node = find_optimal_start_node(graph)
    
    path = []
    current = start_node
    visited_edges = set()
    pen_lifts = 0
    
    while len(visited_edges) < graph.number_of_edges():
        unvisited_neighbors = find_unvisited_neighbors(graph, current, visited_edges)
        
        if unvisited_neighbors:
            next_node = unvisited_neighbors[0]
            edge = get_edge_tuple(graph, current, next_node)
            visited_edges.add(edge)
            path.append(current)
            current = next_node
        else:
            # Need to lift pen
            pen_lifts += 1
            current = find_nearest_unvisited_edge(graph, current, visited_edges)
    
    return path, pen_lifts
```

#### 8.2 Between-Element Ordering
```python
def optimize_stroke_sequence(stroke_data, strategy='travel_minimization'):
    """
    Optimize order of strokes across all elements
    
    Strategies:
    - 'salience_first': Draw most important elements first
    - 'travel_minimization': Minimize travel distance between strokes
    - 'hybrid': Combine both approaches
    """
    if strategy == 'salience_first':
        return order_by_salience(stroke_data)
    elif strategy == 'travel_minimization':
        return order_by_travel_optimization(stroke_data)
    else:
        return hybrid_ordering(stroke_data)

def order_by_travel_optimization(stroke_data):
    """
    Use TSP-like optimization to minimize travel distance
    """
    # Extract start positions of each stroke
    stroke_positions = [(i, stroke['start_pos']) for i, stroke in enumerate(stroke_data.strokes)]
    
    # Solve TSP using greedy nearest neighbor + 2-opt improvement
    ordered_indices = solve_tsp_greedy(stroke_positions)
    ordered_indices = improve_with_2opt(ordered_indices, stroke_positions)
    
    # Reorder strokes
    reordered_strokes = [stroke_data.strokes[i] for i in ordered_indices]
    stroke_data.strokes = reordered_strokes
    
    return stroke_data

def solve_tsp_greedy(stroke_positions):
    """
    Greedy TSP solution starting from canvas center
    """
    canvas_center = [80, 80]  # Center of 160x160mm canvas
    unvisited = set(range(len(stroke_positions)))
    path = []
    current_pos = canvas_center
    
    while unvisited:
        # Find nearest unvisited stroke
        nearest_idx = min(unvisited, 
                         key=lambda i: np.linalg.norm(
                             np.array(stroke_positions[i][1]) - np.array(current_pos)
                         ))
        
        path.append(nearest_idx)
        unvisited.remove(nearest_idx)
        current_pos = stroke_positions[nearest_idx][1]
    
    return path

def improve_with_2opt(path, stroke_positions):
    """
    Improve TSP solution using 2-opt local search
    """
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 1):
            for j in range(i + 1, len(path)):
                # Try 2-opt swap
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                if calculate_total_distance(new_path, stroke_positions) < \
                   calculate_total_distance(path, stroke_positions):
                    path = new_path
                    improved = True
                    break
            if improved:
                break
    
    return path
```

#### 8.3 Trajectory Segmentation and Timing
```python
def calculate_stroke_timing(points, draw_speed_mm_s=20.0, 
                          min_speed_mm_s=10.0, max_speed_mm_s=30.0):
    """
    Calculate timing for stroke execution based on curvature and length
    """
    if len(points) < 2:
        return 0.0, draw_speed_mm_s
    
    # Calculate total length
    total_length = 0.0
    for i in range(1, len(points)):
        total_length += np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
    
    # Calculate average curvature
    curvatures = []
    for i in range(1, len(points) - 1):
        p1, p2, p3 = points[i-1], points[i], points[i+1]
        curvature = calculate_curvature(p1, p2, p3)
        curvatures.append(curvature)
    
    avg_curvature = np.mean(curvatures) if curvatures else 0.0
    
    # Adjust speed based on curvature
    curvature_factor = 1.0 + (avg_curvature * 0.5)  # Higher curvature = slower
    adjusted_speed = draw_speed_mm_s / curvature_factor
    adjusted_speed = np.clip(adjusted_speed, min_speed_mm_s, max_speed_mm_s)
    
    duration = total_length / adjusted_speed
    
    return duration, adjusted_speed

def calculate_curvature(p1, p2, p3):
    """
    Calculate curvature at point p2 given three consecutive points
    """
    # Vector from p1 to p2 and p2 to p3
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    
    # Cross product magnitude (proportional to curvature)
    cross_product = abs(v1[0] * v2[1] - v1[1] * v2[0])
    
    # Normalize by vector magnitudes
    norm_factor = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm_factor == 0:
        return 0.0
    
    return cross_product / norm_factor
```

#### 8.4 Pen Lift Strategy
```python
def plan_pen_lifts(stroke_data, lift_height_mm=15.0, 
                   min_travel_distance_mm=5.0):
    """
    Plan pen lifts for safe travel between strokes
    """
    for i in range(1, len(stroke_data.strokes)):
        current_stroke = stroke_data.strokes[i-1]
        next_stroke = stroke_data.strokes[i]
        
        # Calculate travel distance
        travel_distance = np.linalg.norm(
            np.array(next_stroke['start_pos']) - np.array(current_stroke['end_pos'])
        )
        
        if travel_distance > min_travel_distance_mm:
            # Add pen lift instruction
            lift_instruction = {
                'type': 'pen_lift',
                'height_mm': lift_height_mm,
                'travel_distance_mm': travel_distance,
                'from_stroke': current_stroke['stroke_id'],
                'to_stroke': next_stroke['stroke_id']
            }
            
            # Insert lift instruction (implementation depends on data structure)
            insert_lift_instruction(stroke_data, i, lift_instruction)
```

### Output Format
```python
def export_stroke_data(stroke_data, format='json'):
    """
    Export stroke data in specified format
    """
    if format == 'json':
        return json.dumps({
            'meta': stroke_data.meta,
            'strokes': stroke_data.strokes
        }, indent=2)
    
    elif format == 'gcode':
        return generate_gcode(stroke_data)
    
    elif format == 'csv':
        return generate_csv(stroke_data)
    
    else:
        raise ValueError(f"Unsupported format: {format}")

def generate_gcode(stroke_data):
    """
    Generate G-code for robotic execution
    """
    gcode_lines = [
        "G21",  # Set units to millimeters
        "G90",  # Absolute positioning
        "G0 Z15",  # Lift pen
    ]
    
    for stroke in stroke_data.strokes:
        # Move to start position
        start_x, start_y = stroke['start_pos']
        gcode_lines.append(f"G0 X{start_x:.2f} Y{start_y:.2f}")
        gcode_lines.append("G0 Z0")  # Lower pen
        
        # Draw stroke
        for point in stroke['points']:
            x, y = point
            gcode_lines.append(f"G1 X{x:.2f} Y{y:.2f} F{stroke.get('speed', 20)}")
        
        gcode_lines.append("G0 Z15")  # Lift pen
    
    return "\n".join(gcode_lines)
```

---

## Implementation Requirements

### Dependencies
```python
# Core computer vision
opencv-python>=4.8.0
scikit-image>=0.21.0
numpy>=1.24.0
scipy>=1.11.0

# Segmentation
segment-anything>=1.0
torch>=2.0.0
torchvision>=0.15.0

# Graph processing
networkx>=3.1
sknw>=0.11

# Vectorization
rdp>=0.8
shapely>=2.0.0

# Color analysis
scikit-image>=0.21.0  # For rgb2lab

# Optimization
scikit-learn>=1.3.0  # For clustering if needed
```

### Performance Considerations
- **GPU acceleration**: Use CUDA for SAM and HED models when available
- **Memory management**: Process large images in tiles if memory is limited
- **Parallel processing**: Process multiple masks simultaneously
- **Caching**: Cache intermediate results (skeletons, graphs) for iterative development

### Quality Metrics
- **Segmentation quality**: IoU with ground truth masks
- **Edge detection accuracy**: F1 score on edge pixels
- **Vectorization precision**: Hausdorff distance between original and simplified curves
- **Color classification accuracy**: Accuracy on labeled warm/cool regions
- **Travel optimization**: Total travel distance reduction percentage

### Testing Strategy
- **Unit tests**: Individual pipeline components
- **Integration tests**: End-to-end pipeline with sample images
- **Performance tests**: Processing time and memory usage
- **Quality tests**: Output validation against ground truth data

---

## Conclusion

This specification provides a comprehensive framework for implementing a painting vectorization pipeline. The modular design allows for iterative development and optimization of individual components. Key success factors include:

1. **Robust preprocessing** to handle diverse input images
2. **Effective segmentation** to isolate meaningful elements
3. **High-quality edge extraction** preserving stroke characteristics
4. **Intelligent graph traversal** minimizing pen lifts
5. **Accurate color classification** for dual-pen systems
6. **Optimized stroke ordering** for efficient robotic execution

The pipeline is designed to be configurable and extensible, allowing for adaptation to different artistic styles and robotic systems.
