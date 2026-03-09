# Math & Strokes — Deep Dive

This document explains every mathematical operation in the pipeline and provides a thorough description of what a "stroke" is at each stage of its life, from raw image pixels all the way to a physical mark on canvas.

---

## Table of Contents

1. [What Is a Stroke?](#1-what-is-a-stroke)
2. [Step 1 — Preprocessing Math](#2-step-1--preprocessing-math)
3. [Step 2 — Segmentation Scoring Math](#3-step-2--segmentation-scoring-math)
4. [Step 3 — Edge Extraction Math](#4-step-3--edge-extraction-math)
5. [Step 4 — Stroke Graph Math](#5-step-4--stroke-graph-math)
6. [Step 5 — Vectorization Math (Ramer-Douglas-Peucker)](#6-step-5--vectorization-math-ramerdouglaspeucker)
7. [Step 6 — Sampling & Coordinate Scaling Math](#7-step-6--sampling--coordinate-scaling-math)
8. [Step 7 — Color Detection Math](#8-step-7--color-detection-math)
9. [Step 8 — Stroke Ordering Math (TSP)](#9-step-8--stroke-ordering-math-tsp)
10. [Firmware — Delta Robot Inverse Kinematics](#10-firmware--delta-robot-inverse-kinematics)

---

## 1. What Is a Stroke?

A **stroke** is the fundamental unit of robot motion. Understanding what it means at each stage is key to understanding the whole system.

### Stage-by-stage stroke identity

| Stage | What the stroke IS | Data type |
|-------|--------------------|-----------|
| After Step 3 | A contour path — a sequence of pixel coordinates forming the boundary of one region | `numpy.ndarray` of `(x_px, y_px)` |
| After Step 4 | The same path, reordered to start at the topmost-leftmost point and optionally closed | `numpy.ndarray`, closed loop with first point appended at end |
| After Step 5 | A **simplified** polyline — far fewer points but the same shape within ε=1 px | `numpy.ndarray` of `float32 (x_px, y_px)` |
| After Step 6 | A list of **millimetre-space** waypoints at uniform ~1 mm intervals | `numpy.ndarray` of `(x_mm, y_mm)` |
| After Step 7 | Same waypoints, plus a `warmth` label: `0` (warm) or `1` (cool) | Same array + metadata dict |
| After Step 8 | An entry in the final JSON with `stroke_id`, `element_id`, `warmth`, `points`, `start_pos`, `end_pos`, `length_mm` | Python dict → JSON |
| During execution | A sequence of `command(x, y)` calls to the delta robot, with pen lowered at the start | Robot motor positions |

### What separates strokes: pen lifts

Between two consecutive strokes the robot raises the pen to `pen_lift_height_mm = 15 mm`, moves to the start of the next stroke, then lowers the pen again. A pen lift is inserted whenever the gap between the end of one stroke and the start of the next is greater than `min_travel_distance_mm = 2 mm`.

### Stroke phases

Within each segmented region, strokes are categorized semantically:

- **boundary** — outline/contour strokes that define the edge of the region (generated in outline-only mode; these are the primary strokes)
- **internal** — fill strokes inside a region (traditional mode only)
- **detail** — fine texture strokes (traditional mode only)

In the demo's outline-only mode, every stroke is a `boundary` stroke.

---

## 2. Step 1 — Preprocessing Math

**File:** `painting_vectorization/step1_preprocessing/image_preprocessing.py`

### 2.1 Resize with aspect ratio preservation

Given an image of original dimensions `(H, W)` and a maximum allowed long edge of `max_size = 1024 px`:

```
scale_factor = max_size / max(H, W)

new_H = round(H × scale_factor)
new_W = round(W × scale_factor)
```

If the image is already smaller than 1024 px on both sides, `scale_factor = 1` and no resize happens.

### 2.2 CLAHE — Contrast Limited Adaptive Histogram Equalization

Standard histogram equalization spreads all pixel intensities uniformly across the full range. This over-amplifies noise in low-contrast regions. CLAHE solves this by:

1. Dividing the image into a grid of small tiles (typically 8×8 pixels each).
2. Computing a histogram of intensities within each tile.
3. **Clipping** the histogram at a `clip_limit` (e.g., 2.0) — any bin exceeding the clip limit has its excess redistributed uniformly across all other bins. This prevents any single intensity from dominating.
4. Applying the clipped, equalized mapping to that tile.
5. Using **bilinear interpolation** at tile boundaries to avoid block artifacts.

The result is that local contrast is enhanced (edges become sharper) without blowing out already high-contrast regions. The code uses OpenCV's implementation:

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray_channel)
```

### 2.3 Bilateral filter

The bilateral filter smooths an image while preserving edges. For each pixel `p` with intensity `I(p)`, its new value is:

```
I'(p) = (1/W_p) × Σ_q  I(q) × f_s(‖p - q‖) × f_r(|I(p) - I(q)|)
```

where:
- `f_s` is a Gaussian in **spatial** distance — pixels far away contribute less.
- `f_r` is a Gaussian in **range** (intensity difference) — pixels with very different intensity (i.e., across an edge) contribute very little.
- `W_p` is the sum of all weights (normalization factor).

With parameters `d=9` (kernel diameter), `sigmaColor=75`, `sigmaSpace=75`:

```
f_s(d) = exp(−d² / (2 × 75²))
f_r(ΔI) = exp(−ΔI² / (2 × 75²))
```

The two Gaussians together mean: "blend with neighbours only if they are **spatially close** AND **similarly coloured**." This removes texture noise inside uniform regions without blurring the edges between them.

---

## 3. Step 2 — Segmentation Scoring Math

**File:** `painting_vectorization/step2_segmentation/segmentation.py`

SAM generates many candidate masks. The system scores each one to pick the most meaningful ~10–20.

### 3.1 Intersection over Union (IoU)

Used to decide whether two masks are essentially duplicates:

```
IoU(A, B) = |A ∩ B| / |A ∪ B|
```

`|A ∩ B|` = number of pixels that are `True` in both masks.  
`|A ∪ B|` = number of pixels that are `True` in at least one mask.

If `IoU > 0.6`, the two masks are merged with a logical OR (union of their pixels). This prevents drawing the same region twice.

### 3.2 Mask salience score

Each mask is scored on three criteria that are then multiplied together:

```
salience = area_score × edge_strength × color_distinctiveness
```

**Area score** — fraction of the image covered by the mask:
```
area_score = number_of_mask_pixels / (image_height × image_width)
```
Larger, more prominent elements score higher.

**Edge strength** — mean Canny edge value within the mask:
```
edge_strength = mean(edge_map[mask])
```
Regions that follow real visual boundaries score higher.

**Color distinctiveness** — how different the mask's colors are from the rest of the image:
```
color_distinctiveness = std(pixels inside mask) / (std(pixels outside mask) + ε)
```
where `ε = 1e-6` prevents division by zero. A mask full of a unique color that doesn't appear elsewhere scores high.

Masks are sorted by `salience` (highest first) and only the top ~10–20 are kept.

---

## 4. Step 3 — Edge Extraction Math

**File:** `painting_vectorization/step3_edge_extraction/edge_extraction.py`

### 4.1 Gaussian blur (pre-Canny smoothing)

The image is blurred with a 7×7 Gaussian kernel with σ=2.0 before Canny. The 2D Gaussian kernel is:

```
G(x, y) = (1 / (2π σ²)) × exp(−(x² + y²) / (2σ²))
```

Smoothing removes high-frequency noise so that Canny doesn't detect false edges from texture or film grain.

### 4.2 Canny edge detection

Canny is a multi-stage algorithm:

**Stage 1 — Gradient computation (Sobel operator)**

The image gradient is computed at each pixel using the Sobel operator. The x and y derivatives are:

```
Gx = [−1  0  +1]       Gy = [−1  −2  −1]
     [−2  0  +2]  ×I       [ 0   0   0] ×I
     [−1  0  +1]            [+1  +2  +1]
```

The gradient magnitude and direction at each pixel are:

```
|G| = sqrt(Gx² + Gy²)
θ   = arctan2(Gy, Gx)
```

**Stage 2 — Non-maximum suppression**

For each pixel, compare its gradient magnitude to the two neighbours along the gradient direction `θ`. If the pixel is not a local maximum (i.e., at least one neighbour in the gradient direction has a larger magnitude), suppress it to zero. This thins edges to 1-pixel width.

**Stage 3 — Double thresholding**

With `low_thresh = 80, high_thresh = 160` (outline mode) or `100, 200` (boundary mode):
- Pixels with `|G| > high_thresh` → **strong edge** (keep)
- Pixels with `low_thresh < |G| ≤ high_thresh` → **weak edge** (keep only if connected to a strong edge)
- Pixels with `|G| ≤ low_thresh` → **not an edge** (discard)

**Stage 4 — Hysteresis edge tracking**

Walk each strong edge pixel; any connected weak edge pixel is promoted to a strong edge. This fills gaps in curves and removes isolated weak noise.

### 4.3 Contour approximation (Douglas-Peucker — OpenCV)

After `cv2.findContours`, each contour is simplified:

```python
epsilon = 0.005 × cv2.arcLength(contour, closed=True)
approx = cv2.approxPolyDP(contour, epsilon, closed=True)
```

`arcLength` computes the perimeter of the contour. The tolerance `ε = 0.5%` of the perimeter means points within 0.5% of the perimeter length from the simplified line are removed. This is the same Douglas-Peucker algorithm described in Step 5 but applied early for contour cleanup.

### 4.4 Morphological gradient

Used as a third edge-detection method alongside Canny:

```
gradient = dilate(I, K) − erode(I, K)
```

where `K` is a 5×5 elliptical structuring element. The dilation expands bright regions; the erosion shrinks them. Their difference highlights the **boundary pixels** of every region. A binary threshold at value 30 converts the gradient to a binary edge map.

### 4.5 Skeletonization (Zhang-Suen algorithm)

In traditional mode, the 1-pixel-wide binary edge map is thinned to a skeleton using the iterative Zhang-Suen algorithm (`skimage.morphology.skeletonize`). In each iteration:

1. Mark pixels for deletion if they are **boundary pixels** (have at least one background neighbour) and meet specific connectivity/transition conditions.
2. Apply the deletions in two alternating passes (north-south pass, east-west pass).
3. Repeat until no more pixels are deleted.

The result is that every thick edge region is reduced to a single-pixel-wide centreline (skeleton).

### 4.6 Probabilistic Hough transform (structural line detection)

```
ρ = x cos θ + y sin θ
```

Each edge pixel votes for all `(ρ, θ)` pairs that satisfy this equation. Lines are found where votes accumulate in `(ρ, θ)` space. The probabilistic variant (`HoughLinesP`) works on random subsets of edge pixels and returns line segments rather than infinite lines, parameterized by `(x1, y1, x2, y2)`.

The line angle is computed as:

```
angle = arctan2(y2 − y1, x2 − x1) × (180 / π)
```

Lines within 15° of horizontal (0°/180°), vertical (90°), or diagonal (45°/135°) are kept as structural lines.

---

## 5. Step 4 — Stroke Graph Math

**File:** `painting_vectorization/step4_stroke_graph/stroke_graph.py`

### 5.1 Optimal start point selection

For each closed contour path (array of points `P[0..N-1]`), the best starting point is chosen as:

```
min_y = min(P[i][1])          // topmost row
candidates = {i : P[i][1] == min_y}
start = argmin_{i ∈ candidates}(P[i][0])   // leftmost among topmost
```

This gives a deterministic, human-intuitive starting point (top-left of the shape).

### 5.2 Contour reordering and closure

The path is rotated so that `start_idx` becomes index 0:

```
reordered = P[start_idx:] ++ P[:start_idx]
closed    = reordered ++ [reordered[0]]     // append first point to close loop
```

### 5.3 Graph topology (traditional mode, via sknw)

`sknw.build_sknw(skeleton_image)` converts the binary skeleton into a NetworkX graph where:

- **Nodes** are pixels with **0** or **1** neighbours (endpoints) or **≥3** neighbours (junction points).
- **Edges** carry the pixel sequence between two nodes as an ordered list of coordinates.

A pixel has a neighbour if any of its 8-connected surrounding pixels (horizontal, vertical, or diagonal) is also part of the skeleton.

### 5.4 Eulerian path detection

An **Eulerian path** (a path that visits every edge exactly once) exists in an undirected graph if and only if the graph has **exactly 0 or 2 vertices of odd degree**. A vertex's degree is the number of edges connected to it.

If the skeleton graph has 0 odd-degree vertices → Eulerian **circuit** (closed loop, no pen lifts needed).  
If it has 2 odd-degree vertices → Eulerian **path** (start at one, end at the other).  
If it has more → multiple separate pen lifts are required.

The stroke graph step detects this condition and uses it to minimise pen lifts.

---

## 6. Step 5 — Vectorization Math (Ramer-Douglas-Peucker)

**File:** `painting_vectorization/step5_vectorization/vectorization.py`

### 6.1 Perpendicular distance from a point to a line segment

Given a point `P` and a line from `A` to `B`, the perpendicular distance is computed using the **cross product**:

```
d = |cross(B − A, A − P)| / ‖B − A‖
```

In 2D, the cross product of vectors `u = (ux, uy)` and `v = (vx, vy)` gives a scalar:

```
cross(u, v) = ux × vy − uy × vx
```

So:

```
u = B − A = (Bx − Ax, By − Ay)
v = A − P = (Ax − Px, Ay − Py)

|cross(u, v)| = |(Bx − Ax)(Ay − Py) − (By − Ay)(Ax − Px)|
‖B − A‖       = sqrt((Bx − Ax)² + (By − Ay)²)

d = |cross(u, v)| / ‖B − A‖
```

This is the implementation in `point_line_distance()`.

### 6.2 The RDP recursive algorithm

Given a polyline `P[0..N]` and tolerance `ε`:

```
function RDP(points, ε):
    if len(points) ≤ 2:
        return points

    // Find the point with maximum perpendicular distance from line P[0]→P[N]
    d_max = 0
    index = 0
    for i in 1 .. N-1:
        d = point_line_distance(P[i], P[0], P[N])
        if d > d_max:
            d_max = d
            index = i

    if d_max > ε:
        // Split and recurse
        left  = RDP(P[0..index], ε)
        right = RDP(P[index..N], ε)
        return left[:-1] + right        // merge, avoiding duplicate middle point

    else:
        // All intermediate points are within ε of the line → discard them
        return [P[0], P[N]]
```

**What this achieves:** Starting from a polyline with potentially thousands of dense skeleton pixels, RDP recursively discards points that are within `ε = 1.0 px` of the straight line between their neighbours. The result is a minimal polyline that stays within 1 pixel of the original path.

**Compression example:** A circular arc sampled at every pixel might have 500 points. After RDP with ε=1 px it typically reduces to ~20–50 points (the number needed to keep the maximum deviation under 1 px).

---

## 7. Step 6 — Sampling & Coordinate Scaling Math

**File:** `painting_vectorization/step6_sampling/sampling.py`

### 7.1 Scale factor calculation

The physical canvas is 130 mm × 130 mm (set in `constants.py`). The image has been resized to at most 1024 px on its long edge. The scale factor converts pixels to millimetres:

```
scale_mm_per_px = CANVAS_SIZE_MM / image_width_px
                = 130 / image_width_px
```

For a 1024 px wide image: `scale = 130 / 1024 ≈ 0.127 mm/px`.

### 7.2 Pixel → millimetre coordinate conversion

For points in standard `(x, y)` format (produced by the outline contour path):

```
x_mm = (x_px + x_offset) × scale_mm_per_px
y_mm = (y_px + y_offset) × scale_mm_per_px
```

`(x_offset, y_offset)` is the top-left corner of the mask's bounding box in the full image. Adding the offset maps crop-relative coordinates back to full-image coordinates before scaling.

The coordinate origin `(0, 0)` is at the **top-left** of the canvas with `+x` pointing right and `+y` pointing down (same as standard image convention).

For points produced by `sknw` in `(row, col)` = `(y, x)` format, the columns are swapped before scaling:

```
x_mm = (col + x_offset) × scale_mm_per_px   // col = x
y_mm = (row + y_offset) × scale_mm_per_px   // row = y
```

### 7.3 Arc-length parameterization and uniform resampling

Given a polyline with millimetre-space vertices `Q[0], Q[1], …, Q[M]`:

**Step 1 — Compute cumulative arc length:**
```
Δ[i] = ‖Q[i+1] − Q[i]‖   for i = 0..M-1

cum[0]   = 0
cum[i+1] = cum[i] + Δ[i]

total_length = cum[M]
```

**Step 2 — Generate uniform sample positions:**
```
n_samples = max(2, floor(total_length / spacing_mm) + 1)

t[j] = j × total_length / (n_samples − 1)   for j = 0..n_samples-1
```

These `t[j]` values are distances along the arc at which to place sample points, evenly spaced by approximately `spacing_mm = 1.0 mm`.

**Step 3 — Interpolate x and y at each sample position:**
```
x[j] = interp(t[j], cum[0..M], Q[:, 0])   // linear interpolation
y[j] = interp(t[j], cum[0..M], Q[:, 1])
```

`interp(t, xp, fp)` finds the two `xp` values bracketing `t` and linearly interpolates `fp` between them:

```
// if cum[k] ≤ t < cum[k+1]:
fraction = (t − cum[k]) / (cum[k+1] − cum[k])
x[j]     = Q[k][0] + fraction × (Q[k+1][0] − Q[k][0])
```

**Why this matters:** The RDP-simplified polyline has non-uniform spacing between vertices (long straight segments have only their endpoints; tight curves have more points). Arc-length resampling gives the robot **equal-distance waypoints**, which translates to consistent pen speed and uniform line quality.

### 7.4 Closed loop detection

A stroke is treated as a closed loop if:

```
‖Q[0] − Q[M]‖ ≤ closure_threshold_mm   (default: 2.0 mm)
```

If closed, the first point is appended at the end to ensure the robot returns to the exact starting position:

```
Q_closed = [Q[0], Q[1], ..., Q[M], Q[0]]
```

---

## 8. Step 7 — Color Detection Math

**File:** `painting_vectorization/step7_color_detection/color_detection.py`

### 8.1 CIELAB color space

RGB is converted to **CIELAB** (L\*, a\*, b\*) using `skimage.color.rgb2lab`. CIELAB is perceptually uniform — equal numeric distances correspond to roughly equal perceived color differences.

The three channels are:
- **L\*** — Lightness (0 = black, 100 = white). Irrelevant to warm/cool classification.
- **a\*** — Green–Red axis. Negative values → green; positive values → red/magenta.
- **b\*** — Blue–Yellow axis. Negative values → blue; positive values → yellow.

### 8.2 Per-pixel warm/cool vote

For each pixel within the mask:

```
is_warm = (a* > a_thresh) OR (b* > b_thresh)
         = (a* > 5.0)   OR  (b* > 5.0)
```

A pixel is warm if it leans toward red (`a* > 5`) or toward yellow (`b* > 5`). Cool pixels are blue/cyan/green (both axes at or below the threshold).

### 8.3 Warm ratio

```
warm_ratio = count(is_warm) / total_pixels_in_mask
```

A mask is classified as **warm (0)** if `warm_ratio ≥ warm_ratio_thresh = 0.6` (at least 60% of its pixels are warm-leaning).

### 8.4 Chroma (colorfulness measure)

```
C* = sqrt(a*² + b*²)
```

`C*` is the distance from the neutral grey axis in the `(a*, b*)` plane. Low chroma means the region is near grey/white/black (achromatic). A region with `median(C*) < chroma_thresh = 6.0` is classified as **neutral**, and by default treated as **cool** (safer assumption for most artwork).

### 8.5 HSV hue method (fallback when scikit-image is unavailable)

In the HSV color model, hue `H` ranges from 0° to 360°. Warm hues span from 330° (red-purple end) through 0° (red) to 60° (yellow), wrapping around the 0°/360° boundary:

```
// Wraparound check for red hues:
is_warm = (H ≥ 330°) OR (H ≤ 60°)
```

Pixels with `saturation < 0.2` (near-grey) are excluded from the vote since their hue is unreliable.

---

## 9. Step 8 — Stroke Ordering Math (TSP)

**File:** `painting_vectorization/step8_stroke_ordering/stroke_ordering.py`

The goal is to sequence all strokes so that total pen-up travel distance (the distance the robot moves with the pen raised between strokes) is minimized. This is an instance of the **Travelling Salesman Problem (TSP)** — an NP-hard optimization problem. The system uses fast heuristics.

### 9.1 Greedy nearest-neighbor algorithm

Starting from the canvas center `(65 mm, 65 mm)`:

```
current_pos = center_of_canvas
unvisited   = {all stroke indices}
path        = []

while unvisited is not empty:
    nearest = argmin_{i ∈ unvisited}(‖start_pos[i] − current_pos‖)
    path.append(nearest)
    current_pos = start_pos[nearest]
    unvisited.remove(nearest)
```

The distance is the standard **Euclidean distance** in mm:

```
‖A − B‖ = sqrt((Ax − Bx)² + (Ay − By)²)
```

This greedy approach runs in O(N²) time. For N ≤ 100 strokes it is run exactly. For N > 100, strokes are sorted by their diagonal position instead (`x + y`), which is an O(N log N) spatial approximation.

### 9.2 2-opt local search improvement

After greedy nearest-neighbor, the path can be improved with **2-opt**. The key idea is that crossing paths are always sub-optimal — uncrossing them reduces total distance.

For each pair of edges `(i, i+1)` and `(j, j+1)` in the path, try reversing the sub-path between them:

```
new_path = path[0..i] + reversed(path[i..j]) + path[j+1..N]
```

Accept the swap if:

```
total_distance(new_path) < total_distance(old_path)
```

Repeat until no improving swap is found (or after 50 iterations to cap compute time). In the current implementation 2-opt is defined but not called in the hot path — the greedy solution is used directly for speed.

### 9.3 Pen lift calculation

After ordering, pen lifts are inserted between consecutive strokes `i` and `i+1` if:

```
travel_distance = ‖start_pos[i+1] − end_pos[i]‖ > min_travel_distance_mm
                 = ‖start_pos[i+1] − end_pos[i]‖ > 2.0 mm
```

Travel duration is estimated at 50 mm/s travel speed:

```
duration_s = travel_distance / 50.0
```

### 9.4 Hybrid strategy: semantic ordering + spatial optimization

The hybrid strategy (used by default) splits strokes into three groups by semantic phase: `boundary`, `internal`, `detail`. These are processed in order — boundary strokes are drawn before internal fill, which is drawn before fine details. Within each group, the greedy nearest-neighbor algorithm is applied independently. This preserves artistic intent (draw outlines first, then fill, then detail) while still minimizing travel within each phase.

---

## 10. Firmware — Delta Robot Inverse Kinematics

**Files:** `firmware/driver.py`, `firmware/prodplotter.py`

### 10.1 Robot geometry

The robot is a **2-DOF planar delta** — two vertical rails separated by distance `D = 210 mm`. Each rail has a freely-sliding carriage at vertical position `y_L` (left) and `y_R` (right). A rigid rod of length `L = 210 mm` connects each carriage to a shared **end effector** (the pen tip) at position `(x, y)` in the robot frame.

```
Left rail:  x = −D/2 = −105 mm
Right rail: x = +D/2 = +105 mm
```

### 10.2 Inverse kinematics (IK)

For the end effector to be at position `(x, y)`, both rods must have length exactly `L`. This gives one constraint per rod:

```
(x − x_left )² + (y − y_L)² = L²       // left rod
(x − x_right)² + (y − y_R)² = L²       // right rod
```

Solving for `y_L` (left slider position):

```
(y − y_L)² = L² − (x − x_left)²
           = L² − (x + D/2)²

y_L = y ± sqrt(L² − (x + D/2)²)
```

The sign determines whether the slider is above or below the end effector. The system uses `prefer_above = True`, so:

```
y_L = y + sqrt(L² − (x + D/2)²)   // slider above the effector
y_R = y + sqrt(L² − (x − D/2)²)   // same for right rod
```

**Feasibility condition:** The expression under the square root must be non-negative, which requires:

```
|x − x_side| ≤ L
```

i.e., the horizontal distance from the end effector to either rail must not exceed the rod length.

### 10.3 Coordinate frame transformation (top-left → robot-centered)

The JSON stroke data uses a **top-left origin** system where `(0, 0)` is the top-left corner of the canvas and `+y` points down. The robot uses a **center origin** system where `(0, 0)` is the canvas center and `+y` points up.

`prodplotter.py` performs this transform:

```
HALF = CMD_SIZE / 2 = 75 mm

x_centered = x_tl − HALF              // shift x origin to center
y_centered = HALF − y_tl              // shift and flip y axis

x_robot = x_centered                  // no additional x offset
y_robot = y_centered + OFFSET_Y       // apply -20 mm effective center correction
        = (HALF − y_tl) − 20
```

The `OFFSET_Y = −20 mm` correction accounts for the physical offset between the geometric center of the robot's workspace and the actual center of the drawing canvas.

### 10.4 Rail command inversion

The IK computes `(y_L, y_R)` as distances measured with the same orientation as the robot frame (+y up). The GRBL motor controller uses a different convention (positive = downward travel). The code maps between them:

```
left_cmd  = 270.0 − y_R    // left motor commands from right IK result
right_cmd = 270.0 − y_L    // right motor commands from left IK result
```

The `270.0 − y` flips the direction and the `L↔R` swap corrects for a physical wiring inversion between the left/right rail assignments.

### 10.5 Reachability check

Before sending any IK command the system checks:
1. The target is within the **top-left command space**: `0 ≤ x_tl ≤ 150`, `0 ≤ y_tl ≤ 150`.
2. The robot-frame target is within the **robot workspace**: `x_robot ∈ [−80, 80]`, `y_robot ∈ [−220, 100]`.
3. The computed slider positions are within **rail travel limits**: `y_L, y_R ∈ [0, 270]`.

If any check fails the move is skipped and a warning is logged.

---

## Summary: Math at a glance

| Step | Key math | Formula |
|------|----------|---------|
| Preprocessing | CLAHE contrast | Clip histogram per tile, redistribute excess |
| Preprocessing | Bilateral filter | Weighted blend with spatial + range Gaussians |
| Segmentation | IoU | `|A∩B| / |A∪B|` |
| Segmentation | Salience | `area × edge_strength × color_distinctiveness` |
| Edge extraction | Canny gradient | `\|G\| = sqrt(Gx² + Gy²)`, `θ = arctan2(Gy, Gx)` |
| Edge extraction | Morphological gradient | `dilate(I,K) − erode(I,K)` |
| Edge extraction | Hough line angle | `θ = arctan2(dy, dx)` |
| Vectorization | RDP perpendicular distance | `d = |cross(B−A, A−P)| / ‖B−A‖` |
| Sampling | Scale factor | `s = canvas_mm / image_px` |
| Sampling | Arc-length resampling | Cumulative `Δ = Σ‖Q[i+1]−Q[i]‖`, then linear `interp` |
| Color detection | Chroma | `C* = sqrt(a*² + b*²)` |
| Color detection | Warm vote | `(a* > 5) OR (b* > 5)` |
| Color detection | Hue wraparound | `(H ≥ 330°) OR (H ≤ 60°)` |
| Stroke ordering | Euclidean distance | `d = sqrt((Δx)² + (Δy)²)` |
| Stroke ordering | Nearest-neighbor TSP | `next = argmin_i(‖pos_i − current‖)` |
| Robot IK | Constraint equation | `(x−x_rail)² + (y−y_slider)² = L²` |
| Robot IK | Slider position | `y_slider = y ± sqrt(L² − (x−x_rail)²)` |
| Robot IK | Frame transform | `x_c = x_tl − 75`, `y_c = 75 − y_tl − 20` |
