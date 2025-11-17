# Image Detection Documentation

## Overview

The image detection module (`image_detection.py`) identifies photo/image regions in screenshots by analyzing edge patterns. Photos have **mixed edge angles** (diagonal, curved) while UI elements have mostly **horizontal/vertical edges**.

## Processing Pipeline

### Step 1: Text Removal
**Purpose:** Remove text regions to avoid detecting them as images.

- Samples border pixels of each OCR box to determine background color
- Fills text boxes with background color
- Output: Clean grayscale image without text

### Step 2: Edge Detection & Angle Filtering
**Purpose:** Find characteristic "photo" edges (non-axis-aligned).

1. Compute gradient magnitude (edge strength) using Scharr operator
2. Compute gradient angle (0-360°) for each edge pixel
3. Threshold to keep top 15% strongest edges
4. **Filter OUT axis-aligned edges** (horizontal/vertical ±15°)
5. Keep only **sparse diagonal/curved edges** characteristic of photos

**Output:** 
- `edges_all`: All strong edges
- `edges_sparse`: Only photo-like (non-axis-aligned) edges

### Step 3: Edge Clustering
**Purpose:** Group nearby sparse edges into candidate image regions.

Uses **morphological dilation** to connect nearby edge pixels:

#### Dilation Parameters

- **`kernel_size`** (k): Size of structuring element (3×3, 5×5, 7×7)
  - Larger kernel = connects edges further apart
  - Smaller kernel = tighter, more conservative clusters
  
- **`iterations`** (i): Number of times to apply dilation
  - `i=1`: Apply dilation once
  - `i=2`: Apply dilation twice (grows connections more)
  - More iterations = larger clusters, fewer components
  - Fewer iterations = tighter clusters, more components

The code tries 6 combinations: k={3,5,7} × i={1,2}

**Currently using:** `k=3, i=1` (most conservative - tight clusters)

**Output:** Connected components (clusters) with statistics

### Step 4: Content Map Generation
**Purpose:** Create contrast-invariant image for box growing.

Uses **z-normalization** (local contrast normalization):
1. CLAHE enhancement to handle varying brightness
2. Gaussian blur to get local mean and variance
3. Compute z-score: `(pixel - local_mean) / local_std`
4. Apply tanh compression for robustness

**Output:** Image where content boundaries are clear regardless of brightness/contrast

### Step 5: Box Fitting & Growing
**Purpose:** Fit initial boxes to clusters and grow them to full extent.

For each valid cluster (meets area and edge density thresholds):

1. **Shape Decision:**
   - If aspect ratio ≈ 1.0 (square-ish): Try circular fit
   - Otherwise: Use rectangular bounding box

2. **Circular Fit** (for square-ish clusters):
   - Fit minimum enclosing circle to edge points
   - Convert circle to square bounding box
   - Useful for profile pictures, buttons, icons

3. **Box Growing:**
   - Measure initial content score (std dev of z-normalized region)
   - Grow in each direction (right, down, left, up) by 5px steps
   - Continue while content score stays ≥ 70% of initial
   - Stop when hitting low-content (whitespace) regions
   - Max expansion: 50px per direction

**Output:** Grown boxes that capture full image extent

### Step 6: Filter Contained Boxes
**Purpose:** Remove boxes completely inside other boxes.

Simple geometric check: box A is contained if its entire area fits within box B.

### Step 7: Merge Overlapping Boxes
**Purpose:** Merge boxes that significantly overlap.

Uses **IoU (Intersection over Union)**:
- IoU = overlap_area / union_area
- Merge boxes with IoU > 0.3 (30% overlap)
- Merged box = bounding box of all boxes in group

## Visualization Outputs

The module produces detailed intermediate visualizations (all as numpy arrays):

| Step | Key Name | Description |
|------|----------|-------------|
| 1 | `text_removed` | Grayscale with text filled |
| 2a | `gradient_map` | Edge strength visualization |
| 2b | `edges_all` | All strong edges |
| 2c | `edges_sparse` | Only photo-like edges (cyan) |
| 2d | `angle_visualization` | Gray=all, cyan=photo edges |
| 3 | `edges_connected` | Dilated/connected edges |
| 4 | `z_normalized` | Z-normalized content map |
| 5 | `grown_boxes` | Orange=initial, green=grown, magenta=circles |
| - | `filtered_boxes` | After removing contained boxes (new) |
| - | `final_boxes` | Final merged boxes with labels (new) |

### Dilation Comparison Grid

The module also generates a comparison grid showing results for all 6 dilation combinations (k=3/5/7, i=1/2), each labeled with:
- `k`: kernel size
- `i`: iterations  
- `n`: number of valid clusters detected

This helps visualize the tradeoff between conservative (small k, low i) and aggressive (large k, high i) clustering.

## Parameters

### Main Function: `detect_images()`

```python
def detect_images(image_rgb, boxes, 
                  min_area_ratio=0.015,    # 1.5% of image
                  min_edge_density=30)      # 30 edge pixels
```

- **`min_area_ratio`**: Minimum image size as fraction of total (filters out tiny regions)
- **`min_edge_density`**: Minimum sparse edge pixels in cluster (ensures it's an actual photo, not noise)

### Tuning Guidelines

**If detecting too many false positives:**
- Increase `min_edge_density` (e.g., 40-50)
- Use smaller dilation: k=3, i=1
- Increase `min_area_ratio` (e.g., 0.02 = 2%)

**If missing real images:**
- Decrease `min_edge_density` (e.g., 20-25)
- Use larger dilation: k=5, i=2
- Decrease `min_area_ratio` (e.g., 0.01 = 1%)

## Technical Details

### Why Z-Normalization?

Photos can have any brightness/contrast. Z-normalization makes edges visible regardless of absolute intensity:
- Dark image with dark edges: normalized to visible contrast
- Bright image with subtle edges: normalized to visible contrast
- Works like human visual system (adapts to local context)

### Why Filter Axis-Aligned Edges?

UI elements (text, buttons, dividers) have rectangular boundaries with horizontal/vertical edges. Photos have:
- Diagonal edges (faces, objects at angles)
- Curved edges (rounded shapes, natural objects)
- Mixed angles (complex scenes)

By keeping only non-axis-aligned edges, we focus on photo-like content.

### Why Multiple Dilation Levels?

Different images need different clustering aggressiveness:
- **Tight photos** (clear boundaries): Small dilation works
- **Subtle photos** (low contrast, sparse edges): Need more dilation
- **Fragmented photos**: Need more iterations to connect

The comparison visualization helps debug which level works best for your data.

## Integration with Main Pipeline

The `detect_images()` function is imported into `elements.py` and called from `detect_lines_and_cuts()`:

```python
from image_detection import detect_images

# In detect_lines_and_cuts():
if detect_image_regions:
    image_results = detect_images(image_rgb, boxes)
```

Results are included in the visualization with blue boxes labeled "IMAGE".

