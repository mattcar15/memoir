# Image Detection V2: Flood-Fill Approach

## Why V2?

### Problems with V1 (Dilation-Based)

1. **Over-relies on dilation** to connect gappy edges
   - Needs k=5-7, i=2 to detect anything
   - Connects unrelated regions together
   - Loses precise boundaries

2. **Flawed shape fitting**
   - Circle/rectangle decision based on dilated blob aspect ratio
   - Minimum enclosing circle is way too large
   - Doesn't capture actual image shape

3. **Disconnected from actual content**
   - Fits to edge points, not actual image pixels
   - Content-based growing is a hack to fix bad initial boxes

## V2 Approach: Edge-Seeded Flood-Fill

### Core Idea

**Edges are SEEDS, not boundaries**. Instead of connecting edges with dilation, we:
1. Find sparse photo edges (same as V1)
2. Use edge pixels as seed points
3. Flood-fill similar colors from each seed
4. Merge overlapping filled regions

This naturally handles gappy edges because we're filling the *content*, not connecting the *edges*.

### Algorithm

```
Step 1: Remove text (same as V1)

Step 2: Find sparse photo edges (same as V1)
  - Compute gradients
  - Keep only non-axis-aligned edges (diagonal/curved)
  - These are characteristic of photos, not UI

Step 3: Extract seed points
  - Minimal morphological opening to remove noise
  - Get coordinates of all edge pixels
  - Sample down to ~500 seeds (for performance)

Step 4: Flood-fill from seeds
  For each seed point:
    - Get seed pixel color
    - Flood-fill with tolerance (default ±30)
    - Captures the full image region, even with gappy edges
    - Merge overlapping fills

Step 5: Extract bounding boxes
  - Find contours of filled regions
  - Get tight bounding rectangles

Step 6-7: Filter and merge (same as V1)
```

### Key Parameters

**`color_tolerance`** (default: 30)
- How different colors can be in flood-fill
- Range: 0-255
- Lower = tighter fills (only very similar colors)
- Higher = looser fills (more variation allowed)

**Tuning:**
- Photos with uniform colors (sky, solid backgrounds): use 20-25
- Photos with varied colors (natural scenes): use 30-40
- Complex photos (textures, gradients): use 40-50

## Comparison: V1 vs V2

| Aspect | V1 (Dilation) | V2 (Flood-Fill) |
|--------|---------------|-----------------|
| **Edge gaps** | Needs heavy dilation (k=5-7, i=2) | Handles naturally |
| **Boundaries** | Approximate (dilated blobs) | Tight (actual content) |
| **Shape fitting** | Circle/rectangle heuristic | Content-based (flood-fill) |
| **Precision** | Poor (over-grows) | Good (follows colors) |
| **Performance** | Fast (morphology) | Moderate (flood-fill) |
| **False positives** | Higher (over-connects) | Lower (color-based) |
| **Tuning** | 6 combinations (k×i) | 1 parameter (tolerance) |

## When to Use Which

### Use V1 (Dilation) if:
- Speed is critical
- Images have very strong, continuous edges
- You want to detect edge-outlined regions

### Use V2 (Flood-Fill) if:
- Edges are gappy/sparse (common in photos)
- You want tight, content-based boundaries
- You want fewer false positives
- Moderate performance is acceptable

## Implementation Notes

### Flood-Fill Strategy

We use `cv2.floodFill()` with:
- **FLOODFILL_FIXED_RANGE**: Tolerance relative to seed color (not neighbor)
- **FLOODFILL_MASK_ONLY**: Just mark pixels, don't modify image
- **4-connectivity**: More conservative than 8-connectivity

### Performance Optimization

- **Seed sampling**: Max 500 seeds instead of all edge pixels
- **Early skip**: Skip seeds already in filled regions
- **Region merging**: Merge overlapping fills on-the-fly

### Edge Cases Handled

1. **Overlapping fills**: Merge into single region
2. **Tiny regions**: Filter by min_area
3. **Contained boxes**: Remove fully contained
4. **Adjacent images**: IoU-based merging

## Testing V2

To test the new approach:

```python
# Option 1: Direct usage
from image_detection_v2 import detect_images

result = detect_images(image_rgb, ocr_boxes, 
                       min_area_ratio=0.015,
                       color_tolerance=30)

# Option 2: Compare both
from image_detection import detect_images as detect_v1
from image_detection_v2 import detect_images as detect_v2

result_v1 = detect_v1(image_rgb, ocr_boxes)
result_v2 = detect_v2(image_rgb, ocr_boxes)

print(f"V1 found: {len(result_v1['image_boxes'])} images")
print(f"V2 found: {len(result_v2['image_boxes'])} images")
```

## Migration Path

1. **Test V2** on your dataset
2. **Compare** detection quality (V1 vs V2)
3. **Tune** color_tolerance if needed
4. **Switch** by updating import in `elements.py`:

```python
# Change from:
from image_detection import detect_images

# To:
from image_detection_v2 import detect_images
```

Both modules have identical interfaces, so it's a drop-in replacement!

## Visual Examples

### What V2 Does Better

**Gappy Edges:**
- V1: Needs k=7, i=2 → over-connects → false positives
- V2: Flood-fills through gaps → tight boundaries

**Mixed Content:**
- V1: Circle fits wrong shape → too large → includes non-image
- V2: Follows actual colors → tight → accurate

**Adjacent Images:**
- V1: Dilation merges them → one big box → wrong
- V2: Different colors → separate fills → correct

## Future Improvements

Possible enhancements to V2:
1. **Adaptive tolerance**: Auto-tune based on local color variance
2. **Multi-scale**: Try multiple tolerances, pick best
3. **Alpha shapes**: For non-rectangular images (circles, irregular)
4. **Edge confidence**: Weight seeds by edge strength





