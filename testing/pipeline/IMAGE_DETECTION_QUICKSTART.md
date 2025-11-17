# Image Detection Quick Start

## TL;DR

**Problem with V1:** Circle/rectangle fitting is flawed, relies too heavily on dilation (k=5-7, i=2).

**Solution (V2):** Use edges as seeds for flood-fill instead of connecting them with dilation.

## Test Both Approaches

```bash
cd /Users/mattcarroll/code/memoir/testing/pipeline

# Compare on a single image
python compare_detection_methods.py test_images/YOUR_IMAGE.png

# Check outputs:
# - YOUR_IMAGE_comparison.png (side-by-side visualization)
# - YOUR_IMAGE_comparison_steps/ (intermediate steps)
```

## Switch to V2

If V2 works better, edit **`elements.py`** line 9:

```python
# Change from:
from image_detection import detect_images

# To:
from image_detection_v2 import detect_images
```

Done! Both have identical interfaces.

## Quick Comparison

| Feature | V1 (Dilation) | V2 (Flood-Fill) |
|---------|---------------|-----------------|
| **Gappy edges** | ❌ Needs heavy dilation | ✅ Handles naturally |
| **Boundaries** | ❌ Approximate (blobs) | ✅ Tight (content-based) |
| **Shape fitting** | ❌ Flawed heuristic | ✅ Content-driven |
| **False positives** | ⚠️ Higher | ✅ Lower |
| **Speed** | ✅ Fast | ⚠️ Moderate |
| **Tuning** | 😰 6 combos (k×i) | 😊 1 param (tolerance) |

## Tuning V2

If detection isn't great, adjust `color_tolerance`:

```python
# In elements.py, change the detect_images call:
image_results = detect_images(image_rgb, boxes, color_tolerance=40)
```

- **20-25**: Tight (uniform colors only)
- **30-40**: Balanced (default: 30)
- **40-50**: Loose (varied colors, textures)

## When to Use Which

### Use V1 if:
- ✅ You need maximum speed
- ✅ Images have very strong, continuous edges
- ✅ Current results are already good

### Use V2 if:
- ✅ Edges are sparse/gappy (common)
- ✅ V1 is over-connecting things
- ✅ You want tighter boundaries
- ✅ Moderate performance is OK

## Visual Test Results

After running `compare_detection_methods.py`, you'll see:

```
COMPARISON RESULTS
==============================================================

Detection Counts:
  V1 (Dilation):  3 images
  V2 (Flood-Fill): 2 images

V1 Box Areas:
  Total: 45,000 px²
  Average: 15,000 px²

V2 Box Areas:
  Total: 32,000 px²
  Average: 16,000 px²

Overlap Analysis:
  Both detected: 2 images
  V1 only: 1 images (possible false positive)
  V2 only: 0 images

RECOMMENDATIONS:
  → V1 found more boxes (possible false positives)
  → V2 is more conservative - likely more accurate
  → V1 boxes are larger (likely over-growing)
```

## More Details

- **Full comparison**: `IMAGE_DETECTION_V2.md`
- **V1 approach**: `IMAGE_DETECTION.md`
- **Implementation**: `image_detection_v2.py`


