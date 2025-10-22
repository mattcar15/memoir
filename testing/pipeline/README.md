# OCR Pipeline with Menu Detection

A modular pipeline for processing screenshots with OCR, paragraph grouping, and menu region detection.

## Architecture

The pipeline is organized into clean, focused modules:

### Core Modules

- **`utils.py`** - Image loading, resizing, and geometric helper functions
- **`ocr_processing.py`** - PaddleOCR initialization and result normalization  
- **`layout_analysis.py`** - Chrome filtering, column detection, paragraph grouping
- **`menu_detection.py`** - Heuristic-based menu region detection
- **`visualization.py`** - Drawing functions for OCR boxes, groups, and menu regions
- **`main.py`** - Pipeline orchestrator and CLI entry point

## Features

### OCR Processing
- High-quality image resizing with aspect ratio preservation
- PaddleOCR 3.x integration with optimized settings
- Handles multiple OCR output formats (2.x and 3.x)

### Layout Analysis
- Filters out browser/app chrome
- Multi-column text detection using DBSCAN clustering
- Paragraph grouping based on vertical spacing

### Menu Detection
The pipeline detects three menu regions with confidence scoring:
- **Top strip** (0-16% height) - for top navigation bars
- **Left strip** (0-18% width) - for left sidebars  
- **Right strip** (82-100% width) - for right sidebars

#### Scoring Heuristics
Menu detection uses multiple features:
- **Density** - number of text boxes per area
- **Alignment** - how well boxes align (low std dev)
- **Spacing regularity** - uniform gaps between elements
- **Short tokens** - fraction of 1-2 character text (icons, labels)
- **Spill penalty** - boxes extending beyond the region boundary

Scoring formula:
```
score = 1.6*density + 2.0*alignment + 1.2*spacing + 0.8*short_rate - 1.5*spill
```

Thresholds:
- `score >= 5.0` → "menu" (detected)
- `score >= 3.0` → "maybe" (uncertain)
- `score < 3.0` → not a menu

### Visualization
Output images show:
- ✅ **Green boxes** - OCR text regions
- 🎨 **Colored overlays** - Paragraph groups with labels (G1, G2, etc.)
- 🔴 **Red regions** - Detected menus
- 🟡 **Yellow regions** - Uncertain ("maybe") menus
- ⚪ **Gray regions** - Not menus (low scores)

## Usage

### Basic Usage
```bash
cd testing/pipeline
python main.py
```

This processes images from `../test_images/` and saves results to `./out/`.

### Custom Directories
```bash
python main.py --input-dir /path/to/images --output-dir /path/to/output
```

### Arguments
- `--input-dir` - Directory containing input images (default: `../test_images`)
- `--output-dir` - Directory for output images (default: `./out`)

## Example Output

```
Processing: screenshot.png
================================================================================
Original size: 5120x2818
Resized to: 1080x594 (scale=0.211)
Load time: 0.34s
OCR time: 12.57s
Detected 85 text regions
Paragraph groups: 7
Dropped (chrome): 19
Menu detection time: 0.00s

Menu Detection Results:
     top:   menu (score:   7.54)
    left:   none (score:   2.31)
   right:   none (score:   2.57)
Visualization time: 0.05s
Saved to: ./out/screenshot_processed.png
TOTAL TIME: 12.98s
```

## Dependencies

Required packages (included in main project's `requirements.txt`):
- `paddleocr` - OCR engine
- `opencv-python` (cv2) - Image processing
- `Pillow` (PIL) - Image manipulation
- `numpy` - Numerical operations
- `matplotlib` - Visualization
- `scikit-learn` - Clustering (DBSCAN)

## Pipeline Flow

```
Input Image
    ↓
Load & Resize (PIL, high quality)
    ↓
OCR Detection (PaddleOCR)
    ↓
Result Normalization
    ↓
├─→ Layout Analysis
│   ├─ Filter Chrome
│   ├─ Column Detection
│   └─ Paragraph Grouping
│
└─→ Menu Detection
    ├─ Top Strip Analysis
    ├─ Left Strip Analysis
    └─ Right Strip Analysis
    ↓
Combined Visualization
    ↓
Save Output Image
```

## Extending the Pipeline

### Adding Custom Heuristics
Modify `menu_detection.py` to add new scoring features or adjust weights.

### Custom Visualization
Extend `visualization.py` to add new drawing modes or annotation styles.

### Additional Layout Analysis
Add new grouping strategies in `layout_analysis.py`.

## Performance

Average processing time: ~16-18s per image
- Load: <1s
- OCR: 5-30s (depends on text density)
- Analysis: <0.1s
- Visualization: <0.2s

