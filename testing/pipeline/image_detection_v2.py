"""
Improved image/photo detection using flood-fill approach.

Instead of relying heavily on dilation to connect edges, this approach:
1. Finds sparse photo-like edges (diagonal/curved)
2. Uses edges as SEEDS for flood-fill operations
3. Flood-fills similar-colored regions from each edge seed
4. Merges overlapping flood-filled regions into image boxes

This handles gappy edges naturally without over-dilating.
"""

import numpy as np
from PIL import Image
import cv2


def detect_images(image_rgb, boxes, min_area_ratio=0.015, color_tolerance=30):
    """
    Detect image/photo regions using edge-seeded flood-fill.
    
    Args:
        image_rgb: Input image as RGB numpy array
        boxes: OCR text boxes to suppress (list of polygons)
        min_area_ratio: Minimum area as fraction of total image (default 1.5%)
        color_tolerance: Color similarity tolerance for flood-fill (0-255)
        
    Returns:
        Dict containing:
        - image_boxes: List of (x, y, w, h) rectangles for detected images
        - intermediate_steps: Dict of processing steps for visualization
    """
    height, width = image_rgb.shape[:2]
    total_area = height * width
    min_area = int(total_area * min_area_ratio)
    
    print(f"\n=== Image Detection V2 (Flood-Fill) ===")
    print(f"Image size: {width}x{height}, min area: {min_area} px")
    
    # Step 1: Remove text
    print("Step 1: Removing text regions...")
    gray_no_text = create_text_removed_image(image_rgb, boxes)
    image_no_text = image_rgb.copy()
    
    # Also remove text from color image
    for box in boxes:
        if not box:
            continue
        pts = np.array(box, dtype=np.int32)
        cv2.fillPoly(image_no_text, [pts], (255, 255, 255))
    
    # Step 2: Find sparse photo edges
    print("Step 2: Finding photo-characteristic edges...")
    edges_all, edges_sparse, gradient_vis, angle_vis = find_sparse_edges(gray_no_text)
    
    # Step 3: Get edge seed points (minimal dilation just to denoise)
    print("Step 3: Extracting edge seed points...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_cleaned = cv2.morphologyEx(edges_sparse, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Get coordinates of all edge pixels as seeds
    edge_points = np.column_stack(np.where(edges_cleaned > 0))
    print(f"  → Found {len(edge_points)} edge seed points")
    
    # Step 4: Flood-fill from edge seeds
    print("Step 4: Flood-filling from edge seeds...")
    filled_regions, fill_vis = flood_fill_from_seeds(
        image_no_text, edge_points, color_tolerance, min_area
    )
    print(f"  → Generated {len(filled_regions)} filled regions")
    
    # Step 5: Extract bounding boxes from filled regions
    print("Step 5: Extracting bounding boxes...")
    candidates = extract_boxes_from_regions(filled_regions, min_area)
    print(f"  → {len(candidates)} candidate boxes")
    
    # Step 6: Filter contained boxes
    print("Step 6: Filtering contained boxes...")
    candidates = filter_contained_boxes(candidates)
    print(f"  → {len(candidates)} after filtering")
    
    # Step 7: Merge overlapping boxes
    print("Step 7: Merging overlapping boxes...")
    merged_boxes = merge_overlapping_boxes(candidates, iou_threshold=0.3)
    print(f"  → Final count: {len(merged_boxes)} image regions detected")
    
    # Create visualizations
    boxes_vis = create_boxes_visualization(image_no_text, candidates, merged_boxes)
    
    return {
        "image_boxes": merged_boxes,
        "intermediate_steps": {
            "text_removed": gray_no_text,
            "gradient_map": gradient_vis,
            "edges_all": edges_all,
            "edges_sparse": edges_sparse,
            "angle_visualization": angle_vis,
            "edges_connected": edges_cleaned,
            "z_normalized": gray_no_text,  # Placeholder for compatibility
            "grown_boxes": np.array(boxes_vis['candidates']),
            "filled_regions": fill_vis,
            "final_boxes": np.array(boxes_vis['final']),
            "dilation_comparisons": [],  # Not used in v2
        },
    }


def create_text_removed_image(image_rgb, boxes):
    """Create grayscale image with text regions filled with background color."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    result = gray.copy()
    height, width = gray.shape
    
    for box in boxes:
        if not box:
            continue
        pts = np.array(box, dtype=np.float32)
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        x0 = max(0, int(np.floor(x_coords.min())))
        y0 = max(0, int(np.floor(y_coords.min())))
        x1 = min(width, int(np.ceil(x_coords.max())))
        y1 = min(height, int(np.ceil(y_coords.max())))
        
        if x1 - x0 <= 1 or y1 - y0 <= 1:
            continue
        region = result[y0:y1, x0:x1]
        if region.size == 0:
            continue
        
        border_size = max(1, min(int((x1 - x0) * 0.2), int((y1 - y0) * 0.2)))
        border_pixels = []
        
        if border_size < region.shape[0]:
            border_pixels.append(region[:border_size, :].flatten())
            border_pixels.append(region[-border_size:, :].flatten())
        if border_size < region.shape[1]:
            border_pixels.append(region[:, :border_size].flatten())
            border_pixels.append(region[:, -border_size:].flatten())
        
        if border_pixels:
            all_border = np.concatenate(border_pixels)
            background_color = int(np.median(all_border))
            result[y0:y1, x0:x1] = background_color
    
    return result


def find_sparse_edges(gray_image):
    """Find non-axis-aligned edges characteristic of photos."""
    grad_x = cv2.Scharr(gray_image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray_image, cv2.CV_32F, 0, 1)
    gradient_mag = cv2.magnitude(grad_x, grad_y)
    gradient_angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)
    
    gradient_vis = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    threshold_value = np.percentile(gradient_mag, 85)
    edges_all = (gradient_mag > threshold_value).astype(np.uint8) * 255
    
    angle_tolerance = 15
    is_horizontal = (
        (gradient_angle < angle_tolerance) |
        (gradient_angle > (180 - angle_tolerance)) & (gradient_angle < (180 + angle_tolerance)) |
        (gradient_angle > (360 - angle_tolerance))
    )
    is_vertical = (
        (gradient_angle > (90 - angle_tolerance)) & (gradient_angle < (90 + angle_tolerance)) |
        (gradient_angle > (270 - angle_tolerance)) & (gradient_angle < (270 + angle_tolerance))
    )
    
    is_axis_aligned = is_horizontal | is_vertical
    edges_sparse = edges_all.copy()
    edges_sparse[is_axis_aligned] = 0
    
    height, width = gray_image.shape
    angle_vis = np.zeros((height, width, 3), dtype=np.uint8)
    angle_vis[edges_all > 0] = [100, 100, 100]
    angle_vis[edges_sparse > 0] = [0, 255, 255]
    
    return edges_all, edges_sparse, gradient_vis, angle_vis


def flood_fill_from_seeds(image_rgb, seed_points, tolerance, min_area):
    """
    Flood-fill from edge seed points to capture full image regions.
    
    For each seed point, flood-fill similar colors. Merge overlapping fills.
    """
    height, width = image_rgb.shape[:2]
    filled_mask = np.zeros((height, width), dtype=np.uint8)
    region_labels = np.zeros((height, width), dtype=np.int32)
    
    # Create visualization
    fill_vis = cv2.cvtColor(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    
    regions = []
    region_id = 1
    
    # Sample seeds (too many seeds is slow, so sample)
    max_seeds = 500
    if len(seed_points) > max_seeds:
        indices = np.random.choice(len(seed_points), max_seeds, replace=False)
        seed_points = seed_points[indices]
    
    for y, x in seed_points:
        # Skip if already filled
        if filled_mask[y, x] > 0:
            continue
        
        # Create a temporary mask for this flood-fill
        temp_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        
        # Flood-fill with tolerance
        seed_color = tuple(int(c) for c in image_rgb[y, x])
        lo = (tolerance,) * 3
        hi = (tolerance,) * 3
        
        flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
        area, _, _, rect = cv2.floodFill(
            image_rgb.copy(), temp_mask, (x, y), 
            (255, 255, 255), lo, hi, flags
        )
        
        # Extract the filled region (remove border padding)
        region_mask = temp_mask[1:-1, 1:-1]
        
        # Check if region is large enough
        if area < min_area:
            continue
        
        # Check if this region overlaps significantly with existing
        # If so, merge them; otherwise create new region
        overlap_ids = np.unique(region_labels[region_mask > 0])
        overlap_ids = overlap_ids[overlap_ids > 0]
        
        if len(overlap_ids) > 0:
            # Merge into existing region(s)
            merge_id = overlap_ids[0]
            for oid in overlap_ids:
                region_labels[region_labels == oid] = merge_id
            region_labels[region_mask > 0] = merge_id
        else:
            # New region
            region_labels[region_mask > 0] = region_id
            regions.append(region_id)
            region_id += 1
        
        filled_mask = (region_labels > 0).astype(np.uint8) * 255
    
    # Collect unique regions
    unique_regions = []
    for rid in regions:
        mask = (region_labels == rid).astype(np.uint8)
        if np.sum(mask) >= min_area:
            unique_regions.append(mask)
            # Colorize in visualization
            color = np.random.randint(50, 255, 3).tolist()
            fill_vis[mask > 0] = color
    
    return unique_regions, fill_vis


def extract_boxes_from_regions(regions, min_area):
    """Extract tight bounding boxes from filled regions."""
    candidates = []
    
    for mask in regions:
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Get bounding box of largest contour
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        
        if w * h >= min_area:
            candidates.append({
                'rect': (x, y, w, h),
                'area': w * h,
                'contour': contour
            })
    
    return candidates


def filter_contained_boxes(candidates):
    """Remove boxes completely contained inside other boxes."""
    if not candidates:
        return []
    
    def is_contained(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        return (x1 >= x2 and y1 >= y2 and 
                x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2)
    
    filtered = []
    for i, cand1 in enumerate(candidates):
        is_inside = any(
            i != j and is_contained(cand1["rect"], cand2["rect"])
            for j, cand2 in enumerate(candidates)
        )
        if not is_inside:
            filtered.append(cand1)
    
    return filtered


def merge_overlapping_boxes(candidates, iou_threshold=0.3):
    """Merge boxes with IoU > threshold."""
    if not candidates:
        return []
    
    def iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union if union > 0 else 0.0
    
    merged = []
    used = set()
    
    for i, cand1 in enumerate(candidates):
        if i in used:
            continue
        
        group = [cand1["rect"]]
        used.add(i)
        
        for j, cand2 in enumerate(candidates[i + 1:], start=i + 1):
            if j in used:
                continue
            if iou(cand1["rect"], cand2["rect"]) > iou_threshold:
                group.append(cand2["rect"])
                used.add(j)
        
        if group:
            x_min = min(r[0] for r in group)
            y_min = min(r[1] for r in group)
            x_max = max(r[0] + r[2] for r in group)
            y_max = max(r[1] + r[3] for r in group)
            merged.append((x_min, y_min, x_max - x_min, y_max - y_min))
    
    return merged


def create_boxes_visualization(image, candidates, merged_boxes):
    """Create visualizations showing detected boxes."""
    # Candidates
    cand_vis = image.copy()
    for cand in candidates:
        x, y, w, h = cand['rect']
        cv2.rectangle(cand_vis, (x, y), (x + w, y + h), (255, 165, 0), 2)
    
    # Final
    final_vis = image.copy()
    for x, y, w, h in merged_boxes:
        cv2.rectangle(final_vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(final_vis, "IMAGE", (x + 5, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return {
        'candidates': Image.fromarray(cand_vis),
        'final': Image.fromarray(final_vis),
    }


