"""
Image/photo detection using edge analysis and clustering.

This module detects image regions in screenshots by:
1. Finding non-axis-aligned edges (diagonal/curved) that are characteristic of photos
2. Clustering these edges using morphological dilation
3. Fitting initial bounding boxes to clusters
4. Growing boxes based on content until hitting whitespace
5. Filtering and merging overlapping detections
"""

import numpy as np
from PIL import Image
import cv2


def detect_images(image_rgb, boxes, min_area_ratio=0.015, min_edge_density=30):
    """
    Detect image/photo regions in a screenshot.
    
    Photos have mixed edge angles (diagonal, curved) unlike UI elements which have
    mostly horizontal/vertical edges. We find these sparse mixed edges, cluster them,
    and grow bounding boxes until we hit whitespace.
    
    Args:
        image_rgb: Input image as RGB numpy array
        boxes: OCR text boxes to suppress (list of polygons)
        min_area_ratio: Minimum area as fraction of total image (default 1.5%)
        min_edge_density: Minimum number of edge pixels required in a cluster
        
    Returns:
        Dict containing:
        - image_boxes: List of (x, y, w, h) rectangles for detected images
        - intermediate_steps: Dict of processing steps for visualization
    """
    height, width = image_rgb.shape[:2]
    total_area = height * width
    min_area = int(total_area * min_area_ratio)
    
    print(f"\n=== Image Detection ===")
    print(f"Image size: {width}x{height}, min area: {min_area} px")
    
    # Step 1: Remove text to avoid detecting text as images
    print("Step 1: Removing text regions...")
    gray_no_text = create_text_removed_image(image_rgb, boxes)
    
    # Step 2: Find edges and their angles
    print("Step 2: Computing edge gradients and angles...")
    edges_all, edges_sparse, gradient_vis, angle_vis = find_sparse_edges(gray_no_text)
    
    # Step 3: Cluster sparse edges with multiple dilation levels
    print("Step 3: Clustering edges with different dilation levels...")
    dilation_results = cluster_edges_multiple_levels(edges_sparse, min_area, min_edge_density)
    
    # Use minimal dilation (conservative detection)
    best_result = dilation_results[0]
    print(f"  → Using kernel={best_result['kernel_size']}x{best_result['kernel_size']}, "
          f"iterations={best_result['iterations']}, found {best_result['num_valid_clusters']} clusters")
    
    # Step 4: Compute z-normalized image for content-based box growing
    print("Step 4: Computing content map for box growing...")
    Z_normalized = compute_content_map(gray_no_text, height, width)
    
    # Step 5: Fit and grow boxes for each cluster
    print("Step 5: Fitting and growing boxes...")
    candidates = fit_and_grow_boxes(
        best_result, edges_sparse, Z_normalized, gray_no_text,
        min_area, min_edge_density, width, height
    )
    print(f"  → Generated {len(candidates)} candidate boxes")
    
    # Step 6: Filter contained boxes
    print("Step 6: Filtering contained boxes...")
    candidates = filter_contained_boxes(candidates)
    print(f"  → {len(candidates)} boxes after filtering")
    
    # Step 7: Merge overlapping boxes
    print("Step 7: Merging overlapping boxes...")
    merged_boxes = merge_overlapping_boxes(candidates, iou_threshold=0.3)
    print(f"  → Final count: {len(merged_boxes)} image regions detected")
    
    # Create visualization comparisons for all dilation levels
    dilation_visualizations = []
    for result in dilation_results:
        vis = visualize_dilation_result(
            gray_no_text, result, edges_sparse, min_area, min_edge_density
        )
        dilation_visualizations.append({
            'kernel_size': result['kernel_size'],
            'iterations': result['iterations'],
            'num_clusters': result['num_valid_clusters'],
            'image': vis
        })
    
    # Create step-by-step visualization
    step_vis = create_step_visualization(
        gray_no_text, edges_sparse, best_result, candidates, merged_boxes,
        Z_normalized
    )
    
    return {
        "image_boxes": merged_boxes,
        "intermediate_steps": {
            # Original key names (for backward compatibility with main.py)
            # All as numpy arrays for consistency
            "text_removed": gray_no_text,
            "gradient_map": gradient_vis,
            "edges_all": edges_all,
            "edges_sparse": edges_sparse,
            "angle_visualization": angle_vis,
            "edges_connected": best_result['edges_dilated'],
            "z_normalized": Z_normalized,
            "grown_boxes": np.array(step_vis['grown_boxes']),  # Convert PIL to numpy
            "dilation_comparisons": dilation_visualizations,
            # New additional steps (for future use)
            "filtered_boxes": np.array(step_vis['filtered_boxes']),
            "final_boxes": np.array(step_vis['final_boxes']),
        },
    }


def create_text_removed_image(image_rgb, boxes):
    """
    Create grayscale image with text regions filled with background color.
    
    For each OCR box, samples the border pixels to detect background color
    and fills the box with it, effectively removing text.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    result = gray.copy()
    height, width = gray.shape
    
    for box in boxes:
        if not box:
            continue
            
        # Get bounding box
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
        
        # Sample border (20% from edges) to get background color
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
    """
    Find non-axis-aligned edges that are characteristic of photos.
    
    Returns edges_all, edges_sparse, gradient_vis, angle_vis
    """
    # Compute gradient magnitude and angle
    grad_x = cv2.Scharr(gray_image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray_image, cv2.CV_32F, 0, 1)
    gradient_mag = cv2.magnitude(grad_x, grad_y)
    gradient_angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)
    
    # Normalize gradient for visualization
    gradient_vis = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Threshold to get strong edges
    threshold_value = np.percentile(gradient_mag, 85)  # Top 15%
    edges_all = (gradient_mag > threshold_value).astype(np.uint8) * 255
    
    # Filter OUT axis-aligned edges (keep diagonal/curved edges for photos)
    angle_tolerance = 15  # degrees
    
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
    
    # Keep only non-axis-aligned edges
    edges_sparse = edges_all.copy()
    edges_sparse[is_axis_aligned] = 0
    
    # Create angle visualization
    height, width = gray_image.shape
    angle_vis = np.zeros((height, width, 3), dtype=np.uint8)
    angle_vis[edges_all > 0] = [100, 100, 100]  # Gray: all edges
    angle_vis[edges_sparse > 0] = [0, 255, 255]  # Cyan: sparse (photo) edges
    
    return edges_all, edges_sparse, gradient_vis, angle_vis


def cluster_edges_multiple_levels(edges_sparse, min_area, min_edge_density):
    """
    Cluster sparse edges using multiple dilation levels.
    
    Dilation connects nearby edge pixels into clusters. We try different:
    - kernel_size: Size of dilation structuring element (3x3, 5x5, 7x7)
    - iterations: How many times to apply dilation (1 or 2)
    
    More dilation = larger clusters, fewer components
    Less dilation = tighter clusters, more components
    
    Returns list of dilation results, sorted by (kernel_size, iterations).
    """
    results = []
    
    for kernel_size in [3, 5, 7]:
        for iterations in [1, 2]:
            # Create circular structuring element
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Dilate edges to connect nearby pixels
            edges_dilated = cv2.dilate(edges_sparse, kernel, iterations=iterations)
            
            # Find connected components (clusters)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                edges_dilated, connectivity=8
            )
            
            # Count valid clusters (meet area and density requirements)
            num_valid = 0
            for label_id in range(1, num_labels):
                x, y, w, h, area = stats[label_id]
                if area < min_area:
                    continue
                    
                cluster_mask = (labels == label_id).astype(np.uint8)
                edge_pixels = np.sum((edges_sparse > 0) & (cluster_mask > 0))
                if edge_pixels >= min_edge_density:
                    num_valid += 1
            
            results.append({
                'kernel_size': kernel_size,
                'iterations': iterations,
                'edges_dilated': edges_dilated,
                'num_labels': num_labels,
                'labels': labels,
                'stats': stats,
                'centroids': centroids,
                'num_valid_clusters': num_valid,
            })
    
    return results


def compute_content_map(gray_image, height, width):
    """
    Compute z-normalized image for content-aware box growing.
    
    Z-normalization makes the image contrast-invariant, so we can detect
    content boundaries regardless of brightness/contrast.
    """
    # CLAHE enhancement
    tile = 64 if max(height, width) >= 1200 else 32
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(tile, tile))
    gray_clahe = clahe.apply(gray_image)
    gray_enhanced = cv2.GaussianBlur(gray_clahe, (0, 0), 1.0)
    
    # Gaussian divisive normalization
    r = int(max(9, round(0.03 * min(height, width))))
    sigma_blur = max(1.5, r / 6.0)
    
    L = gray_enhanced.astype(np.float32)
    mu = cv2.GaussianBlur(L, (0, 0), sigma_blur)
    sq = cv2.GaussianBlur(L * L, (0, 0), sigma_blur)
    sigma = np.sqrt(np.maximum(sq - mu * mu, 1e-6))
    sigma = np.clip(sigma, 5.0, 40.0)
    
    Z = (L - mu) / (sigma + 1e-6)
    Z = np.tanh(0.5 * Z)
    Z_normalized = np.clip((Z * 80.0 + 128.0), 0, 255).astype(np.uint8)
    
    return Z_normalized


def fit_and_grow_boxes(dilation_result, edges_sparse, content_map, gray_image,
                       min_area, min_edge_density, width, height):
    """
    For each edge cluster, fit initial box and grow based on content.
    
    Steps for each cluster:
    1. Get bounding box from connected component stats
    2. Count actual edge pixels in cluster (filter by density)
    3. Decide shape: circle for square-ish clusters, rectangle otherwise
    4. Grow box based on content map until hitting low-content regions
    
    Returns list of candidate dicts with 'rect', 'edge_density', 'initial_box', 'shape_type'.
    """
    num_labels = dilation_result['num_labels']
    labels = dilation_result['labels']
    stats = dilation_result['stats']
    
    candidates = []
    
    for label_id in range(1, num_labels):  # Skip background (label 0)
        x, y, w, h, area = stats[label_id]
        
        # Filter by minimum area
        if area < min_area:
            continue
        
        # Count actual sparse edge pixels in this cluster
        cluster_mask = (labels == label_id).astype(np.uint8)
        edge_pixels_in_cluster = np.sum((edges_sparse > 0) & (cluster_mask > 0))
        
        if edge_pixels_in_cluster < min_edge_density:
            continue
        
        # Get edge points for shape fitting
        edge_points = np.column_stack(np.where((edges_sparse > 0) & (cluster_mask > 0)))
        edge_points = edge_points[:, ::-1]  # Convert (row, col) to (x, y)
        
        # Decide shape based on aspect ratio
        aspect_ratio = w / h if h > 0 else 1.0
        if 0.7 <= aspect_ratio <= 1.3:  # Nearly square - try circular fit
            circle_box = fit_circle_to_points(edge_points, width, height)
            initial_box = circle_box if circle_box else (x, y, w, h)
            shape_type = "circle"
        else:
            initial_box = (x, y, w, h)
            shape_type = "rectangle"
        
        # Grow box based on content map
        grown_box = grow_box_on_content(initial_box, content_map, max_expansion=50)
        
        gx, gy, gw, gh = grown_box
        if gw * gh < min_area:
            continue
        
        candidates.append({
            "rect": grown_box,
            "edge_density": edge_pixels_in_cluster,
            "initial_box": initial_box,
            "shape_type": shape_type,
        })
    
    return candidates


def fit_circle_to_points(edge_points, width, height):
    """
    Fit minimum enclosing circle to edge points.
    
    Returns (x, y, w, h) bounding box of circle, or None if fit fails.
    """
    if len(edge_points) < 3:
        return None
    
    points = edge_points.astype(np.float32)
    
    try:
        (cx, cy), radius = cv2.minEnclosingCircle(points)
        
        x = int(max(0, cx - radius))
        y = int(max(0, cy - radius))
        w = int(min(width - x, 2 * radius))
        h = int(min(height - y, 2 * radius))
        
        return (x, y, w, h)
    except:
        return None


def grow_box_on_content(initial_box, content_image, max_expansion=50):
    """
    Grow bounding box based on content until hitting low-content regions.
    
    Uses content variance (std dev) as measure of "interesting" regions.
    Stops growing when content score drops below 70% of initial score.
    
    Args:
        initial_box: (x, y, w, h)
        content_image: Z-normalized grayscale image
        max_expansion: Max pixels to expand in each direction
        
    Returns:
        (x, y, w, h) grown box
    """
    height, width = content_image.shape
    x, y, w, h = initial_box
    
    def get_content_score(bx, by, bw, bh):
        """Measure content richness via standard deviation."""
        bx = max(0, bx)
        by = max(0, by)
        bx2 = min(width, bx + bw)
        by2 = min(height, by + bh)
        if bx2 <= bx or by2 <= by:
            return 0
        roi = content_image[by:by2, bx:bx2]
        return np.std(roi)
    
    initial_score = get_content_score(x, y, w, h)
    threshold_score = initial_score * 0.7
    expansion_step = 5
    
    # Grow right
    current_w = w
    for _ in range(max_expansion // expansion_step):
        new_w = current_w + expansion_step
        if x + new_w > width:
            break
        if get_content_score(x, y, new_w, h) >= threshold_score:
            current_w = new_w
        else:
            break
    
    # Grow down
    current_h = h
    for _ in range(max_expansion // expansion_step):
        new_h = current_h + expansion_step
        if y + new_h > height:
            break
        if get_content_score(x, y, current_w, new_h) >= threshold_score:
            current_h = new_h
        else:
            break
    
    # Grow left
    current_x = x
    current_w_final = current_w
    for _ in range(max_expansion // expansion_step):
        new_x = current_x - expansion_step
        new_w = current_w_final + expansion_step
        if new_x < 0:
            break
        if get_content_score(new_x, y, new_w, current_h) >= threshold_score:
            current_x = new_x
            current_w_final = new_w
        else:
            break
    
    # Grow up
    current_y = y
    current_h_final = current_h
    for _ in range(max_expansion // expansion_step):
        new_y = current_y - expansion_step
        new_h = current_h_final + expansion_step
        if new_y < 0:
            break
        if get_content_score(current_x, new_y, current_w_final, new_h) >= threshold_score:
            current_y = new_y
            current_h_final = new_h
        else:
            break
    
    return (current_x, current_y, current_w_final, current_h_final)


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
    """
    Merge boxes with IoU (Intersection over Union) > threshold.
    
    Returns list of (x, y, w, h) merged boxes.
    """
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
        
        # Merge group into bounding box
        if group:
            x_min = min(r[0] for r in group)
            y_min = min(r[1] for r in group)
            x_max = max(r[0] + r[2] for r in group)
            y_max = max(r[1] + r[3] for r in group)
            merged.append((x_min, y_min, x_max - x_min, y_max - y_min))
    
    return merged


def visualize_dilation_result(gray_image, dilation_result, edges_sparse,
                               min_area, min_edge_density):
    """
    Create visualization showing detected clusters for a specific dilation level.
    
    Draws green boxes around valid clusters that meet area and density requirements.
    """
    vis = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    
    num_labels = dilation_result['num_labels']
    labels = dilation_result['labels']
    stats = dilation_result['stats']
    
    box_count = 0
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        
        if area < min_area:
            continue
        
        cluster_mask = (labels == label_id).astype(np.uint8)
        edge_pixels = np.sum((edges_sparse > 0) & (cluster_mask > 0))
        
        if edge_pixels < min_edge_density:
            continue
        
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        box_count += 1
    
    # Add label
    label = f"k={dilation_result['kernel_size']}, i={dilation_result['iterations']}, n={box_count}"
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    return Image.fromarray(vis)


def create_step_visualization(gray_image, edges_sparse, dilation_result, 
                              candidates, merged_boxes, content_map):
    """
    Create step-by-step visualizations showing box evolution.
    
    Returns dict with:
    - grown_boxes: Initial boxes (orange) and grown boxes (green)
    - filtered_boxes: After removing contained boxes
    - final_boxes: After merging overlapping boxes
    """
    # Visualization 1: Box growing
    grown_vis = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    for cand in candidates:
        ix, iy, iw, ih = cand['initial_box']
        gx, gy, gw, gh = cand['rect']
        
        # Orange: initial box
        cv2.rectangle(grown_vis, (ix, iy), (ix + iw, iy + ih), (255, 165, 0), 1)
        # Green: grown box
        cv2.rectangle(grown_vis, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
        
        # Magenta circle for circular fits
        if cand['shape_type'] == "circle":
            cx, cy = ix + iw // 2, iy + ih // 2
            radius = max(iw, ih) // 2
            cv2.circle(grown_vis, (cx, cy), radius, (255, 0, 255), 1)
    
    # Visualization 2: After filtering (just the boxes, no initial)
    filtered_vis = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    for cand in candidates:
        gx, gy, gw, gh = cand['rect']
        cv2.rectangle(filtered_vis, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
    
    # Visualization 3: Final merged boxes
    final_vis = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    for x, y, w, h in merged_boxes:
        cv2.rectangle(final_vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Add "IMAGE" label
        cv2.putText(final_vis, "IMAGE", (x + 5, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return {
        'grown_boxes': Image.fromarray(grown_vis),
        'filtered_boxes': Image.fromarray(filtered_vis),
        'final_boxes': Image.fromarray(final_vis),
    }

