"""
Element segmentation functions for detecting UI elements and structure.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from utils import box_bounds
from image_detection import detect_images


__all__ = [
    "draw_element_ocr_boxes",
    "create_text_removed_image",
    "detect_structural_lines",
    "detect_whitespace_cuts",
    "detect_lines_and_cuts",
]


def draw_element_ocr_boxes(image_rgb, boxes, texts=None, scores=None, drop_score=0.5):
    """
    Draw OCR bounding boxes on the cropped image (element segmentation).
    """
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    font = None

    for i, box in enumerate(boxes):
        score = scores[i] if scores and i < len(scores) else None
        if score is not None and score < drop_score:
            continue

        poly = [(float(x), float(y)) for x, y in (box or [])]
        if not poly:
            continue
        draw.line(poly + [poly[0]], fill="lime", width=2)

        if texts and i < len(texts):
            label = str(texts[i])
            text_width = (
                draw.textlength(label, font=font)
                if hasattr(draw, "textlength")
                else 8 * len(label)
            )
            text_height = font.size if font else 16
            text_pos = (poly[0][0], max(0, poly[0][1] - text_height - 2))
            background = [
                (text_pos[0], text_pos[1]),
                (text_pos[0] + text_width + 4, text_pos[1] + text_height + 4),
            ]
            draw.rectangle(background, fill="black")
            draw.text(
                (text_pos[0] + 2, text_pos[1] + 2), label, fill="white", font=font
            )

    return pil_image


def create_text_removed_image(image_rgb, boxes):
    """
    Create a black and white image with OCR box text removed by background color detection.
    For each box, detect the main background color and fill the box with it.
    Returns the processed image as a numpy array.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    result = gray.copy()
    height, width = gray.shape

    for box in boxes:
        if not box:
            continue
        bx0, by0, bx1, by1 = box_bounds(box)
        x0 = max(0, int(round(bx0)))
        y0 = max(0, int(round(by0)))
        x1 = min(width, int(round(bx1)))
        y1 = min(height, int(round(by1)))

        if x1 - x0 <= 1 or y1 - y0 <= 1:
            continue

        region = result[y0:y1, x0:x1]
        if region.size == 0:
            continue

        # Sample corners and edges to get background color
        # Use a 20% border around the edge
        border_size = max(1, min(int((x1 - x0) * 0.2), int((y1 - y0) * 0.2)))

        # Collect border pixels
        border_pixels = []

        # Top and bottom edges
        if border_size < region.shape[0]:
            border_pixels.append(region[:border_size, :].flatten())
            border_pixels.append(region[-border_size:, :].flatten())

        # Left and right edges
        if border_size < region.shape[1]:
            border_pixels.append(region[:, :border_size].flatten())
            border_pixels.append(region[:, -border_size:].flatten())

        if border_pixels:
            all_border = np.concatenate(border_pixels)
            # Use median as it's robust to outliers (text pixels)
            background_color = int(np.median(all_border))

            # Fill the entire region with background color to remove text
            result[y0:y1, x0:x1] = background_color

    return result


def detect_structural_lines(
    image_rgb,
    boxes,
    min_line_length_ratio=0.16,
    max_gap=8,
    angle_tolerance=2.0,
    dilation_size=4,
):
    """
    Detect structural lines and borders using Hough transform.

    Returns:
        - edges: Canny edge image
        - horizontal_lines: Raw horizontal line candidates with length
        - vertical_lines: Raw vertical line candidates with length
        - h_lines: Merged horizontal lines
        - v_lines: Merged vertical lines
        - separator_mask: Binary mask of separator lines
        - gray_processed: Text-removed grayscale image
    """
    height, width = image_rgb.shape[:2]

    # Step 1: Create text-removed BW image for better line detection
    gray_processed = create_text_removed_image(image_rgb, boxes)  # uint8, 0..255

    # --- Step 2: CLAHE (debanded) ---
    # Bigger tiles + lower clip to avoid blocky "steps". Use L-only style since we're already gray.
    # Pick tile size by image scale.
    tile = 64 if max(height, width) >= 1200 else 32
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(tile, tile))
    gray_clahe = clahe.apply(gray_processed)

    # very light debanding to smooth tile seams (does not wash edges)
    gray_enhanced = cv2.GaussianBlur(gray_clahe, (0, 0), 1.0)

    # --- Step 3: Gaussian divisive normalization (contrast-invariant) ---
    # Window scale ~3% of min dimension; ensures normalization matches UI divider scale.
    r = int(max(9, round(0.03 * min(height, width))))  # radius proxy (odd-ish scale)
    sigma_blur = max(1.5, r / 6.0)  # convert to Gaussian sigma

    L = gray_enhanced.astype(np.float32)
    mu = cv2.GaussianBlur(L, (0, 0), sigma_blur)
    sq = cv2.GaussianBlur(L * L, (0, 0), sigma_blur)
    sigma = np.sqrt(np.maximum(sq - mu * mu, 1e-6))

    # clip sigma to avoid over-boosting flat/noisy patches
    sigma = np.clip(sigma, 5.0, 40.0)

    Z = (L - mu) / (sigma + 1e-6)  # local Z-score
    Z = np.tanh(0.5 * Z)  # soft-compress outliers
    Z = np.clip((Z * 80.0 + 128.0), 0, 255).astype(np.uint8)  # back to uint8 0..255

    # Step 4: Edge detection without fixed Canny thresholds
    # Use Scharr for better small-signal response
    gx = cv2.Scharr(Z, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(Z, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)

    # Percentile-based adaptive thresholds
    hi = np.percentile(mag, 96)
    lo = 0.4 * hi

    strong = (mag >= hi).astype(np.uint8) * 255
    weak = ((mag >= lo) & (mag < hi)).astype(np.uint8) * 255

    # Hysteresis: keep weak pixels connected to strong
    M = cv2.dilate(strong, np.ones((3, 3), np.uint8), 1)
    edges_adapt = cv2.bitwise_or(strong, cv2.bitwise_and(weak, M))

    # Step 5: Black-hat for dark-on-dark dividers
    k_bh = 21  # divider thickness scale
    bh = cv2.morphologyEx(Z, cv2.MORPH_BLACKHAT, np.ones((k_bh, k_bh), np.uint8))
    bh_mask = (bh >= np.percentile(bh, 92)).astype(np.uint8) * 255

    # Step 6: Filter for axis-aligned edges (prefer horizontal/vertical)
    angle = cv2.phase(gx, gy, angleInDegrees=True)
    axis = ((np.abs(((angle + 90) % 180) - 90) < 8) | (np.abs(angle) < 8)).astype(
        np.uint8
    )
    axis_mask = (axis * 255).astype(np.uint8)
    edges_axis = cv2.bitwise_and(edges_adapt, axis_mask)

    # Step 7: Combine contrast-normalized edges and black-hat
    edges = cv2.bitwise_or(edges_axis, bh_mask)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), 1)

    # Step 8: Probabilistic Hough Line Transform
    min_line_length = int(width * min_line_length_ratio)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,  # Lower threshold to catch subtle lines (CLAHE helps reduce noise)
        minLineLength=min_line_length,
        maxLineGap=max_gap,
    )

    # Step 9: Filter and merge near-horizontal/vertical lines
    horizontal_lines = []
    vertical_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx * dx + dy * dy)

            if length < 10:  # Skip very short lines
                continue

            angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)

            # Near-horizontal (within tolerance of 0° or 180°)
            if angle < angle_tolerance or angle > (180 - angle_tolerance):
                # Ensure line goes from left to right
                if x1 > x2:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                # Use average y position for consistency
                y_avg = (y1 + y2) // 2
                horizontal_lines.append((x1, y_avg, x2, y_avg, length))
            # Near-vertical (within tolerance of 90°)
            elif np.abs(angle - 90) < angle_tolerance:
                # Ensure line goes from top to bottom
                if y1 > y2:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                # Use average x position for consistency
                x_avg = (x1 + x2) // 2
                vertical_lines.append((x_avg, y1, x_avg, y2, length))

    # Merge collinear/overlapping segments
    def merge_lines(lines_list, is_horizontal):
        """Merge nearby parallel lines and cluster by position."""
        if not lines_list:
            return []

        # Sort by position (y for horizontal, x for vertical) then by start coordinate
        if is_horizontal:
            lines_sorted = sorted(lines_list, key=lambda l: (l[1], l[0]))
        else:
            lines_sorted = sorted(lines_list, key=lambda l: (l[0], l[1]))

        merged = []

        for line in lines_sorted:
            x1, y1, x2, y2, length = line

            # Try to merge with existing merged lines
            merged_with_existing = False
            for i, (mx1, my1, mx2, my2) in enumerate(merged):
                if is_horizontal:
                    # Check if lines are on approximately same row
                    if abs(y1 - my1) <= 5:
                        # Check if they overlap or are close in x
                        if x1 <= mx2 + 20 and x2 >= mx1 - 20:
                            # Merge by extending
                            new_x1 = min(mx1, x1)
                            new_x2 = max(mx2, x2)
                            merged[i] = (new_x1, my1, new_x2, my2)
                            merged_with_existing = True
                            break
                else:
                    # Check if lines are in approximately same column
                    if abs(x1 - mx1) <= 5:
                        # Check if they overlap or are close in y
                        if y1 <= my2 + 20 and y2 >= my1 - 20:
                            # Merge by extending
                            new_y1 = min(my1, y1)
                            new_y2 = max(my2, y2)
                            merged[i] = (mx1, new_y1, mx2, new_y2)
                            merged_with_existing = True
                            break

            if not merged_with_existing:
                merged.append((x1, y1, x2, y2))

        # Filter out very short merged lines
        if is_horizontal:
            merged = [
                line for line in merged if (line[2] - line[0]) >= min_line_length * 0.5
            ]
        else:
            merged = [line for line in merged if (line[3] - line[1]) >= height * 0.3]

        return merged

    h_lines = merge_lines(horizontal_lines, True)
    v_lines = merge_lines(vertical_lines, False)

    # Step 10: Create separator mask by dilating lines
    separator_mask = np.zeros((height, width), dtype=np.uint8)

    for x1, y1, x2, y2 in h_lines + v_lines:
        cv2.line(separator_mask, (x1, y1), (x2, y2), 255, thickness=1)

    # Dilate to make separators more prominent
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    separator_mask = cv2.dilate(separator_mask, kernel, iterations=1)

    return {
        "edges": edges,
        "horizontal_lines": horizontal_lines,
        "vertical_lines": vertical_lines,
        "h_lines": h_lines,
        "v_lines": v_lines,
        "separator_mask": separator_mask,
        "gray_processed": gray_processed,
        # Intermediate processing steps for visualization
        "intermediate_steps": {
            "text_removed": gray_processed,
            "clahe_enhanced": gray_enhanced,
            "z_normalized": Z,
            "gradient_magnitude": np.clip(mag / np.max(mag) * 255, 0, 255).astype(
                np.uint8
            ),
            "edges_adaptive": edges_adapt,
            "blackhat_mask": bh_mask,
            "axis_aligned_edges": edges_axis,
            "edges_final": edges,
        },
    }


def detect_whitespace_cuts(image_rgb, boxes, projection_gap=15):
    """
    Detect whitespace gaps using XY-cut projection analysis.
    Uses the original image (not text-removed) to find actual content gaps.

    Returns:
        - horizontal_cuts: Y positions of horizontal gaps
        - vertical_cuts: X positions of vertical gaps
        - capped_h_cuts: Horizontal cuts with x1, y, x2, y bounds
        - capped_v_cuts: Vertical cuts with x, y1, x, y2 bounds
    """
    height, width = image_rgb.shape[:2]

    # Convert to grayscale (use original image, not text-removed)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Apply Canny to get edges from original image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Create text mask from OCR boxes
    text_mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        if not box:
            continue
        pts = np.array([[int(pt[0]), int(pt[1])] for pt in box], dtype=np.int32)
        cv2.fillPoly(text_mask, [pts], 255)

    # Combine edges and text boxes
    foreground = cv2.bitwise_or(edges, text_mask)

    # Dilate to connect nearby elements
    dilation_kernel = np.ones((3, 3), np.uint8)
    foreground = cv2.dilate(foreground, dilation_kernel, iterations=1)

    # Compute projection profiles
    horizontal_projection = np.sum(foreground, axis=1)  # Sum across columns
    vertical_projection = np.sum(foreground, axis=0)  # Sum across rows

    # Find gaps (low density regions)
    def find_gaps(projection, min_gap_size):
        """Find positions where projection dips near zero."""
        gaps = []
        in_gap = False
        gap_start = 0
        threshold = np.max(projection) * 0.05  # 5% of max

        for i, val in enumerate(projection):
            if val < threshold:
                if not in_gap:
                    in_gap = True
                    gap_start = i
            else:
                if in_gap:
                    gap_size = i - gap_start
                    if gap_size >= min_gap_size:
                        gaps.append((gap_start + gap_size // 2))
                    in_gap = False

        return gaps

    horizontal_cuts = find_gaps(horizontal_projection, projection_gap)
    vertical_cuts = find_gaps(vertical_projection, projection_gap)

    # Cap the cut lines to span only the content regions
    def cap_horizontal_cut(y_pos, vertical_proj, threshold):
        """Find left and right bounds for a horizontal cut line."""
        # Find leftmost content
        x_start = 0
        for x in range(len(vertical_proj)):
            if vertical_proj[x] > threshold:
                x_start = x
                break

        # Find rightmost content
        x_end = len(vertical_proj) - 1
        for x in range(len(vertical_proj) - 1, -1, -1):
            if vertical_proj[x] > threshold:
                x_end = x
                break

        return x_start, x_end

    def cap_vertical_cut(x_pos, horizontal_proj, threshold):
        """Find top and bottom bounds for a vertical cut line."""
        # Find topmost content
        y_start = 0
        for y in range(len(horizontal_proj)):
            if horizontal_proj[y] > threshold:
                y_start = y
                break

        # Find bottommost content
        y_end = len(horizontal_proj) - 1
        for y in range(len(horizontal_proj) - 1, -1, -1):
            if horizontal_proj[y] > threshold:
                y_end = y
                break

        return y_start, y_end

    # Cap cut lines to content bounds
    threshold = (
        np.max(np.concatenate([horizontal_projection, vertical_projection])) * 0.05
    )
    capped_h_cuts = []
    for y in horizontal_cuts:
        x_start, x_end = cap_horizontal_cut(y, vertical_projection, threshold)
        capped_h_cuts.append((x_start, y, x_end, y))

    capped_v_cuts = []
    for x in vertical_cuts:
        y_start, y_end = cap_vertical_cut(x, horizontal_projection, threshold)
        capped_v_cuts.append((x, y_start, x, y_end))

    return {
        "horizontal_cuts": horizontal_cuts,
        "vertical_cuts": vertical_cuts,
        "capped_h_cuts": capped_h_cuts,
        "capped_v_cuts": capped_v_cuts,
    }


def detect_lines_and_cuts(
    image_rgb,
    boxes,
    min_line_length_ratio=0.16,
    max_gap=8,
    angle_tolerance=2.0,
    dilation_size=4,
    projection_gap=15,
    detect_image_regions=True,
):
    """
    Detect structural lines, whitespace cuts, and image regions.
    Combines Hough line detection with XY-cut projection analysis and image detection.

    Returns:
        - edges: Canny edge image
        - lines: Detected line segments
        - separator_mask: Binary mask of separator lines
        - cut_lines: Dict with 'horizontal' and 'vertical' cut positions
        - capped_cut_lines: Dict with capped cut line coordinates
        - image_boxes: List of detected image/photo regions
        - visualization: RGB image showing the results
    """
    # Detect structural lines using text-removed image
    line_results = detect_structural_lines(
        image_rgb, boxes, min_line_length_ratio, max_gap, angle_tolerance, dilation_size
    )

    # Detect whitespace cuts using original image
    cut_results = detect_whitespace_cuts(image_rgb, boxes, projection_gap)

    # Detect image/photo regions
    image_results = None
    if detect_image_regions:
        image_results = detect_images(image_rgb, boxes)

    # Create visualization
    vis_image = cv2.cvtColor(line_results["gray_processed"], cv2.COLOR_GRAY2RGB)

    # Draw edges in yellow (faint)
    edge_overlay = vis_image.copy()
    edge_overlay[line_results["edges"] > 0] = [255, 255, 0]
    vis_image = cv2.addWeighted(vis_image, 0.7, edge_overlay, 0.3, 0)

    # Draw raw Hough candidates in orange (before merging)
    for x1, y1, x2, y2, _ in (
        line_results["horizontal_lines"] + line_results["vertical_lines"]
    ):
        cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 165, 0), 1)

    # Draw merged lines in green (thicker)
    for x1, y1, x2, y2 in line_results["h_lines"] + line_results["v_lines"]:
        cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Draw separator mask in magenta (semi-transparent)
    separator_overlay = vis_image.copy()
    separator_overlay[line_results["separator_mask"] > 0] = [255, 0, 255]
    vis_image = cv2.addWeighted(vis_image, 0.8, separator_overlay, 0.2, 0)

    # Draw capped cut lines in cyan
    for x1, y, x2, _ in cut_results["capped_h_cuts"]:
        cv2.line(vis_image, (int(x1), int(y)), (int(x2), int(y)), (0, 255, 255), 1)
    for x, y1, _, y2 in cut_results["capped_v_cuts"]:
        cv2.line(vis_image, (int(x), int(y1)), (int(x), int(y2)), (0, 255, 255), 1)

    # Draw detected image boxes in blue
    if image_results:
        for x, y, w, h in image_results["image_boxes"]:
            cv2.rectangle(
                vis_image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2
            )
            # Add label
            cv2.putText(
                vis_image,
                "IMAGE",
                (int(x + 5), int(y + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

    # Convert numpy types to Python types for JSON serialization
    def convert_to_python_types(obj):
        """Recursively convert numpy types to Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_python_types(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: convert_to_python_types(value) for key, value in obj.items()}
        return obj

    result = {
        "edges": line_results["edges"],
        "lines": convert_to_python_types(
            {"horizontal": line_results["h_lines"], "vertical": line_results["v_lines"]}
        ),
        "separator_mask": line_results["separator_mask"],
        "cut_lines": convert_to_python_types(
            {
                "horizontal": cut_results["horizontal_cuts"],
                "vertical": cut_results["vertical_cuts"],
            }
        ),
        "capped_cut_lines": convert_to_python_types(
            {
                "horizontal": cut_results["capped_h_cuts"],
                "vertical": cut_results["capped_v_cuts"],
            }
        ),
        "visualization": Image.fromarray(vis_image),
        "intermediate_steps": line_results["intermediate_steps"],
    }

    if image_results:
        result["image_boxes"] = convert_to_python_types(image_results["image_boxes"])
        result["image_detection_steps"] = image_results["intermediate_steps"]

    return result
