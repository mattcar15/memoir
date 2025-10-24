"""
Element segmentation functions for detecting UI elements and structure.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from utils import box_bounds


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


def detect_lines_and_cuts(
    image_rgb,
    boxes,
    min_line_length_ratio=0.16,
    max_gap=8,
    angle_tolerance=2.0,
    dilation_size=4,
    projection_gap=15,
):
    """
    Detect structural lines and whitespace cuts using Hough transform and XY-cut.

    Returns:
        - edges: Canny edge image
        - lines: Detected line segments
        - separator_mask: Binary mask of separator lines
        - cut_lines: Dict with 'horizontal' and 'vertical' cut positions
        - visualization: RGB image showing the results
    """
    height, width = image_rgb.shape[:2]

    # Step 1: Create text-removed BW image
    gray_processed = create_text_removed_image(image_rgb, boxes)

    # Step 2: Enhanced edge detection
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray_processed, 9, 75, 75)

    # Use adaptive thresholding to find strong edges
    binary = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Canny edge detection with tighter thresholds
    edges = cv2.Canny(filtered, 50, 150)

    # Step 3: Probabilistic Hough Line Transform with stricter parameters
    min_line_length = int(width * min_line_length_ratio)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,  # Increased threshold for stronger lines
        minLineLength=min_line_length,
        maxLineGap=max_gap,
    )

    # Step 4: Filter and merge near-horizontal/vertical lines
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

    # Step 5: Create separator mask by dilating lines
    separator_mask = np.zeros((height, width), dtype=np.uint8)

    for x1, y1, x2, y2 in h_lines + v_lines:
        cv2.line(separator_mask, (x1, y1), (x2, y2), 255, thickness=1)

    # Dilate to make separators more prominent
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    separator_mask = cv2.dilate(separator_mask, kernel, iterations=1)

    # Step 6: XY-cut - find whitespace gaps
    # Create binary foreground: edges OR text boxes
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

    # Step 7: Create visualization
    vis_image = cv2.cvtColor(gray_processed, cv2.COLOR_GRAY2RGB)

    # Draw edges in yellow (faint)
    edge_overlay = vis_image.copy()
    edge_overlay[edges > 0] = [255, 255, 0]
    vis_image = cv2.addWeighted(vis_image, 0.7, edge_overlay, 0.3, 0)

    # Draw raw Hough candidates in orange (before merging)
    for x1, y1, x2, y2, _ in horizontal_lines + vertical_lines:
        cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 165, 0), 1)

    # Draw merged lines in green (thicker)
    for x1, y1, x2, y2 in h_lines + v_lines:
        cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Draw separator mask in magenta (semi-transparent)
    separator_overlay = vis_image.copy()
    separator_overlay[separator_mask > 0] = [255, 0, 255]
    vis_image = cv2.addWeighted(vis_image, 0.8, separator_overlay, 0.2, 0)

    # Draw cut lines in cyan
    for y in horizontal_cuts:
        cv2.line(vis_image, (0, int(y)), (width, int(y)), (0, 255, 255), 1)
    for x in vertical_cuts:
        cv2.line(vis_image, (int(x), 0), (int(x), height), (0, 255, 255), 1)

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

    return {
        "edges": edges,
        "lines": convert_to_python_types({"horizontal": h_lines, "vertical": v_lines}),
        "separator_mask": separator_mask,
        "cut_lines": convert_to_python_types(
            {"horizontal": horizontal_cuts, "vertical": vertical_cuts}
        ),
        "visualization": Image.fromarray(vis_image),
    }
