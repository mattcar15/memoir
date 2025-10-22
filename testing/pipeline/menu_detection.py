"""
Menu detection using heuristic-based scoring on screen regions.
"""

import numpy as np
from utils import box_bounds, x_center, y_center


def filter_boxes_in_rect(boxes, rect):
    """
    Find boxes that overlap with the given rectangle.
    rect: (x0, y0, x1, y1)
    Returns list of box indices.
    """
    rx0, ry0, rx1, ry1 = rect
    indices = []
    for i, box in enumerate(boxes):
        if not box:
            continue
        bx0, by0, bx1, by1 = box_bounds(box)
        # Check if box overlaps with rect
        if bx1 >= rx0 and bx0 <= rx1 and by1 >= ry0 and by0 <= ry1:
            indices.append(i)
    return indices


def alignment_strength(boxes, box_indices, axis):
    """
    Measure alignment along an axis using standard deviation.
    Lower std dev = better alignment → higher score.
    axis: "x" (horizontal alignment) or "y" (vertical alignment)
    Returns normalized score (higher = more aligned).
    """
    if len(box_indices) < 2:
        return 0.0

    if axis == "x":
        positions = [x_center(boxes[i]) for i in box_indices]
    else:  # "y"
        positions = [y_center(boxes[i]) for i in box_indices]

    std = np.std(positions)
    # Normalize: lower std is better, convert to score 0-1
    # For typical screen widths ~1000-2000px, std < 50 is well-aligned
    score = max(0.0, 1.0 - std / 100.0)
    return score


def spacing_regular(boxes, box_indices, axis):
    """
    Measure spacing regularity using coefficient of variation of gaps.
    axis: "x" (measure vertical gaps) or "y" (measure horizontal gaps)
    Returns normalized score (higher = more regular).
    """
    if len(box_indices) < 3:
        return 0.0

    # Sort by position along the measurement axis
    if axis == "y":
        # Measuring horizontal spacing (left to right)
        sorted_indices = sorted(box_indices, key=lambda i: x_center(boxes[i]))
        positions = [x_center(boxes[i]) for i in sorted_indices]
    else:  # "x"
        # Measuring vertical spacing (top to bottom)
        sorted_indices = sorted(box_indices, key=lambda i: y_center(boxes[i]))
        positions = [y_center(boxes[i]) for i in sorted_indices]

    # Calculate gaps between consecutive elements
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]

    if not gaps or np.mean(gaps) == 0:
        return 0.0

    # Coefficient of variation: std/mean
    cv = np.std(gaps) / np.mean(gaps)
    # Lower CV = more regular → higher score
    score = max(0.0, 1.0 - cv)
    return score


def short_token_rate(boxes, box_indices, texts):
    """
    Calculate fraction of short (1-2 character) tokens.
    Menu items often have short labels (icons, initials, etc.)
    """
    if not box_indices:
        return 0.0

    short_count = 0
    for i in box_indices:
        text = texts[i] if i < len(texts) else ""
        if text and len(text.strip()) <= 2:
            short_count += 1

    return short_count / len(box_indices)


def spill_penalty(boxes, box_indices, rect):
    """
    Calculate penalty for boxes that extend beyond the rectangle.
    Returns value 0-1 where 0 = no spill, 1 = significant spill.
    """
    if not box_indices:
        return 0.0

    rx0, ry0, rx1, ry1 = rect
    rect_area = (rx1 - rx0) * (ry1 - ry0)

    total_spill = 0.0
    for i in box_indices:
        bx0, by0, bx1, by1 = box_bounds(boxes[i])
        box_area = (bx1 - bx0) * (by1 - by0)

        # Calculate intersection area
        ix0 = max(bx0, rx0)
        iy0 = max(by0, ry0)
        ix1 = min(bx1, rx1)
        iy1 = min(by1, ry1)

        if ix1 > ix0 and iy1 > iy0:
            intersection = (ix1 - ix0) * (iy1 - iy0)
            spill = max(0, box_area - intersection)
            total_spill += spill

    # Normalize by rect area
    return min(1.0, total_spill / rect_area)


def detect_menus(boxes, texts, img_size):
    """
    Detect menu regions (top, left, right strips) using heuristic scoring.

    Returns list of tuples: (name, status, score)
    where status is "menu", "maybe", or None
    """
    W, H = img_size

    # Define strip regions
    strips = {
        "top": (0, 0, W, int(0.16 * H)),
        "left": (0, 0, int(0.18 * W), H),
        "right": (int(0.82 * W), 0, W, H),
    }

    # Thresholds
    T_DETECT = 5.0
    T_UNCERTAIN = 3.0

    results = []

    for name, rect in strips.items():
        # Find boxes in this strip
        box_indices = filter_boxes_in_rect(boxes, rect)

        if len(box_indices) < 3:
            results.append((name, None, 0.0))
            continue

        # Calculate features
        align_axis = "x" if name != "top" else "y"
        space_axis = "y" if name != "top" else "x"

        align = alignment_strength(boxes, box_indices, align_axis)
        space = spacing_regular(boxes, box_indices, space_axis)
        short_rate = short_token_rate(boxes, box_indices, texts)
        spill = spill_penalty(boxes, box_indices, rect)

        # Density
        rect_area = (rect[2] - rect[0]) * (rect[3] - rect[1])
        dens = len(box_indices) / (rect_area / 10000.0)  # normalize to 100x100 blocks

        # Score calculation
        score = 1.6 * dens + 2.0 * align + 1.2 * space + 0.8 * short_rate - 1.5 * spill

        # Classify
        if score >= T_DETECT:
            status = "menu"
        elif score >= T_UNCERTAIN:
            status = "maybe"
        else:
            status = None

        results.append((name, status, score))

    return results
