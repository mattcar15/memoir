"""
Menu detection using heuristic-based scoring on screen regions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils import box_bounds, x_center, y_center


MenuResult = Dict[str, object]

# Initial strip coverage (matches the prior heuristic defaults)
INITIAL_STRIP_RATIOS = {
    "top": (0.0, 0.0, 1.0, 0.08),  # (x0_ratio, y0_ratio, x1_ratio, y1_ratio)
    "left": (0.0, 0.0, 0.18, 1.0),
    "right": (0.82, 0.0, 1.0, 1.0),
}

# Thresholds and heuristics
T_DETECT = 5.0
T_UNCERTAIN = 3.0
REFINE_OVERLAP = 0.6
SUBSTANTIAL_MIN_CHARS = 4
SUBSTANTIAL_MIN_WORDS = 2
PADDING_RATIO = 0.015
TOP_MAX_HEIGHT_RATIO = 0.2
DIVIDER_HORIZONTAL_COVERAGE = 0.65
DIVIDER_VERTICAL_COVERAGE = 0.6
DIVIDER_EDGE_TOL = 0.1  # acceptable fractional deviation from strip edge
CONFIDENCE_GAP_THRESHOLD = 1.5
DIVIDER_BONUS = 0.7
SIDE_MIN_HEIGHT_RATIO = 0.75
SOBEL_EDGE_THRESHOLD = 40


def _clamp_rect(rect: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    return int(x0), int(y0), int(x1), int(y1)


def _rect_area(rect: Tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = rect
    return max(0, x1 - x0) * max(0, y1 - y0)


def _box_area(box) -> float:
    x0, y0, x1, y1 = box_bounds(box)
    return max(0, x1 - x0) * max(0, y1 - y0)


def _intersection(rect: Tuple[int, int, int, int], box) -> Tuple[int, int, int, int]:
    rx0, ry0, rx1, ry1 = rect
    bx0, by0, bx1, by1 = box_bounds(box)
    ix0 = max(rx0, bx0)
    iy0 = max(ry0, by0)
    ix1 = min(rx1, bx1)
    iy1 = min(ry1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0, 0, 0, 0
    return ix0, iy0, ix1, iy1


def _overlap_ratio(rect: Tuple[int, int, int, int], box) -> float:
    inter = _intersection(rect, box)
    ix0, iy0, ix1, iy1 = inter
    inter_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter_area <= 0:
        return 0.0
    box_area = _box_area(box)
    if box_area == 0:
        return 0.0
    return inter_area / box_area


def _is_substantial_text(text: Optional[str]) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) >= SUBSTANTIAL_MIN_CHARS:
        return True
    words = [tok for tok in stripped.split() if tok]
    return len(words) >= SUBSTANTIAL_MIN_WORDS


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
    valid = 0
    for i in box_indices:
        if i >= len(texts):
            continue
        text = texts[i]
        if not text:
            continue
        valid += 1
        if len(text.strip()) <= 2:
            short_count += 1

    if valid == 0:
        return 0.0
    return short_count / valid


def spill_penalty(boxes, box_indices, rect):
    """
    Calculate penalty for boxes that extend beyond the rectangle.
    Returns value 0-1 where 0 = no spill, 1 = significant spill.
    """
    if not box_indices:
        return 0.0

    rx0, ry0, rx1, ry1 = rect
    rect_area = (rx1 - rx0) * (ry1 - ry0)
    if rect_area == 0:
        return 0.0

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


def _score_region(
    name: str,
    rect: Tuple[int, int, int, int],
    boxes,
    texts,
    divider_found: bool = False,
) -> Tuple[Optional[str], float, List[int]]:
    all_indices = filter_boxes_in_rect(boxes, rect)
    main_indices = [
        i
        for i in all_indices
        if i < len(texts) and _is_substantial_text(texts[i])
    ]

    if len(main_indices) < 3:
        return None, 0.0, main_indices

    align_axis = "x" if name != "top" else "y"
    space_axis = "y" if name != "top" else "x"

    align = alignment_strength(boxes, main_indices, align_axis)
    space = spacing_regular(boxes, main_indices, space_axis)
    short_rate = short_token_rate(boxes, main_indices, texts)
    spill = spill_penalty(boxes, main_indices, rect)

    rect_area = max(1.0, _rect_area(rect))
    density = len(main_indices) / (rect_area / 10000.0)

    score = 1.6 * density + 2.0 * align + 1.2 * space + 0.8 * short_rate - 1.5 * spill
    if divider_found:
        score += DIVIDER_BONUS

    if score >= T_DETECT:
        status = "menu"
    elif score >= T_UNCERTAIN:
        status = "maybe"
    else:
        status = None

    return status, score, main_indices


def _refine_rect(
    name: str,
    rect: Tuple[int, int, int, int],
    boxes,
    texts,
    width: int,
    height: int,
) -> Tuple[Tuple[int, int, int, int], List[int]]:
    indices = filter_boxes_in_rect(boxes, rect)
    substantial = [
        i
        for i in indices
        if i < len(texts)
        and _is_substantial_text(texts[i])
        and _overlap_ratio(rect, boxes[i]) >= REFINE_OVERLAP
    ]

    if not substantial:
        return rect, []

    xs0 = []
    ys0 = []
    xs1 = []
    ys1 = []
    for idx in substantial:
        x0, y0, x1, y1 = box_bounds(boxes[idx])
        xs0.append(x0)
        ys0.append(y0)
        xs1.append(x1)
        ys1.append(y1)

    pad_x = int(round(width * PADDING_RATIO))
    pad_y = int(round(height * PADDING_RATIO))

    if name == "top":
        refined = (
            0,
            0,
            width,
            min(height, max(ys1) + pad_y),
        )
    elif name == "left":
        refined = (
            0,
            max(0, min(ys0) - pad_y),
            min(width, max(xs1) + pad_x),
            min(height, max(ys1) + pad_y),
        )
    else:  # right
        refined = (
            max(0, min(xs0) - pad_x),
            max(0, min(ys0) - pad_y),
            width,
            min(height, max(ys1) + pad_y),
        )

    # Enforce top height cap
    if name == "top":
        max_top = int(round(height * TOP_MAX_HEIGHT_RATIO))
        refined = (refined[0], refined[1], refined[2], min(refined[3], max_top))

    refined = _clamp_rect(refined, width, height)
    return refined, substantial


def _find_divider(
    name: str,
    rect: Tuple[int, int, int, int],
    gray_image: np.ndarray,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    x0, y0, x1, y1 = rect
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None, None

    region = gray_image[y0:y1, x0:x1]
    if region.size == 0:
        return None, None

    if name == "top":
        sobel = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
    else:
        sobel = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)

    edges = cv2.convertScaleAbs(sobel)
    _, binary = cv2.threshold(edges, SOBEL_EDGE_THRESHOLD, 255, cv2.THRESH_BINARY)

    if name == "top":
        coverage = binary.sum(axis=1) / 255.0
        if coverage.size == 0:
            return None, binary
        idx = int(np.argmax(coverage))
        span = x1 - x0
        if span <= 0:
            return None, binary
        fraction = coverage[idx] / span
        if fraction >= DIVIDER_HORIZONTAL_COVERAGE and idx >= int(0.5 * coverage.size):
            line_y = y0 + idx
            return (x0, line_y, x1, line_y), binary
        return None, binary

    # Vertical divider for side menus
    coverage = binary.sum(axis=0) / 255.0
    if coverage.size == 0:
        return None, binary
    idx = int(np.argmax(coverage))
    span = y1 - y0
    if span <= 0:
        return None, binary
    fraction = coverage[idx] / span
    if fraction < DIVIDER_VERTICAL_COVERAGE:
        return None, binary

    width = x1 - x0
    expected_edge = width - 1 if name == "left" else 0
    tolerance = max(2, int(round(width * DIVIDER_EDGE_TOL)))
    if abs(idx - expected_edge) > tolerance:
        return None, binary

    line_x = x0 + idx
    return (line_x, y0, line_x, y1), binary


def detect_menus(image_rgb, boxes, texts) -> List[MenuResult]:
    """
    Detect menu regions (top, left, right strips) using adaptive sizing.

    Returns a list of dicts with initial and final region metadata.
    """
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_for_edges = gray.copy()

    for box in boxes:
        if not box:
            continue
        bx0, by0, bx1, by1 = box_bounds(box)
        x0, y0 = max(0, int(round(bx0))), max(0, int(round(by0)))
        x1, y1 = min(width, int(round(bx1))), min(height, int(round(by1)))
        if x1 - x0 <= 1 or y1 - y0 <= 1:
            continue
        region = gray_for_edges[y0:y1, x0:x1]
        if region.size == 0:
            continue
        kernel_w = 5 if (x1 - x0) >= 5 else 3
        kernel_h = 5 if (y1 - y0) >= 5 else 3
        if kernel_w < 3 or kernel_h < 3:
            continue
        blurred = cv2.GaussianBlur(region, (kernel_w, kernel_h), 0)
        gray_for_edges[y0:y1, x0:x1] = blurred

    results: List[MenuResult] = []

    for name, ratios in INITIAL_STRIP_RATIOS.items():
        x0 = int(round(ratios[0] * width))
        y0 = int(round(ratios[1] * height))
        x1 = int(round(ratios[2] * width))
        y1 = int(round(ratios[3] * height))
        initial_rect = _clamp_rect((x0, y0, x1, y1), width, height)

        initial_status, initial_score, initial_indices = _score_region(
            name, initial_rect, boxes, texts
        )

        refined_rect, substantial_indices = _refine_rect(
            name, initial_rect, boxes, texts, width, height
        )

        note = None
        divider_line: Optional[Tuple[int, int, int, int]] = None
        divider_edges: Optional[np.ndarray] = None
        final_status = None
        final_score = 0.0
        final_indices: List[int] = []

        side_height_ok = True
        if name in ("left", "right"):
            strip_height = refined_rect[3] - refined_rect[1]
            side_height_ok = strip_height >= height * SIDE_MIN_HEIGHT_RATIO
            if not side_height_ok:
                note = "insufficient_height"

        if refined_rect[2] - refined_rect[0] > 0 and refined_rect[3] - refined_rect[1] > 0:
            divider_line, divider_edges = _find_divider(name, refined_rect, gray_for_edges)

        if not substantial_indices or not side_height_ok:
            divider_line = None

        if side_height_ok:
            final_status, final_score, final_indices = _score_region(
                name,
                refined_rect,
                boxes,
                texts,
                divider_found=divider_line is not None,
            )

        results.append(
            {
                "name": name,
                "initial_rect": initial_rect,
                "initial_status": initial_status,
                "initial_score": initial_score,
                "initial_indices": initial_indices,
                "rect": refined_rect,
                "status": final_status,
                "score": final_score,
                "indices": final_indices,
                "divider": divider_line,
                "divider_edges": divider_edges,
                "notes": note,
            }
        )

    # Enforce combined width constraint for side menus
    left = next((r for r in results if r["name"] == "left"), None)
    right = next((r for r in results if r["name"] == "right"), None)

    if left and right and left.get("status") and right.get("status"):
        left_width = left["rect"][2] - left["rect"][0]
        right_width = right["rect"][2] - right["rect"][0]
        if left_width + right_width > 0.5 * width:
            gap = abs(float(left["score"]) - float(right["score"]))
            if gap >= CONFIDENCE_GAP_THRESHOLD:
                if left["score"] < right["score"]:
                    left["notes"] = "dropped_due_to_width"
                    left["status"] = None
                else:
                    right["notes"] = "dropped_due_to_width"
                    right["status"] = None

    return results
