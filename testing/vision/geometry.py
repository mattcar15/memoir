"""
Geometry utilities for coordinate conversion and box operations.
"""

from typing import List, Tuple


def vision_box_to_polygon(
    norm_box: Tuple[float, float, float, float], width: int, height: int
) -> List[List[float]]:
    """
    Convert Vision's normalized box (origin bottom-left) to a 4-point polygon in pixel coordinates.

    Vision format: (x, y, w, h) where origin is bottom-left, values 0-1
    Pipeline format: [[x0,y0], [x1,y0], [x1,y1], [x0,y1]] in pixel coordinates
    """
    x, y, w, h = norm_box
    # Convert to pixel coordinates (flip Y axis since Vision uses bottom-left origin)
    px0 = x * width
    py0 = (1.0 - y - h) * height  # top-left Y
    px1 = (x + w) * width
    py1 = (1.0 - y) * height  # bottom-left Y

    return [[px0, py0], [px1, py0], [px1, py1], [px0, py1]]


def box_bounds(box: List[List[float]]) -> Tuple[float, float, float, float]:
    """Get bounding box (x0, y0, x1, y1) from a 4-point polygon."""
    xs = [pt[0] for pt in box]
    ys = [pt[1] for pt in box]
    return min(xs), min(ys), max(xs), max(ys)


def box_in_region(box: List[List[float]], rect: Tuple[int, int, int, int]) -> bool:
    """Check if a box overlaps with a rectangle region."""
    rx0, ry0, rx1, ry1 = rect
    bx0, by0, bx1, by1 = box_bounds(box)
    return bx1 >= rx0 and bx0 <= rx1 and by1 >= ry0 and by0 <= ry1
