"""
Menu detection utilities for sidebar filtering.
"""

from typing import Dict, List, Tuple

from PIL import Image

# Match the pipeline's resize behavior
MAX_LONG_SIDE = 1080


def adjust_menu_rectangles_for_display(
    menu_results: List[Dict], width: int, height: int
) -> List[Dict]:
    """
    Adjust menu rectangles to span full height/width aligned to dividers.
    Adapted from main.py in the pipeline.
    """

    def _clamp(a, lo, hi):
        return max(lo, min(hi, a))

    adjusted = []
    for region in menu_results or []:
        r = dict(region)
        name = r.get("name")
        status = r.get("status")
        base_rect = r.get("rect_aligned") or r.get("rect")

        if status in ("menu", "maybe") and base_rect:
            x0, y0, x1, y1 = [int(round(v)) for v in base_rect]
            divider = r.get("divider")

            if name == "left":
                target_x = None
                if divider:
                    target_x = int(round(min(divider[0], divider[2])))
                else:
                    target_x = x1
                target_x = _clamp(target_x, 0, width)
                new_rect = (x0, 0, max(x0 + 1, target_x), height)
            elif name == "right":
                target_x = None
                if divider:
                    target_x = int(round(max(divider[0], divider[2])))
                else:
                    target_x = x0
                target_x = _clamp(target_x, 0, width)
                new_rect = (min(target_x, x1 - 1), 0, x1, height)
            elif name == "top":
                target_y = None
                if divider:
                    target_y = int(round(max(divider[1], divider[3])))
                else:
                    target_y = y1
                target_y = _clamp(target_y, 0, height)
                new_rect = (0, 0, width, max(1, target_y))
            else:
                new_rect = base_rect

            nx0, ny0, nx1, ny1 = new_rect
            if nx1 <= nx0:
                nx1 = nx0 + 1
            if ny1 <= ny0:
                ny1 = ny0 + 1
            r["rect_display"] = (int(nx0), int(ny0), int(nx1), int(ny1))

        adjusted.append(r)

    return adjusted


def resize_image_for_detection(
    image: Image.Image, max_long_side: int = MAX_LONG_SIDE
) -> Tuple[Image.Image, float]:
    """
    Resize image proportionally if it exceeds max_long_side.
    Returns (resized_image, scale_factor).
    """
    width, height = image.size
    max_dim = max(width, height)

    if max_dim > max_long_side:
        scale = max_long_side / max_dim
        new_size = (int(round(width * scale)), int(round(height * scale)))
        resized = image.resize(new_size, Image.LANCZOS)
        return resized, scale

    return image, 1.0
