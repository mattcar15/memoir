"""
Utility functions for image loading and geometric operations.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import cv2


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MAX_LONG_SIDE = 1080  # cap longest edge at ~HD resolution


def iter_image_paths(directories, extensions=SUPPORTED_EXTENSIONS):
    """Iterate over image files in given directories."""
    seen = set()
    for directory in directories:
        directory = Path(directory).resolve()
        if not directory.exists():
            print(f"Skipping missing directory: {directory}")
            continue
        for child in sorted(directory.iterdir()):
            if child.is_file() and child.suffix.lower() in extensions:
                if child not in seen:
                    seen.add(child)
                    yield child


def load_and_resize_image(image_path: Path, max_long_side=MAX_LONG_SIDE):
    """Load image and resize proportionally using PIL (higher quality)"""
    # Load with PIL for better resize quality
    pil_img_original = Image.open(image_path)

    # Convert RGBA to RGB if needed
    if pil_img_original.mode == "RGBA":
        pil_img_original = pil_img_original.convert("RGB")
    elif pil_img_original.mode != "RGB":
        pil_img_original = pil_img_original.convert("RGB")

    original_size = pil_img_original.size  # (width, height)
    original_rgb = np.array(pil_img_original)

    # Calculate if we need to resize
    max_dim = max(original_size)

    if max_dim > max_long_side:
        # Use PIL's resize for proportional scaling
        scale = max_long_side / max_dim
        new_size = (
            int(round(original_size[0] * scale)),
            int(round(original_size[1] * scale)),
        )
        # Use LANCZOS for high-quality downsampling
        pil_img_resized = pil_img_original.resize(new_size, Image.LANCZOS)
        resized_size = pil_img_resized.size

        # Convert resized to numpy
        image_rgb = np.array(pil_img_resized)
    else:
        scale = 1.0
        resized_size = original_size
        image_rgb = original_rgb

    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    return image_bgr, image_rgb, original_rgb, original_size, resized_size, scale


# Geometry helpers
def box_bounds(box):
    """Get bounding box (min_x, min_y, max_x, max_y) from polygon points."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def poly_height(box):
    """Get height of bounding box."""
    x0, y0, x1, y1 = box_bounds(box)
    return y1 - y0


def x_center(box):
    """Get horizontal center of bounding box."""
    x0, _, x1, _ = box_bounds(box)
    return (x0 + x1) / 2.0


def y_top(box):
    """Get top y-coordinate of bounding box."""
    _, y0, _, _ = box_bounds(box)
    return y0


def y_center(box):
    """Get vertical center of bounding box."""
    _, y0, _, y1 = box_bounds(box)
    return (y0 + y1) / 2.0
