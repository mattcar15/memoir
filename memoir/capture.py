"""
Screenshot capture and image processing functionality.
"""

import io
import time
from PIL import Image, ImageGrab

from .config import RESOLUTION_TIERS


def capture_screenshot(resolution_tier="detailed"):
    """
    Capture a screenshot and scale it based on the resolution tier.

    Args:
        resolution_tier: One of 'efficient', 'balanced', or 'detailed'

    Returns:
        Tuple of (PIL Image object (scaled), stats dict) or (None, None) if capture fails
    """
    try:
        start_time = time.time()

        # Capture the entire screen
        screenshot = ImageGrab.grab()
        capture_time = time.time() - start_time

        # Get target resolution
        target_width, target_height = RESOLUTION_TIERS.get(
            resolution_tier, RESOLUTION_TIERS["detailed"]
        )

        # Calculate scaling to maintain aspect ratio
        original_width, original_height = screenshot.size
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        scale_ratio = min(width_ratio, height_ratio)

        # Calculate new dimensions
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)

        # Resize image
        resize_start = time.time()
        scaled_screenshot = screenshot.resize(
            (new_width, new_height), resample=Image.LANCZOS
        )
        resize_time = time.time() - resize_start

        # Calculate image size in memory
        img_buffer = io.BytesIO()
        scaled_screenshot.save(img_buffer, format="PNG")
        image_size_bytes = len(img_buffer.getvalue())

        stats = {
            "original_resolution": f"{original_width}x{original_height}",
            "scaled_resolution": f"{new_width}x{new_height}",
            "capture_time_seconds": round(capture_time, 3),
            "resize_time_seconds": round(resize_time, 3),
            "image_size_bytes": image_size_bytes,
            "image_size_kb": round(image_size_bytes / 1024, 2),
        }

        return scaled_screenshot, stats

    except Exception as e:
        print(f"❌ Error capturing screenshot: {e}")
        return None, None
