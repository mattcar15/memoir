"""
Screenshot capture and image processing functionality.
"""

import io
import time
import pyautogui
from PIL import Image, ImageGrab

from .config import RESOLUTION_TIERS


def get_cursor_position():
    """
    Get the current cursor position.

    Returns:
        Tuple of (x, y) coordinates or (None, None) if failed
    """
    try:
        return pyautogui.position()
    except Exception as e:
        print(f"❌ Error getting cursor position: {e}")
        return None, None


def get_screen_bounds():
    """
    Get the bounds of all available screens.

    Returns:
        List of tuples (x, y, width, height) for each screen
    """
    try:
        # Get all screen information
        screens = []

        # Get primary screen bounds
        primary_bounds = pyautogui.size()
        screens.append((0, 0, primary_bounds[0], primary_bounds[1]))

        # For multi-monitor setups, we need to check if there are additional screens
        # This is a simplified approach - on macOS, we can get more detailed info
        try:
            import Quartz

            # Get all display information on macOS
            displays = Quartz.CGGetActiveDisplayList(10, None, None)[1]
            for display_id in displays:
                bounds = Quartz.CGDisplayBounds(display_id)
                screens.append(
                    (
                        int(bounds.origin.x),
                        int(bounds.origin.y),
                        int(bounds.size.width),
                        int(bounds.size.height),
                    )
                )
        except ImportError:
            # Fallback: assume single screen if Quartz not available
            pass

        return screens
    except Exception as e:
        print(f"❌ Error getting screen bounds: {e}")
        # Fallback to primary screen
        try:
            size = pyautogui.size()
            return [(0, 0, size[0], size[1])]
        except:
            return [(0, 0, 1920, 1080)]  # Default fallback


def get_cursor_screen_bounds():
    """
    Get the bounds of the screen that contains the cursor.

    Returns:
        Tuple of (x, y, width, height) for the cursor's screen, or None if failed
    """
    cursor_x, cursor_y = get_cursor_position()
    if cursor_x is None or cursor_y is None:
        return None

    screens = get_screen_bounds()

    # Find which screen contains the cursor
    for screen_x, screen_y, screen_width, screen_height in screens:
        if (
            screen_x <= cursor_x < screen_x + screen_width
            and screen_y <= cursor_y < screen_y + screen_height
        ):
            return (screen_x, screen_y, screen_width, screen_height)

    # If cursor is not found in any screen, return the first screen as fallback
    return screens[0] if screens else None


def capture_screenshot(resolution_tier="detailed"):
    """
    Capture a screenshot of the screen containing the cursor and scale it based on the resolution tier.

    Args:
        resolution_tier: One of 'efficient', 'balanced', or 'detailed'

    Returns:
        Tuple of (PIL Image object (scaled), stats dict) or (None, None) if capture fails
    """
    try:
        start_time = time.time()

        # Get the bounds of the screen containing the cursor
        screen_bounds = get_cursor_screen_bounds()
        if screen_bounds is None:
            print(
                "❌ Could not determine cursor screen, falling back to primary screen"
            )
            screenshot = ImageGrab.grab()
        else:
            screen_x, screen_y, screen_width, screen_height = screen_bounds
            # Capture the specific screen containing the cursor
            screenshot = ImageGrab.grab(
                bbox=(
                    screen_x,
                    screen_y,
                    screen_x + screen_width,
                    screen_y + screen_height,
                )
            )

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

        # Get cursor position for stats
        cursor_x, cursor_y = get_cursor_position()
        cursor_info = f"({cursor_x}, {cursor_y})" if cursor_x is not None else "unknown"

        stats = {
            "original_resolution": f"{original_width}x{original_height}",
            "scaled_resolution": f"{new_width}x{new_height}",
            "capture_time_seconds": round(capture_time, 3),
            "resize_time_seconds": round(resize_time, 3),
            "image_size_bytes": image_size_bytes,
            "image_size_kb": round(image_size_bytes / 1024, 2),
            "cursor_position": cursor_info,
            "screen_bounds": f"{screen_bounds}" if screen_bounds else "primary_screen",
        }

        return scaled_screenshot, stats

    except Exception as e:
        print(f"❌ Error capturing screenshot: {e}")
        return None, None
