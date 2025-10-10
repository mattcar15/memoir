"""
Storage and logging functionality for screenshots and log entries.
"""

import json
from datetime import datetime
from pathlib import Path


def setup_directories():
    """Create necessary directories for logs and screenshots"""
    logs_dir = Path("logs")
    screenshots_dir = logs_dir / "screenshots"

    logs_dir.mkdir(exist_ok=True)
    screenshots_dir.mkdir(exist_ok=True)

    return logs_dir, screenshots_dir


def save_log_entry(summary, resolution_tier, screenshot_path, logs_dir, stats=None):
    """
    Save a log entry as an individual JSON file.

    Args:
        summary: Model's output text
        resolution_tier: Resolution tier used
        screenshot_path: Path to saved screenshot or None
        logs_dir: Path to logs directory
        stats: Dictionary of performance statistics

    Returns:
        Path to the created log file
    """
    timestamp = datetime.now()

    # Create filename with timestamp
    filename = timestamp.strftime("%Y-%m-%d_%H-%M-%S.json")
    filepath = logs_dir / filename

    # Create log entry
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "summary": summary,
        "resolution_tier": resolution_tier,
        "screenshot_path": str(screenshot_path) if screenshot_path else None,
    }

    # Add stats if provided
    if stats:
        log_entry["stats"] = stats

    # Write to file
    try:
        with open(filepath, "w") as f:
            json.dump(log_entry, f, indent=2)
        return filepath
    except Exception as e:
        print(f"❌ Error saving log entry: {e}")
        return None


def save_screenshot(image, screenshots_dir):
    """
    Save screenshot image to disk.

    Args:
        image: PIL Image object
        screenshots_dir: Path to screenshots directory

    Returns:
        Path to saved screenshot or None
    """
    timestamp = datetime.now()
    filename = timestamp.strftime("%Y-%m-%d_%H-%M-%S.png")
    filepath = screenshots_dir / filename

    try:
        image.save(filepath, format="PNG")
        return filepath
    except Exception as e:
        print(f"❌ Error saving screenshot: {e}")
        return None
