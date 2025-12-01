"""
Configuration and constants for the Screenshot Memory System.
"""

from pathlib import Path

# Resolution tier configurations
RESOLUTION_TIERS = {
    "efficient": (800, 600),
    "balanced": (1024, 768),
    "detailed": (1920, 1080),
}

# Debug mode (keep captured frames after processing)
DEBUG_MODE = False

# Queue and processing paths
# Use project-relative paths for easier testing (override with CLI args if needed)
_PROJECT_ROOT = Path(__file__).parent.parent
QUEUE_DB_PATH = _PROJECT_ROOT / "logs" / "processing_queue.db"
CAPTURE_STAGING_DIR = _PROJECT_ROOT / "logs" / "staging"

# MLX model configuration
MLX_MODEL_NAME = "lmstudio-community/Qwen3-VL-2B-Instruct-MLX-8bit"
MLX_MAX_IMAGE_SIDE = 1024
MLX_MAX_NEW_TOKENS = 128

# Window monitoring and deduplication
PHASH_THRESHOLD_PER_TAB = 10
WINDOW_POLL_INTERVAL = 0.5  # seconds

# System monitoring
SYSTEM_MONITOR_CACHE_SECONDS = 30
QUEUE_CHECK_INTERVAL = 2.0  # seconds
