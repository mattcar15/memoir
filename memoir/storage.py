"""
Storage and logging functionality for screenshots and log entries.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .embeddings import create_embedding
from .vector_store import VectorStore


def setup_directories():
    """Create necessary directories for logs and screenshots"""
    logs_dir = Path("logs")
    screenshots_dir = logs_dir / "screenshots"

    logs_dir.mkdir(exist_ok=True)
    screenshots_dir.mkdir(exist_ok=True)

    return logs_dir, screenshots_dir


def save_log_entry(
    summary,
    resolution_tier,
    screenshot_path,
    logs_dir,
    stats=None,
    vector_store: Optional[VectorStore] = None,
    embedding_model: str = "embeddinggemma",
):
    """
    Save a log entry as an individual JSON file and add to vector store.

    Args:
        summary: Model's output text
        resolution_tier: Resolution tier used
        screenshot_path: Path to saved screenshot or None
        logs_dir: Path to logs directory
        stats: Dictionary of performance statistics
        vector_store: Optional VectorStore instance for semantic search
        embedding_model: Embedding model name

    Returns:
        Path to the created log file
    """
    timestamp = datetime.now()

    # Create filename with timestamp
    filename = timestamp.strftime("%Y-%m-%d_%H-%M-%S.json")
    filepath = logs_dir / filename
    memory_id = filename.replace(".json", "")

    # Create log entry
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "summary": summary,
        "resolution_tier": resolution_tier,
        "screenshot_path": str(screenshot_path) if screenshot_path else None,
        "memory_id": memory_id,
    }

    # Add stats if provided
    if stats:
        log_entry["stats"] = stats

    # Write to file
    try:
        with open(filepath, "w") as f:
            json.dump(log_entry, f, indent=2)
    except Exception as e:
        print(f"❌ Error saving log entry: {e}")
        return None

    # Add to vector store if provided
    if vector_store:
        print("🔄 Creating embedding for memory...")
        embedding = create_embedding(summary, embedding_model)

        if embedding:
            # Prepare metadata
            metadata = {
                "timestamp": timestamp.isoformat(),
                "resolution_tier": resolution_tier,
                "screenshot_path": str(screenshot_path) if screenshot_path else None,
                "stats": stats,
            }

            success = vector_store.add_memory(
                memory_id=memory_id,
                summary=summary,
                embedding=embedding,
                metadata=metadata,
            )

            if success:
                print(
                    f"✅ Memory added to vector store (Total: {vector_store.count()})"
                )
            else:
                print("⚠️  Failed to add memory to vector store")
        else:
            print("⚠️  Skipping vector store (embedding failed)")

    return filepath


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
