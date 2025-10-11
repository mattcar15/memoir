"""
Snapshot retrieval and filtering functionality for the API server.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dateutil import parser as date_parser

from .vector_store import VectorStore
from .embeddings import create_embedding


def get_snapshots_in_range(
    start_date: str, end_date: str, logs_dir: Path
) -> List[Dict[str, Any]]:
    """
    Load JSON files within a time range and return snapshot data.

    Args:
        start_date: ISO format start date string
        end_date: ISO format end date string
        logs_dir: Path to logs directory

    Returns:
        List of snapshot dictionaries with loaded JSON data
    """
    try:
        start_dt = date_parser.parse(start_date)
        end_dt = date_parser.parse(end_date)
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")

    snapshots = []

    # Get all JSON files in logs directory
    json_files = list(logs_dir.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Parse timestamp from the data
            snapshot_time = date_parser.parse(data.get("timestamp", ""))

            # Check if within range
            if start_dt <= snapshot_time <= end_dt:
                snapshots.append(
                    {
                        "file_path": json_file,
                        "memory_id": data.get("memory_id", json_file.stem),
                        "timestamp": data.get("timestamp"),
                        "summary": data.get("summary"),
                        "stats": data.get("stats", {}),
                        "screenshot_path": data.get("screenshot_path"),
                        "raw_data": data,
                    }
                )

        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue

    return snapshots


def filter_top_k_by_tokens(
    snapshots: List[Dict[str, Any]], k: int = 30
) -> List[Dict[str, Any]]:
    """
    Sort snapshots by response_token_count and return top K.

    Args:
        snapshots: List of snapshot dictionaries
        k: Number of snapshots to return

    Returns:
        Top K snapshots sorted by response_token_count (descending)
    """

    def get_token_count(snapshot):
        stats = snapshot.get("stats", {})
        return stats.get("response_token_count", 0)

    # Sort by response_token_count descending
    sorted_snapshots = sorted(snapshots, key=get_token_count, reverse=True)

    return sorted_snapshots[:k]


def search_snapshots(
    query: str,
    vector_store: VectorStore,
    embedding_model: str,
    k: int = 30,
    threshold: float = 0.3,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Search snapshots using vector similarity with optional time filtering.

    Args:
        query: Search query text
        vector_store: VectorStore instance
        embedding_model: Embedding model name
        k: Maximum number of results to return
        threshold: Minimum similarity threshold (0-1)
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        logs_dir: Path to logs directory for loading full data

    Returns:
        List of snapshot dictionaries with similarity scores
    """
    # Create embedding for query
    query_embedding = create_embedding(query, embedding_model)
    if not query_embedding:
        return []

    # Search vector store (no time filtering at this stage)
    search_results = vector_store.search(
        query_embedding=query_embedding,
        n_results=k * 3,  # Get more results to account for time filtering
    )

    # Filter by similarity threshold
    # ChromaDB uses cosine distance: 0=perfect match, 1=orthogonal, 2=opposite
    # Convert to similarity: similarity = 1 - (distance / 2)
    filtered_results = [
        result
        for result in search_results
        if (1 - (result["distance"] / 2)) >= threshold
    ]

    # Apply time filtering if dates provided
    if start_date and end_date:
        try:
            start_dt = date_parser.parse(start_date)
            end_dt = date_parser.parse(end_date)

            time_filtered_results = []
            for result in filtered_results:
                # Get timestamp from metadata
                timestamp_str = result.get("metadata", {}).get("timestamp")
                if timestamp_str:
                    try:
                        result_time = date_parser.parse(timestamp_str)
                        if start_dt <= result_time <= end_dt:
                            time_filtered_results.append(result)
                    except Exception as e:
                        print(
                            f"Warning: Could not parse timestamp {timestamp_str}: {e}"
                        )
                        continue

            filtered_results = time_filtered_results
        except Exception as e:
            print(f"Warning: Invalid date format for time filtering: {e}")

    # Take top K results
    top_results = filtered_results[:k]

    # If we have logs_dir, load full snapshot data
    if logs_dir:
        enriched_results = []
        for result in top_results:
            memory_id = result["id"]
            json_file = logs_dir / f"{memory_id}.json"

            if json_file.exists():
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)

                    enriched_result = {
                        "memory_id": memory_id,
                        "timestamp": data.get("timestamp"),
                        "summary": data.get("summary"),
                        "stats": data.get("stats", {}),
                        "screenshot_path": data.get("screenshot_path"),
                        "similarity": 1 - (result["distance"] / 2),
                        "raw_data": data,
                    }
                    enriched_results.append(enriched_result)
                except Exception as e:
                    print(f"Warning: Could not load full data for {memory_id}: {e}")
                    # Fall back to basic data from vector store
                    enriched_results.append(
                        {
                            "memory_id": memory_id,
                            "timestamp": result["metadata"].get("timestamp"),
                            "summary": result["summary"],
                            "stats": result["metadata"].get("stats", {}),
                            "screenshot_path": result["metadata"].get(
                                "screenshot_path"
                            ),
                            "similarity": 1 - (result["distance"] / 2),
                            "raw_data": None,
                        }
                    )
            else:
                # Fall back to basic data from vector store
                enriched_results.append(
                    {
                        "memory_id": memory_id,
                        "timestamp": result["metadata"].get("timestamp"),
                        "summary": result["summary"],
                        "stats": result["metadata"].get("stats", {}),
                        "screenshot_path": result["metadata"].get("screenshot_path"),
                        "similarity": 1 - (result["distance"] / 2),
                        "raw_data": None,
                    }
                )

        return enriched_results

    # Return basic results without full data loading
    return [
        {
            "memory_id": result["id"],
            "timestamp": result["metadata"].get("timestamp"),
            "summary": result["summary"],
            "stats": result["metadata"].get("stats", {}),
            "screenshot_path": result["metadata"].get("screenshot_path"),
            "similarity": 1 - result["distance"],
            "raw_data": None,
        }
        for result in top_results
    ]


def load_snapshot_data(
    snapshot: Dict[str, Any], include_stats: bool = False, include_image: bool = False
) -> Dict[str, Any]:
    """
    Format snapshot data for API response.

    Args:
        snapshot: Snapshot dictionary from get_snapshots_in_range or search_snapshots
        include_stats: Whether to include stats in response
        include_image: Whether to include image URL in response

    Returns:
        Formatted snapshot dictionary for API response
    """
    response = {
        "timestamp": snapshot["timestamp"],
        "summary": snapshot["summary"],
        "memory_id": snapshot["memory_id"],
    }

    if include_stats and "stats" in snapshot:
        response["stats"] = snapshot["stats"]

    if include_image and snapshot.get("screenshot_path"):
        # Extract filename from path for URL
        screenshot_path = Path(snapshot["screenshot_path"])
        response["image_url"] = f"/images/{screenshot_path.name}"

    return response


def get_snapshot_by_id(
    memory_id: str,
    logs_dir: Path,
    include_stats: bool = False,
    include_image: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Get a specific snapshot by memory ID.

    Args:
        memory_id: Memory ID to retrieve
        logs_dir: Path to logs directory
        include_stats: Whether to include stats in response
        include_image: Whether to include image URL in response

    Returns:
        Formatted snapshot dictionary or None if not found
    """
    json_file = logs_dir / f"{memory_id}.json"

    if not json_file.exists():
        return None

    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        snapshot = {
            "memory_id": memory_id,
            "timestamp": data.get("timestamp"),
            "summary": data.get("summary"),
            "stats": data.get("stats", {}),
            "screenshot_path": data.get("screenshot_path"),
            "raw_data": data,
        }

        return load_snapshot_data(snapshot, include_stats, include_image)

    except Exception as e:
        print(f"Error loading snapshot {memory_id}: {e}")
        return None


def get_oldest_snapshot_timestamp(logs_dir: Path) -> Optional[str]:
    """
    Find the oldest snapshot timestamp in the logs directory.

    Args:
        logs_dir: Path to logs directory

    Returns:
        ISO format timestamp string of the oldest snapshot, or None if no snapshots exist
    """
    json_files = list(logs_dir.glob("*.json"))

    if not json_files:
        return None

    oldest_timestamp = None
    oldest_datetime = None

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            timestamp_str = data.get("timestamp")
            if timestamp_str:
                timestamp_dt = date_parser.parse(timestamp_str)

                if oldest_datetime is None or timestamp_dt < oldest_datetime:
                    oldest_datetime = timestamp_dt
                    oldest_timestamp = timestamp_str

        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue

    return oldest_timestamp
