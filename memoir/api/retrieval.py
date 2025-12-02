"""
Snapshot and memory retrieval functionality for the API server.

Uses SQLite (via SQLAlchemy) as the source of truth, with Chroma for semantic search.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dateutil import parser as date_parser

from sqlalchemy.orm import Session

from ..storage.vector_store import VectorStore
from ..storage.embeddings import create_embedding
from ..storage.models import Memory, Snapshot, Episode
from ..storage import repository


def get_snapshots_in_range(
    session: Session,
    start_date: str,
    end_date: str,
    app: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Get snapshots within a time range from SQLite.

    Args:
        session: SQLAlchemy session
        start_date: ISO format start date string
        end_date: ISO format end date string
        app: Optional app filter
        limit: Maximum number of results

    Returns:
        List of snapshot dictionaries with associated memory data
    """
    try:
        start_dt = date_parser.parse(start_date)
        end_dt = date_parser.parse(end_date)
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")

    # Convert to unix ms
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # Get snapshots from SQLite
    snapshots = repository.get_snapshots_in_range(session, start_ms, end_ms, app, limit)

    # Enrich with memory data
    results = []
    for snapshot in snapshots:
        memory = repository.get_memory_by_snapshot(session, snapshot.id)
        results.append(_format_snapshot_with_memory(snapshot, memory))

    return results


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

    sorted_snapshots = sorted(snapshots, key=get_token_count, reverse=True)
    return sorted_snapshots[:k]


def search_memories(
    session: Session,
    query: str,
    vector_store: VectorStore,
    embedding_model: str,
    k: int = 30,
    threshold: float = 0.3,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    kind: Optional[str] = None,
    app: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search memories using hybrid vector + optional keyword search.

    Pipeline:
    1. Query Chroma for semantic candidates
    2. Apply time/metadata filters
    3. Hydrate full data from SQLite
    4. Return ranked results

    Args:
        session: SQLAlchemy session
        query: Search query text
        vector_store: VectorStore instance
        embedding_model: Embedding model name
        k: Maximum number of results to return
        threshold: Minimum similarity threshold (0-1)
        start_date: Optional start date filter (ISO format)
        end_date: Optional end date filter (ISO format)
        kind: Optional memory kind filter ('snapshot' or 'episode')
        app: Optional app filter

    Returns:
        List of memory dictionaries with similarity scores and related data
    """
    # Create embedding for query
    query_embedding = create_embedding(query, embedding_model)
    if not query_embedding:
        return []

    # Build Chroma where filter
    where_filter = _build_chroma_filter(kind, app, start_date, end_date)

    # Search Chroma (get more results to account for filtering)
    chroma_results = vector_store.search(
        query_embedding=query_embedding,
        n_results=k * 3,
        where=where_filter if where_filter else None,
    )

    if not chroma_results:
        return []

    # Filter by similarity threshold
    # ChromaDB uses cosine distance: 0=perfect match, 1=orthogonal, 2=opposite
    # Convert to similarity: similarity = 1 - (distance / 2)
    filtered_results = []
    for result in chroma_results:
        similarity = 1 - (result["distance"] / 2)
        if similarity >= threshold:
            result["similarity"] = similarity
            filtered_results.append(result)

    # Apply time filtering if dates provided and not already filtered in Chroma
    if start_date and end_date:
        filtered_results = _filter_by_time(filtered_results, start_date, end_date)

    # Take top K
    top_results = filtered_results[:k]

    if not top_results:
        return []

    # Hydrate from SQLite
    memory_ids = [r["id"] for r in top_results]
    memories = repository.get_memories_by_ids(session, memory_ids)

    # Build result list with full data
    results = []
    memory_dict = {m.id: m for m in memories}

    for chroma_result in top_results:
        memory_id = chroma_result["id"]
        memory = memory_dict.get(memory_id)

        if not memory:
            continue

        result = _format_memory_result(session, memory, chroma_result["similarity"])
        results.append(result)

    return results


def _build_chroma_filter(
    kind: Optional[str] = None,
    app: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build a Chroma where filter from the provided parameters."""
    conditions = []

    if kind:
        conditions.append({"kind": kind})

    if app:
        conditions.append({"app": app})

    # Time filtering in Chroma (using captured_at in metadata)
    if start_date and end_date:
        try:
            start_ms = int(date_parser.parse(start_date).timestamp() * 1000)
            end_ms = int(date_parser.parse(end_date).timestamp() * 1000)
            conditions.append({"captured_at": {"$gte": start_ms}})
            conditions.append({"captured_at": {"$lte": end_ms}})
        except Exception:
            pass  # Skip time filter if parsing fails

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def _filter_by_time(
    results: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    """Filter results by time range using metadata."""
    try:
        start_ms = int(date_parser.parse(start_date).timestamp() * 1000)
        end_ms = int(date_parser.parse(end_date).timestamp() * 1000)
    except Exception:
        return results

    filtered = []
    for result in results:
        captured_at = result.get("metadata", {}).get("captured_at", 0)
        if isinstance(captured_at, str):
            try:
                captured_at = int(date_parser.parse(captured_at).timestamp() * 1000)
            except Exception:
                captured_at = 0

        if start_ms <= captured_at <= end_ms:
            filtered.append(result)

    return filtered


def _format_memory_result(
    session: Session,
    memory: Memory,
    similarity: float,
) -> Dict[str, Any]:
    """Format a memory with its related snapshot/episode data."""
    result = {
        "memory_id": memory.id,
        "kind": memory.kind,
        "title": memory.title,
        "summary": memory.summary,
        "bullets": memory.bullets,
        "tags": memory.tags,
        "entities": memory.entities,
        "similarity": round(similarity, 4),
        "created_at": memory.created_at,
    }

    # Add snapshot data if this is a snapshot memory
    if memory.kind == "snapshot" and memory.snapshot_id:
        snapshot = repository.get_snapshot_by_id(session, memory.snapshot_id)
        if snapshot:
            result["snapshot"] = {
                "id": snapshot.id,
                "captured_at": snapshot.captured_at,
                "app": snapshot.app,
                "url": snapshot.url,
                "window_title": snapshot.window_title,
                "image_path": snapshot.image_path,
            }
            # Derive timestamp from snapshot
            result["timestamp"] = datetime.fromtimestamp(
                snapshot.captured_at / 1000
            ).isoformat()

    # Add episode info if available
    if memory.episode_id:
        episode = repository.get_episode_by_id(session, memory.episode_id)
        if episode:
            result["episode"] = {
                "id": episode.id,
                "title": episode.title,
                "started_at": episode.started_at,
            }

    return result


def _format_snapshot_with_memory(
    snapshot: Snapshot,
    memory: Optional[Memory],
) -> Dict[str, Any]:
    """Format a snapshot dict with associated memory data."""
    result = {
        "snapshot_id": snapshot.id,
        "timestamp": datetime.fromtimestamp(snapshot.captured_at / 1000).isoformat(),
        "captured_at": snapshot.captured_at,
        "app": snapshot.app,
        "url": snapshot.url,
        "window_title": snapshot.window_title,
        "image_path": snapshot.image_path,
        "episode_id": snapshot.episode_id,
    }

    if memory:
        result["memory_id"] = memory.id
        result["title"] = memory.title
        result["summary"] = memory.summary
        result["bullets"] = memory.bullets
        result["tags"] = memory.tags
        result["entities"] = memory.entities
    else:
        result["memory_id"] = None
        result["title"] = None
        result["summary"] = None
        result["bullets"] = []
        result["tags"] = []
        result["entities"] = []

    return result


def load_snapshot_data(
    snapshot: Dict[str, Any],
    include_stats: bool = False,
    include_image: bool = False,
) -> Dict[str, Any]:
    """
    Format snapshot data for API response.

    Args:
        snapshot: Snapshot dictionary
        include_stats: Whether to include stats in response
        include_image: Whether to include image URL in response

    Returns:
        Formatted snapshot dictionary for API response
    """
    response = {
        "timestamp": snapshot.get("timestamp"),
        "memory_id": snapshot.get("memory_id"),
        "title": snapshot.get("title"),
        "summary": snapshot.get("summary"),
        "bullets": snapshot.get("bullets", []),
        "tags": snapshot.get("tags", []),
        "entities": snapshot.get("entities", []),
    }

    if include_stats and "stats" in snapshot:
        response["stats"] = snapshot["stats"]

    if include_image and snapshot.get("image_path"):
        # Extract filename from path for URL
        screenshot_path = Path(snapshot["image_path"])
        response["image_url"] = f"/images/{screenshot_path.name}"

    return response


def get_snapshot_by_id(
    session: Session,
    snapshot_id: str,
    include_stats: bool = False,
    include_image: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Get a specific snapshot by ID.

    Args:
        session: SQLAlchemy session
        snapshot_id: Snapshot ID to retrieve
        include_stats: Whether to include stats in response
        include_image: Whether to include image URL in response

    Returns:
        Formatted snapshot dictionary or None if not found
    """
    snapshot = repository.get_snapshot_by_id(session, snapshot_id)
    if not snapshot:
        return None

    memory = repository.get_memory_by_snapshot(session, snapshot_id)
    formatted = _format_snapshot_with_memory(snapshot, memory)

    return load_snapshot_data(formatted, include_stats, include_image)


def get_memory_by_id(
    session: Session,
    memory_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Get a specific memory by ID with related data.

    Args:
        session: SQLAlchemy session
        memory_id: Memory ID to retrieve

    Returns:
        Memory dictionary with related snapshot/episode data, or None
    """
    memory = repository.get_memory_by_id(session, memory_id)
    if not memory:
        return None

    return _format_memory_result(session, memory, similarity=1.0)


def get_oldest_snapshot_timestamp(session: Session) -> Optional[str]:
    """
    Find the oldest snapshot timestamp.

    Args:
        session: SQLAlchemy session

    Returns:
        ISO format timestamp string of the oldest snapshot, or None
    """
    oldest_ms = repository.get_oldest_snapshot_timestamp(session)
    if oldest_ms is None:
        return None

    return datetime.fromtimestamp(oldest_ms / 1000).isoformat()


def get_episodes(
    session: Session,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Get episodes with pagination.

    Args:
        session: SQLAlchemy session
        limit: Maximum number to return
        offset: Pagination offset

    Returns:
        List of episode dictionaries
    """
    episodes = repository.get_episodes(session, limit, offset)

    results = []
    for episode in episodes:
        # Get snapshot count for this episode
        snapshots = repository.get_snapshots_by_episode(session, episode.id)

        results.append(
            {
                "id": episode.id,
                "started_at": episode.started_at,
                "ended_at": episode.ended_at,
                "title": episode.title,
                "summary": episode.summary,
                "tags": episode.tags,
                "snapshot_count": len(snapshots),
                "created_at": episode.created_at,
            }
        )

    return results


def get_episode_with_snapshots(
    session: Session,
    episode_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Get an episode with all its snapshots and memories.

    Args:
        session: SQLAlchemy session
        episode_id: Episode ID

    Returns:
        Episode dictionary with snapshots, or None
    """
    episode = repository.get_episode_by_id(session, episode_id)
    if not episode:
        return None

    snapshots = repository.get_snapshots_by_episode(session, episode_id)

    snapshot_data = []
    for snapshot in snapshots:
        memory = repository.get_memory_by_snapshot(session, snapshot.id)
        snapshot_data.append(_format_snapshot_with_memory(snapshot, memory))

    return {
        "id": episode.id,
        "started_at": episode.started_at,
        "ended_at": episode.ended_at,
        "title": episode.title,
        "summary": episode.summary,
        "tags": episode.tags,
        "snapshots": snapshot_data,
        "created_at": episode.created_at,
    }


# =============================================================================
# Legacy compatibility functions (for existing JSON-based code)
# =============================================================================


def get_snapshots_in_range_legacy(
    start_date: str,
    end_date: str,
    logs_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Legacy function for backward compatibility with JSON file storage.

    Reads JSON files from the logs directory.
    """
    import json

    try:
        start_dt = date_parser.parse(start_date)
        end_dt = date_parser.parse(end_date)
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")

    snapshots = []
    json_files = list(logs_dir.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            snapshot_time = date_parser.parse(data.get("timestamp", ""))

            if start_dt <= snapshot_time <= end_dt:
                snapshots.append(
                    {
                        "file_path": json_file,
                        "memory_id": data.get("memory_id", json_file.stem),
                        "timestamp": data.get("timestamp"),
                        "title": data.get("title"),
                        "summary": data.get("summary"),
                        "bullets": data.get("bullets", []),
                        "tags": data.get("tags", []),
                        "entities": data.get("entities", []),
                        "stats": data.get("stats", {}),
                        "screenshot_path": data.get("screenshot_path"),
                        "raw_data": data,
                    }
                )

        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue

    return snapshots


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
    Legacy search function for backward compatibility.

    Uses Chroma for search but hydrates from JSON files.
    """
    import json

    # Create embedding for query
    query_embedding = create_embedding(query, embedding_model)
    if not query_embedding:
        return []

    # Search vector store
    search_results = vector_store.search(
        query_embedding=query_embedding,
        n_results=k * 3,
    )

    # Filter by similarity threshold
    filtered_results = [
        result
        for result in search_results
        if (1 - (result["distance"] / 2)) >= threshold
    ]

    # Apply time filtering if dates provided
    if start_date and end_date:
        filtered_results = _filter_by_time_legacy(
            filtered_results, start_date, end_date
        )

    top_results = filtered_results[:k]

    # If we have logs_dir, load full snapshot data from JSON
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
                        "title": data.get("title"),
                        "summary": data.get("summary"),
                        "bullets": data.get("bullets", []),
                        "tags": data.get("tags", []),
                        "entities": data.get("entities", []),
                        "stats": data.get("stats", {}),
                        "screenshot_path": data.get("screenshot_path"),
                        "similarity": 1 - (result["distance"] / 2),
                        "raw_data": data,
                    }
                    enriched_results.append(enriched_result)
                except Exception as e:
                    print(f"Warning: Could not load full data for {memory_id}: {e}")
                    enriched_results.append(_format_legacy_result(result))
            else:
                enriched_results.append(_format_legacy_result(result))

        return enriched_results

    return [_format_legacy_result(r) for r in top_results]


def _filter_by_time_legacy(
    results: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    """Filter legacy results by time range."""
    try:
        start_dt = date_parser.parse(start_date)
        end_dt = date_parser.parse(end_date)
    except Exception:
        return results

    filtered = []
    for result in results:
        timestamp_str = result.get("metadata", {}).get("timestamp")
        if timestamp_str:
            try:
                result_time = date_parser.parse(timestamp_str)
                if start_dt <= result_time <= end_dt:
                    filtered.append(result)
            except Exception:
                continue

    return filtered


def _format_legacy_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format a legacy Chroma result."""
    return {
        "memory_id": result["id"],
        "timestamp": result.get("metadata", {}).get("timestamp"),
        "title": result.get("metadata", {}).get("title"),
        "summary": result.get("document", ""),
        "bullets": [],
        "tags": result.get("metadata", {}).get("tags", []),
        "entities": result.get("metadata", {}).get("entities", []),
        "stats": result.get("metadata", {}).get("stats", {}),
        "screenshot_path": result.get("metadata", {}).get("screenshot_path"),
        "similarity": 1 - (result["distance"] / 2),
        "raw_data": None,
    }
