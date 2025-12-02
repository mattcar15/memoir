"""
Repository layer for the three-layer memoir architecture.

Provides the single write path (upsert_memory_record) and query helpers.
Uses SQLAlchemy ORM for all database operations.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Tuple

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from chromadb import Collection

from .models import Episode, Memory, Snapshot, now_ms


# =============================================================================
# Write Operations - Single Path
# =============================================================================


def upsert_memory_record(
    session: Session,
    chroma_collection: Collection,
    memory: Memory,
    embedding: list[float],
    snapshot: Optional[Snapshot] = None,
    episode: Optional[Episode] = None,
) -> None:
    """
    Atomic upsert to SQLite + Chroma.

    This is the ONLY function that should write to both databases.
    It ensures SQLite and Chroma stay in lockstep.

    Args:
        session: SQLAlchemy session
        chroma_collection: Chroma collection for vector search
        memory: Memory object to upsert
        embedding: Embedding vector for the memory's search_text
        snapshot: Optional Snapshot to upsert (for snapshot-level memories)
        episode: Optional Episode to upsert
    """
    # 1. Upsert episode if provided
    if episode:
        _upsert_episode(session, episode)

    # 2. Upsert snapshot if provided
    if snapshot:
        _upsert_snapshot(session, snapshot)

    # 3. Upsert memory
    _upsert_memory(session, memory)

    # Flush to ensure IDs are available
    session.flush()

    # 4. Upsert into Chroma
    _upsert_chroma(chroma_collection, memory, embedding, snapshot)


def _upsert_episode(session: Session, episode: Episode) -> None:
    """Insert or update an episode."""
    existing = session.get(Episode, episode.id)
    if existing:
        existing.started_at = episode.started_at
        existing.ended_at = episode.ended_at
        existing.title = episode.title
        existing.summary = episode.summary
        existing.tags_json = episode.tags_json
        existing.updated_at = now_ms()
    else:
        session.add(episode)


def _upsert_snapshot(session: Session, snapshot: Snapshot) -> None:
    """Insert or update a snapshot."""
    existing = session.get(Snapshot, snapshot.id)
    if existing:
        existing.episode_id = snapshot.episode_id
        existing.captured_at = snapshot.captured_at
        existing.app = snapshot.app
        existing.url = snapshot.url
        existing.window_title = snapshot.window_title
        existing.image_path = snapshot.image_path
        existing.ocr_text = snapshot.ocr_text
        existing.extra_json = snapshot.extra_json
    else:
        session.add(snapshot)


def _upsert_memory(session: Session, memory: Memory) -> None:
    """Insert or update a memory."""
    existing = session.get(Memory, memory.id)
    if existing:
        existing.kind = memory.kind
        existing.episode_id = memory.episode_id
        existing.snapshot_id = memory.snapshot_id
        existing.title = memory.title
        existing.summary = memory.summary
        existing.bullets_json = memory.bullets_json
        existing.tags_json = memory.tags_json
        existing.entities_json = memory.entities_json
        existing.search_text = memory.search_text
        existing.updated_at = now_ms()
    else:
        session.add(memory)


def _upsert_chroma(
    collection: Collection,
    memory: Memory,
    embedding: list[float],
    snapshot: Optional[Snapshot] = None,
) -> None:
    """Upsert memory into Chroma collection."""
    metadata = memory.to_chroma_metadata(snapshot)

    # Convert None values to empty strings (Chroma requirement)
    metadata = {k: (v if v is not None else "") for k, v in metadata.items()}

    collection.upsert(
        ids=[memory.id],
        embeddings=[embedding],
        documents=[memory.search_text],
        metadatas=[metadata],
    )


# =============================================================================
# Delete Operations
# =============================================================================


def delete_memory(
    session: Session,
    chroma_collection: Collection,
    memory_id: str,
) -> bool:
    """
    Delete a memory from both SQLite and Chroma.

    Args:
        session: SQLAlchemy session
        chroma_collection: Chroma collection
        memory_id: ID of the memory to delete

    Returns:
        True if deleted, False if not found
    """
    memory = session.get(Memory, memory_id)
    if memory is None:
        return False

    session.delete(memory)

    try:
        chroma_collection.delete(ids=[memory_id])
    except Exception:
        pass  # Chroma might not have it

    return True


def delete_snapshot(session: Session, snapshot_id: str) -> bool:
    """Delete a snapshot (memory will cascade delete if linked)."""
    snapshot = session.get(Snapshot, snapshot_id)
    if snapshot is None:
        return False

    session.delete(snapshot)
    return True


def delete_episode(session: Session, episode_id: str) -> bool:
    """Delete an episode and all its snapshots/memories (via cascade)."""
    episode = session.get(Episode, episode_id)
    if episode is None:
        return False

    session.delete(episode)
    return True


# =============================================================================
# Read Operations - Episodes
# =============================================================================


def get_episode_by_id(session: Session, episode_id: str) -> Optional[Episode]:
    """Get a single episode by ID."""
    return session.get(Episode, episode_id)


def get_episodes(
    session: Session,
    limit: int = 100,
    offset: int = 0,
    order_desc: bool = True,
) -> List[Episode]:
    """Get episodes with pagination."""
    query = select(Episode)
    if order_desc:
        query = query.order_by(Episode.started_at.desc())
    else:
        query = query.order_by(Episode.started_at.asc())
    query = query.limit(limit).offset(offset)

    return list(session.scalars(query).all())


def get_episode_by_time(session: Session, timestamp_ms: int) -> Optional[Episode]:
    """Get the episode that contains a given timestamp."""
    query = (
        select(Episode)
        .where(Episode.started_at <= timestamp_ms)
        .where((Episode.ended_at.is_(None)) | (Episode.ended_at >= timestamp_ms))
        .order_by(Episode.started_at.desc())
        .limit(1)
    )
    return session.scalars(query).first()


# =============================================================================
# Read Operations - Snapshots
# =============================================================================


def get_snapshot_by_id(session: Session, snapshot_id: str) -> Optional[Snapshot]:
    """Get a single snapshot by ID."""
    return session.get(Snapshot, snapshot_id)


def get_snapshots_by_episode(
    session: Session,
    episode_id: str,
    order_asc: bool = True,
) -> List[Snapshot]:
    """Get all snapshots in an episode, ordered by capture time."""
    query = select(Snapshot).where(Snapshot.episode_id == episode_id)
    if order_asc:
        query = query.order_by(Snapshot.captured_at.asc())
    else:
        query = query.order_by(Snapshot.captured_at.desc())

    return list(session.scalars(query).all())


def get_snapshots_in_range(
    session: Session,
    start_ms: int,
    end_ms: int,
    app: Optional[str] = None,
    limit: int = 100,
) -> List[Snapshot]:
    """Get snapshots within a time range."""
    query = (
        select(Snapshot)
        .where(Snapshot.captured_at >= start_ms)
        .where(Snapshot.captured_at <= end_ms)
    )

    if app:
        query = query.where(Snapshot.app == app)

    query = query.order_by(Snapshot.captured_at.desc()).limit(limit)

    return list(session.scalars(query).all())


def get_recent_snapshots(
    session: Session,
    limit: int = 100,
    app: Optional[str] = None,
) -> List[Snapshot]:
    """Get the most recent snapshots."""
    query = select(Snapshot)

    if app:
        query = query.where(Snapshot.app == app)

    query = query.order_by(Snapshot.captured_at.desc()).limit(limit)

    return list(session.scalars(query).all())


def get_oldest_snapshot_timestamp(session: Session) -> Optional[int]:
    """Get the timestamp of the oldest snapshot."""
    result = session.scalar(select(func.min(Snapshot.captured_at)))
    return result


# =============================================================================
# Read Operations - Memories
# =============================================================================


def get_memory_by_id(session: Session, memory_id: str) -> Optional[Memory]:
    """Get a single memory by ID."""
    return session.get(Memory, memory_id)


def get_memories_by_ids(session: Session, memory_ids: List[str]) -> List[Memory]:
    """Get multiple memories by their IDs (preserves order)."""
    if not memory_ids:
        return []

    query = select(Memory).where(Memory.id.in_(memory_ids))
    memories = {m.id: m for m in session.scalars(query).all()}

    # Return in the requested order
    return [memories[mid] for mid in memory_ids if mid in memories]


def get_memories_by_episode(
    session: Session,
    episode_id: str,
    kind: Optional[str] = None,
) -> List[Memory]:
    """Get all memories for an episode."""
    query = select(Memory).where(Memory.episode_id == episode_id)

    if kind:
        query = query.where(Memory.kind == kind)

    query = query.order_by(Memory.created_at.asc())

    return list(session.scalars(query).all())


def get_memory_by_snapshot(session: Session, snapshot_id: str) -> Optional[Memory]:
    """Get the memory associated with a snapshot."""
    query = (
        select(Memory)
        .where(Memory.snapshot_id == snapshot_id)
        .where(Memory.kind == "snapshot")
    )
    return session.scalars(query).first()


def get_recent_memories(
    session: Session,
    limit: int = 100,
    kind: Optional[str] = None,
) -> List[Memory]:
    """Get the most recent memories."""
    query = select(Memory)

    if kind:
        query = query.where(Memory.kind == kind)

    query = query.order_by(Memory.created_at.desc()).limit(limit)

    return list(session.scalars(query).all())


# =============================================================================
# Search Operations (SQLite FTS)
# =============================================================================


def search_memories_fts(
    session: Session,
    query_text: str,
    limit: int = 100,
    kind: Optional[str] = None,
) -> List[Tuple[Memory, float]]:
    """
    Search memories using FTS5 BM25.

    Returns list of (Memory, score) tuples ordered by relevance.
    """
    try:
        # Build the raw SQL query for FTS
        if kind:
            sql = text(
                """
                SELECT m.*, bm25(memories_fts) as score
                FROM memories_fts
                JOIN memories m ON m.rowid = memories_fts.rowid
                WHERE memories_fts MATCH :query AND m.kind = :kind
                ORDER BY score
                LIMIT :limit
            """
            )
            result = session.execute(
                sql, {"query": query_text, "kind": kind, "limit": limit}
            )
        else:
            sql = text(
                """
                SELECT m.*, bm25(memories_fts) as score
                FROM memories_fts
                JOIN memories m ON m.rowid = memories_fts.rowid
                WHERE memories_fts MATCH :query
                ORDER BY score
                LIMIT :limit
            """
            )
            result = session.execute(sql, {"query": query_text, "limit": limit})

        results = []
        for row in result:
            # Convert row to Memory object
            memory = Memory(
                id=row.id,
                kind=row.kind,
                episode_id=row.episode_id,
                snapshot_id=row.snapshot_id,
                title=row.title,
                summary=row.summary,
                bullets_json=row.bullets_json,
                tags_json=row.tags_json,
                entities_json=row.entities_json,
                search_text=row.search_text,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            results.append((memory, row.score))

        return results

    except Exception as e:
        print(f"FTS search error: {e}")
        return []


# =============================================================================
# Aggregation Queries
# =============================================================================


def get_apps(session: Session) -> List[str]:
    """Get list of unique apps from snapshots."""
    query = (
        select(Snapshot.app)
        .where(Snapshot.app.isnot(None))
        .distinct()
        .order_by(Snapshot.app)
    )
    return [row for row in session.scalars(query).all()]


def get_tags(session: Session) -> List[str]:
    """Get list of unique tags from memories."""
    all_tags = set()
    for memory in session.scalars(select(Memory)).all():
        all_tags.update(memory.tags)
    return sorted(all_tags)


def get_entities(session: Session) -> List[str]:
    """Get list of unique entities from memories."""
    all_entities = set()
    for memory in session.scalars(select(Memory)).all():
        all_entities.update(memory.entities)
    return sorted(all_entities)


def count_snapshots_by_app(session: Session) -> dict[str, int]:
    """Get snapshot counts grouped by app."""
    query = (
        select(Snapshot.app, func.count(Snapshot.id))
        .where(Snapshot.app.isnot(None))
        .group_by(Snapshot.app)
        .order_by(func.count(Snapshot.id).desc())
    )
    return {row[0]: row[1] for row in session.execute(query).all()}


def count_snapshots_by_day(session: Session) -> List[dict[str, Any]]:
    """Get snapshot counts grouped by day."""
    # Use SQLite's date function
    sql = text(
        """
        SELECT 
            date(captured_at / 1000, 'unixepoch') as day,
            COUNT(*) as count
        FROM snapshots
        GROUP BY day
        ORDER BY day DESC
    """
    )
    result = session.execute(sql)
    return [{"day": row.day, "count": row.count} for row in result]
