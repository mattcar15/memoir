"""
Storage module for the three-layer memoir architecture.

This module provides:
- models: SQLAlchemy ORM models (Episode, Snapshot, Memory)
- database: SQLAlchemy engine/session management
- repository: CRUD operations and the single write path (upsert_memory_record)
- vector_store: Chroma-based semantic search
- embeddings: Embedding creation utilities
- migrations: Alembic migrations for schema management
"""

from .models import (
    Base,
    Episode,
    Snapshot,
    Memory,
    SearchIndex,
    episode_from_dict,
    snapshot_from_dict,
    memory_from_dict,
)
from . import search_index
from . import hybrid_search
from .database import (
    init_database,
    get_session,
    get_db_path,
    get_db_url,
    create_db_engine,
    get_engine,
    get_session_factory,
    create_session,
    get_database_info,
    rebuild_fts,
    # Legacy compatibility
    get_connection,
    create_connection,
)
from .repository import (
    upsert_memory_record,
    delete_memory,
    delete_snapshot,
    delete_episode,
    get_episode_by_id,
    get_episodes,
    get_episode_by_time,
    get_snapshot_by_id,
    get_snapshots_by_episode,
    get_snapshots_in_range,
    get_recent_snapshots,
    get_oldest_snapshot_timestamp,
    get_memory_by_id,
    get_memories_by_ids,
    get_memory_by_snapshot,
    get_memories_by_episode,
    get_recent_memories,
    search_memories_fts,
    get_apps,
    get_tags,
    get_entities,
    count_snapshots_by_app,
    count_snapshots_by_day,
)
from .vector_store import VectorStore
from .embeddings import create_embedding, create_embeddings_batch

__all__ = [
    # Base
    "Base",
    # Models
    "Episode",
    "Snapshot",
    "Memory",
    "SearchIndex",
    "episode_from_dict",
    "snapshot_from_dict",
    "memory_from_dict",
    # Search
    "search_index",
    "hybrid_search",
    # Database
    "init_database",
    "get_session",
    "get_db_path",
    "get_db_url",
    "create_db_engine",
    "get_engine",
    "get_session_factory",
    "create_session",
    "get_database_info",
    "rebuild_fts",
    "get_connection",
    "create_connection",
    # Repository
    "upsert_memory_record",
    "delete_memory",
    "delete_snapshot",
    "delete_episode",
    "get_episode_by_id",
    "get_episodes",
    "get_episode_by_time",
    "get_snapshot_by_id",
    "get_snapshots_by_episode",
    "get_snapshots_in_range",
    "get_recent_snapshots",
    "get_oldest_snapshot_timestamp",
    "get_memory_by_id",
    "get_memories_by_ids",
    "get_memory_by_snapshot",
    "get_memories_by_episode",
    "get_recent_memories",
    "search_memories_fts",
    "get_apps",
    "get_tags",
    "get_entities",
    "count_snapshots_by_app",
    "count_snapshots_by_day",
    # Vector Store
    "VectorStore",
    # Embeddings
    "create_embedding",
    "create_embeddings_batch",
]
