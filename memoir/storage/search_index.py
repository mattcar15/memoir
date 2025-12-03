"""
Search index management for unified hybrid search.

This module provides functions to:
- Sync the search_index table when entities are created/updated/deleted
- Update the Chroma vector index in lockstep with SQLite
"""

from __future__ import annotations

from typing import List, Optional
import uuid

from sqlalchemy.orm import Session

from .models import SearchIndex, Snapshot, Episode, Memory, now_ms


# =============================================================================
# Index Management Functions
# =============================================================================

def index_snapshot(
    session: Session,
    snapshot: Snapshot,
    chroma_collection=None,
    embedding: Optional[List[float]] = None,
) -> SearchIndex:
    """
    Add or update a snapshot in the search index.
    
    Args:
        session: SQLAlchemy session
        snapshot: Snapshot to index
        chroma_collection: Optional Chroma collection for vector indexing
        embedding: Optional pre-computed embedding vector
        
    Returns:
        The created/updated SearchIndex entry
    """
    search_text = SearchIndex.build_snapshot_search_text(snapshot)
    
    # Skip if no searchable content
    if not search_text.strip():
        return None
    
    index_id = f"si_snap_{snapshot.id}"
    
    # Check if exists
    existing = session.get(SearchIndex, index_id)
    
    if existing:
        existing.search_text = search_text
        existing.title = snapshot.window_title
        existing.captured_at = snapshot.captured_at
        existing.app = snapshot.app
        existing.updated_at = now_ms()
        entry = existing
    else:
        entry = SearchIndex(
            id=index_id,
            entity_type="snapshot",
            entity_id=snapshot.id,
            search_text=search_text,
            title=snapshot.window_title,
            captured_at=snapshot.captured_at,
            app=snapshot.app,
            created_at=now_ms(),
            updated_at=now_ms(),
        )
        session.add(entry)
    
    session.flush()
    
    # Update Chroma if provided
    if chroma_collection is not None and embedding is not None:
        _upsert_chroma(chroma_collection, entry, embedding)
    
    return entry


def index_episode(
    session: Session,
    episode: Episode,
    chroma_collection=None,
    embedding: Optional[List[float]] = None,
) -> SearchIndex:
    """
    Add or update an episode in the search index.
    
    Args:
        session: SQLAlchemy session
        episode: Episode to index
        chroma_collection: Optional Chroma collection for vector indexing
        embedding: Optional pre-computed embedding vector
        
    Returns:
        The created/updated SearchIndex entry
    """
    search_text = SearchIndex.build_episode_search_text(episode)
    
    # Skip if no searchable content
    if not search_text.strip():
        return None
    
    index_id = f"si_ep_{episode.id}"
    
    # Check if exists
    existing = session.get(SearchIndex, index_id)
    
    if existing:
        existing.search_text = search_text
        existing.title = episode.title
        existing.captured_at = episode.started_at
        existing.updated_at = now_ms()
        entry = existing
    else:
        entry = SearchIndex(
            id=index_id,
            entity_type="episode",
            entity_id=episode.id,
            search_text=search_text,
            title=episode.title,
            captured_at=episode.started_at,
            app=None,
            created_at=now_ms(),
            updated_at=now_ms(),
        )
        session.add(entry)
    
    session.flush()
    
    # Update Chroma if provided
    if chroma_collection is not None and embedding is not None:
        _upsert_chroma(chroma_collection, entry, embedding)
    
    return entry


def index_memory(
    session: Session,
    memory: Memory,
    snapshot: Optional[Snapshot] = None,
    chroma_collection=None,
    embedding: Optional[List[float]] = None,
) -> SearchIndex:
    """
    Add or update a memory in the search index.
    
    Args:
        session: SQLAlchemy session
        memory: Memory to index
        snapshot: Optional associated snapshot (for captured_at, app)
        chroma_collection: Optional Chroma collection for vector indexing
        embedding: Optional pre-computed embedding vector
        
    Returns:
        The created/updated SearchIndex entry
    """
    index_id = f"si_mem_{memory.id}"
    
    captured_at = None
    app = None
    if snapshot:
        captured_at = snapshot.captured_at
        app = snapshot.app
    elif memory.snapshot_id:
        # Try to fetch snapshot if not provided
        snap = session.get(Snapshot, memory.snapshot_id)
        if snap:
            captured_at = snap.captured_at
            app = snap.app
    
    # Check if exists
    existing = session.get(SearchIndex, index_id)
    
    if existing:
        existing.search_text = memory.search_text
        existing.title = memory.title
        existing.captured_at = captured_at
        existing.app = app
        existing.updated_at = now_ms()
        entry = existing
    else:
        entry = SearchIndex(
            id=index_id,
            entity_type="memory",
            entity_id=memory.id,
            search_text=memory.search_text,
            title=memory.title,
            captured_at=captured_at,
            app=app,
            created_at=now_ms(),
            updated_at=now_ms(),
        )
        session.add(entry)
    
    session.flush()
    
    # Update Chroma if provided
    if chroma_collection is not None and embedding is not None:
        _upsert_chroma(chroma_collection, entry, embedding)
    
    return entry


def remove_from_index(
    session: Session,
    entity_type: str,
    entity_id: str,
    chroma_collection=None,
) -> bool:
    """
    Remove an entity from the search index.
    
    Args:
        session: SQLAlchemy session
        entity_type: 'snapshot', 'episode', or 'memory'
        entity_id: ID of the entity to remove
        chroma_collection: Optional Chroma collection
        
    Returns:
        True if removed, False if not found
    """
    index_id = f"si_{entity_type[:4]}_{entity_id}"
    
    entry = session.get(SearchIndex, index_id)
    if entry is None:
        return False
    
    session.delete(entry)
    
    # Remove from Chroma if provided
    if chroma_collection is not None:
        try:
            chroma_collection.delete(ids=[index_id])
        except Exception:
            pass  # Ignore if not in Chroma
    
    return True


# =============================================================================
# Chroma Integration
# =============================================================================

def _upsert_chroma(collection, entry: SearchIndex, embedding: List[float]) -> None:
    """Upsert a search index entry to Chroma."""
    metadata = {
        "entity_type": entry.entity_type,
        "entity_id": entry.entity_id,
        "title": entry.title or "",
        "captured_at": entry.captured_at or 0,
        "app": entry.app or "",
    }
    
    collection.upsert(
        ids=[entry.id],
        embeddings=[embedding],
        documents=[entry.search_text],
        metadatas=[metadata],
    )


def get_search_index_collection_name() -> str:
    """Get the Chroma collection name for the search index."""
    return "search_index"


# =============================================================================
# Batch Indexing
# =============================================================================

def reindex_all_snapshots(
    session: Session,
    chroma_collection=None,
    embedding_fn=None,
    verbose: bool = False,
) -> int:
    """
    Reindex all snapshots into the search index.
    
    Args:
        session: SQLAlchemy session
        chroma_collection: Optional Chroma collection
        embedding_fn: Optional function to create embeddings (text) -> List[float]
        verbose: Print progress
        
    Returns:
        Number of snapshots indexed
    """
    from sqlalchemy import select
    
    count = 0
    snapshots = list(session.scalars(select(Snapshot)).all())
    
    for snapshot in snapshots:
        search_text = SearchIndex.build_snapshot_search_text(snapshot)
        if not search_text.strip():
            continue
        
        embedding = None
        if embedding_fn:
            embedding = embedding_fn(search_text)
        
        index_snapshot(session, snapshot, chroma_collection, embedding)
        count += 1
        
        if verbose and count % 100 == 0:
            print(f"  Indexed {count} snapshots...")
    
    return count


def reindex_all_episodes(
    session: Session,
    chroma_collection=None,
    embedding_fn=None,
    verbose: bool = False,
) -> int:
    """
    Reindex all episodes into the search index.
    
    Args:
        session: SQLAlchemy session
        chroma_collection: Optional Chroma collection
        embedding_fn: Optional function to create embeddings
        verbose: Print progress
        
    Returns:
        Number of episodes indexed
    """
    from sqlalchemy import select
    
    count = 0
    episodes = list(session.scalars(select(Episode)).all())
    
    for episode in episodes:
        search_text = SearchIndex.build_episode_search_text(episode)
        if not search_text.strip():
            continue
        
        embedding = None
        if embedding_fn:
            embedding = embedding_fn(search_text)
        
        index_episode(session, episode, chroma_collection, embedding)
        count += 1
        
        if verbose and count % 100 == 0:
            print(f"  Indexed {count} episodes...")
    
    return count


def reindex_all_memories(
    session: Session,
    chroma_collection=None,
    embedding_fn=None,
    verbose: bool = False,
) -> int:
    """
    Reindex all memories into the search index.
    
    Args:
        session: SQLAlchemy session
        chroma_collection: Optional Chroma collection
        embedding_fn: Optional function to create embeddings
        verbose: Print progress
        
    Returns:
        Number of memories indexed
    """
    from sqlalchemy import select
    
    count = 0
    memories = list(session.scalars(select(Memory)).all())
    
    for memory in memories:
        # Get associated snapshot if any
        snapshot = None
        if memory.snapshot_id:
            snapshot = session.get(Snapshot, memory.snapshot_id)
        
        embedding = None
        if embedding_fn:
            embedding = embedding_fn(memory.search_text)
        
        index_memory(session, memory, snapshot, chroma_collection, embedding)
        count += 1
        
        if verbose and count % 100 == 0:
            print(f"  Indexed {count} memories...")
    
    return count


def reindex_all(
    session: Session,
    chroma_collection=None,
    embedding_fn=None,
    verbose: bool = False,
) -> dict:
    """
    Reindex all entities into the search index.
    
    Args:
        session: SQLAlchemy session
        chroma_collection: Optional Chroma collection
        embedding_fn: Optional function to create embeddings
        verbose: Print progress
        
    Returns:
        Dict with counts by entity type
    """
    if verbose:
        print("Reindexing snapshots...")
    snapshots = reindex_all_snapshots(session, chroma_collection, embedding_fn, verbose)
    
    if verbose:
        print("Reindexing episodes...")
    episodes = reindex_all_episodes(session, chroma_collection, embedding_fn, verbose)
    
    if verbose:
        print("Reindexing memories...")
    memories = reindex_all_memories(session, chroma_collection, embedding_fn, verbose)
    
    return {
        "snapshots": snapshots,
        "episodes": episodes,
        "memories": memories,
        "total": snapshots + episodes + memories,
    }

