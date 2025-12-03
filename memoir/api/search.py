"""
Unified search API for hybrid BM25 + vector search.

This module provides the /search endpoint that searches across
snapshots, episodes, and memories with configurable weights.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..storage.database import get_session
from ..storage.hybrid_search import search, DEFAULT_VECTOR_WEIGHT, DEFAULT_BM25_WEIGHT
from ..storage.embeddings import create_embedding


# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter(prefix="/search", tags=["search"])


# =============================================================================
# Request/Response Models
# =============================================================================

class SearchResultItem(BaseModel):
    """A single search result - matches legacy snapshot format."""
    
    # Identifiers
    snapshot_id: str
    memory_id: Optional[str] = None
    episode_id: Optional[str] = None
    
    # Timestamps
    timestamp: Optional[str] = None  # ISO format
    captured_at: Optional[int] = None
    
    # Snapshot metadata
    app: Optional[str] = None
    url: Optional[str] = None
    window_title: Optional[str] = None
    image_path: Optional[str] = None
    
    # Memory content (from LLM)
    title: Optional[str] = None
    summary: Optional[str] = None
    bullets: List[str] = []
    tags: List[str] = []
    entities: List[str] = []
    
    # Search scoring
    similarity: float  # Final blended score
    vector_score: float = 0.0
    bm25_score: float = 0.0


class SearchResponse(BaseModel):
    """Response from the search endpoint."""
    
    results: List[SearchResultItem]
    count: int
    query: str
    filters: dict
    weights: dict


# =============================================================================
# Global State (set by server initialization)
# =============================================================================

_chroma_collection = None
_embedding_model = "embeddinggemma"


def configure_search(chroma_collection, embedding_model: str = "embeddinggemma"):
    """Configure the search module with Chroma collection and embedding model."""
    global _chroma_collection, _embedding_model
    _chroma_collection = chroma_collection
    _embedding_model = embedding_model


# =============================================================================
# Endpoints
# =============================================================================

@router.get("", response_model=SearchResponse)
async def unified_search(
    q: str = Query(..., description="Search query text", min_length=1),
    k: int = Query(30, ge=1, le=100, description="Maximum number of results"),
    types: Optional[str] = Query(
        None,
        description="Comma-separated entity types to search: snapshot,episode,memory"
    ),
    app: Optional[str] = Query(None, description="Filter by app name"),
    start: Optional[int] = Query(
        None,
        description="Filter by start time (unix ms)"
    ),
    end: Optional[int] = Query(
        None,
        description="Filter by end time (unix ms)"
    ),
    vector_weight: float = Query(
        DEFAULT_VECTOR_WEIGHT,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity score"
    ),
    bm25_weight: float = Query(
        DEFAULT_BM25_WEIGHT,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 keyword score"
    ),
):
    """
    Unified hybrid search across snapshots, episodes, and memories.
    
    Combines vector similarity (semantic) and BM25 (keyword) search
    with configurable weights for each scoring method.
    
    Results are ranked by blended score and include full entity data.
    """
    if _chroma_collection is None:
        raise HTTPException(
            status_code=503,
            detail="Search not configured. Server initialization required."
        )
    
    # Parse entity types
    entity_types = None
    if types:
        entity_types = [t.strip() for t in types.split(",") if t.strip()]
        valid_types = {"snapshot", "episode", "memory"}
        invalid = set(entity_types) - valid_types
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid entity types: {invalid}. Valid types: {valid_types}"
            )
    
    # Create embedding function
    def embedding_fn(text: str) -> Optional[List[float]]:
        return create_embedding(text, _embedding_model)
    
    try:
        with get_session() as session:
            from ..storage.hybrid_search import hybrid_search
            from ..storage.embeddings import create_embedding
            from ..storage.models import Snapshot, Memory
            from ..storage import repository
            
            # Create embedding
            query_embedding = create_embedding(q, _embedding_model)
            if not query_embedding:
                raise HTTPException(status_code=500, detail="Failed to create query embedding")
            
            # Run hybrid search
            raw_results = hybrid_search(
                session=session,
                query=q,
                query_embedding=query_embedding,
                chroma_collection=_chroma_collection,
                k=k,
                entity_types=entity_types,
                app=app,
                start_time=start,
                end_time=end,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
            )
            
            # Format results to match legacy snapshot format
            formatted_results = []
            for r in raw_results:
                # Get the snapshot
                snapshot = session.get(Snapshot, r.entity_id) if r.entity_type == "snapshot" else None
                memory = None
                
                if snapshot:
                    # Get associated memory if any
                    memory = repository.get_memory_by_snapshot(session, snapshot.id)
                
                item = SearchResultItem(
                    snapshot_id=r.entity_id if r.entity_type == "snapshot" else "",
                    memory_id=memory.id if memory else None,
                    episode_id=snapshot.episode_id if snapshot else None,
                    timestamp=datetime.fromtimestamp(r.captured_at / 1000).isoformat() if r.captured_at else None,
                    captured_at=r.captured_at,
                    app=r.app,
                    url=snapshot.url if snapshot else None,
                    window_title=snapshot.window_title if snapshot else r.title,
                    image_path=snapshot.image_path if snapshot else None,
                    title=memory.title if memory else r.title,
                    summary=memory.summary if memory else None,
                    bullets=memory.bullets if memory else [],
                    tags=memory.tags if memory else [],
                    entities=memory.entities if memory else [],
                    similarity=r.final_score,
                    vector_score=r.vector_score,
                    bm25_score=r.bm25_score,
                )
                formatted_results.append(item)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search error: {str(e)}"
        )
    
    return SearchResponse(
        results=formatted_results,
        count=len(formatted_results),
        query=q,
        filters={
            "types": entity_types,
            "app": app,
            "start": start,
            "end": end,
        },
        weights={
            "vector": vector_weight,
            "bm25": bm25_weight,
        },
    )


@router.get("/suggest")
async def search_suggestions(
    q: str = Query(..., description="Partial query text", min_length=2),
    limit: int = Query(10, ge=1, le=50, description="Maximum suggestions"),
):
    """
    Get search suggestions based on partial query.
    
    Uses FTS5 prefix matching for fast autocomplete.
    """
    from sqlalchemy import text as sql_text
    
    # Escape and add prefix operator
    safe_query = "".join(c for c in q if c.isalnum() or c in " _-")
    if not safe_query:
        return {"suggestions": [], "query": q}
    
    # FTS5 prefix search
    fts_query = f'"{safe_query}"*'
    
    try:
        with get_session() as session:
            result = session.execute(sql_text("""
                SELECT DISTINCT si.title
                FROM search_index_fts
                JOIN search_index si ON si.rowid = search_index_fts.rowid
                WHERE search_index_fts MATCH :query
                AND si.title IS NOT NULL
                ORDER BY bm25(search_index_fts)
                LIMIT :limit
            """), {"query": fts_query, "limit": limit})
            
            suggestions = [row.title for row in result if row.title]
    except Exception:
        suggestions = []
    
    return {
        "suggestions": suggestions,
        "query": q,
    }


@router.get("/similar")
async def find_similar(
    snapshot_id: Optional[str] = Query(None, description="Find items similar to this snapshot"),
    memory_id: Optional[str] = Query(None, description="Find items similar to this memory"),
    episode_id: Optional[str] = Query(None, description="Find items similar to this episode"),
    k: int = Query(10, ge=1, le=50, description="Number of similar items to return"),
    types: Optional[str] = Query(
        None,
        description="Comma-separated entity types to return: snapshot,episode,memory (default: all)"
    ),
    exclude_same_episode: bool = Query(True, description="Exclude items from the same episode"),
):
    """
    Find semantically similar items based on a snapshot, memory, or episode.
    
    Provide exactly one of: snapshot_id, memory_id, or episode_id.
    Returns similar items of all types (snapshots, episodes, memories) ranked by semantic similarity.
    Use the 'types' parameter to filter which types to return.
    """
    from ..storage.models import Snapshot, Memory, Episode, SearchIndex
    from ..storage import repository
    from ..storage.embeddings import create_embedding
    
    if _chroma_collection is None:
        raise HTTPException(
            status_code=503,
            detail="Search not configured. Server initialization required."
        )
    
    # Parse entity types filter
    allowed_types = {"snapshot", "episode", "memory"}
    filter_types = None
    if types:
        filter_types = {t.strip() for t in types.split(",") if t.strip()}
        invalid = filter_types - allowed_types
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid entity types: {invalid}. Valid types: {allowed_types}"
            )
    
    # Count how many IDs provided
    provided = sum(1 for x in [snapshot_id, memory_id, episode_id] if x is not None)
    if provided != 1:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of: snapshot_id, memory_id, or episode_id"
        )
    
    try:
        with get_session() as session:
            source_episode_id = None
            search_text = None
            source_index_id = None  # The ID in the search index
            
            # Get the source item's search text
            if snapshot_id:
                snapshot = session.get(Snapshot, snapshot_id)
                if not snapshot:
                    raise HTTPException(status_code=404, detail="Snapshot not found")
                source_episode_id = snapshot.episode_id
                source_index_id = f"si_snap_{snapshot_id}"
                # Get the memory for this snapshot
                memory = repository.get_memory_by_snapshot(session, snapshot_id)
                if memory:
                    search_text = memory.search_text
                    source_index_id = f"si_mem_{memory.id}"  # Use memory's index if available
                else:
                    search_text = SearchIndex.build_snapshot_search_text(snapshot)
            
            elif memory_id:
                memory = session.get(Memory, memory_id)
                if not memory:
                    raise HTTPException(status_code=404, detail="Memory not found")
                source_episode_id = memory.episode_id
                search_text = memory.search_text
                source_index_id = f"si_mem_{memory_id}"
            
            elif episode_id:
                episode = session.get(Episode, episode_id)
                if not episode:
                    raise HTTPException(status_code=404, detail="Episode not found")
                source_episode_id = episode_id
                search_text = SearchIndex.build_episode_search_text(episode)
                source_index_id = f"si_ep_{episode_id}"
            
            if not search_text:
                return {"similar": [], "count": 0, "message": "No searchable text found for source item"}
            
            # Create embedding for the search text
            query_embedding = create_embedding(search_text, _embedding_model)
            if not query_embedding:
                raise HTTPException(status_code=500, detail="Failed to create embedding")
            
            # Search for similar items (get extra to filter)
            results = _chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=k * 5,  # Get extra to account for filtering
                include=["metadatas", "distances"],
            )
            
            if not results["ids"] or not results["ids"][0]:
                return {"similar": [], "count": 0}
            
            # Format and filter results
            similar_items = []
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                
                # Convert distance to similarity (0-1 scale)
                similarity = 1 - (distance / 2)
                
                # Skip the source item itself
                if doc_id == source_index_id:
                    continue
                
                # Get entity type and ID from metadata
                entity_type = metadata.get("entity_type", "")
                entity_id = metadata.get("entity_id", "")
                
                # Filter by types if specified
                if filter_types and entity_type not in filter_types:
                    continue
                
                # Build the result item based on entity type
                item = {
                    "id": doc_id,
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "similarity": round(similarity, 4),
                    "title": metadata.get("title"),
                    "app": metadata.get("app"),
                    "captured_at": metadata.get("captured_at"),
                    # These will be populated below based on type
                    "snapshot_id": None,
                    "memory_id": None,
                    "episode_id": None,
                    "summary": None,
                    "image_path": None,
                    "tags": [],
                }
                
                # Hydrate from SQLite based on entity type
                if entity_type == "snapshot":
                    snap = session.get(Snapshot, entity_id)
                    if snap:
                        # Optionally exclude same episode
                        if exclude_same_episode and source_episode_id and snap.episode_id == source_episode_id:
                            continue
                        item["snapshot_id"] = snap.id
                        item["episode_id"] = snap.episode_id
                        item["app"] = snap.app
                        item["captured_at"] = snap.captured_at
                        item["image_path"] = snap.image_path
                        item["title"] = snap.window_title or item["title"]
                        # Get memory for this snapshot
                        snap_memory = repository.get_memory_by_snapshot(session, snap.id)
                        if snap_memory:
                            item["memory_id"] = snap_memory.id
                            item["title"] = snap_memory.title
                            item["summary"] = snap_memory.summary
                            item["tags"] = snap_memory.tags
                
                elif entity_type == "memory":
                    mem = session.get(Memory, entity_id)
                    if mem:
                        # Optionally exclude same episode
                        if exclude_same_episode and source_episode_id and mem.episode_id == source_episode_id:
                            continue
                        item["memory_id"] = mem.id
                        item["episode_id"] = mem.episode_id
                        item["title"] = mem.title
                        item["summary"] = mem.summary
                        item["tags"] = mem.tags
                        # Get snapshot for this memory
                        if mem.snapshot_id:
                            snap = session.get(Snapshot, mem.snapshot_id)
                            if snap:
                                item["snapshot_id"] = snap.id
                                item["app"] = snap.app
                                item["captured_at"] = snap.captured_at
                                item["image_path"] = snap.image_path
                
                elif entity_type == "episode":
                    ep = session.get(Episode, entity_id)
                    if ep:
                        # Skip same episode
                        if exclude_same_episode and source_episode_id and ep.id == source_episode_id:
                            continue
                        item["episode_id"] = ep.id
                        item["title"] = ep.title
                        item["summary"] = ep.summary
                        item["tags"] = ep.tags
                        item["captured_at"] = ep.started_at
                
                similar_items.append(item)
                
                if len(similar_items) >= k:
                    break
            
            return {
                "similar": similar_items,
                "count": len(similar_items),
                "source": {
                    "snapshot_id": snapshot_id,
                    "memory_id": memory_id,
                    "episode_id": episode_id,
                },
            }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error finding similar items: {str(e)}"
        )


@router.get("/stats")
async def search_stats():
    """
    Get statistics about the search index.
    """
    from sqlalchemy import func, select
    from ..storage.models import SearchIndex
    
    try:
        with get_session() as session:
            # Count by entity type
            counts = {}
            for entity_type in ["snapshot", "episode", "memory"]:
                count = session.scalar(
                    select(func.count(SearchIndex.id))
                    .where(SearchIndex.entity_type == entity_type)
                )
                counts[entity_type] = count or 0
            
            total = sum(counts.values())
            
            # Get app distribution
            app_counts = {}
            result = session.execute(
                select(SearchIndex.app, func.count(SearchIndex.id))
                .where(SearchIndex.app.isnot(None))
                .group_by(SearchIndex.app)
                .order_by(func.count(SearchIndex.id).desc())
                .limit(20)
            )
            for row in result:
                if row[0]:
                    app_counts[row[0]] = row[1]
            
            # Get time range
            time_range = session.execute(
                select(
                    func.min(SearchIndex.captured_at),
                    func.max(SearchIndex.captured_at)
                )
            ).first()
            
            chroma_count = _chroma_collection.count() if _chroma_collection else 0
            
            return {
                "total_indexed": total,
                "by_type": counts,
                "by_app": app_counts,
                "time_range": {
                    "min": time_range[0] if time_range else None,
                    "max": time_range[1] if time_range else None,
                },
                "chroma_count": chroma_count,
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )

