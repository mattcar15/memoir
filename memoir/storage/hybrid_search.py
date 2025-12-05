"""
Hybrid search implementation combining BM25 (FTS5) and vector search.

This module provides the core search algorithm that:
1. Queries both FTS5 (BM25) and Chroma (vector) indexes
2. Merges and normalizes scores
3. Blends scores with configurable weights
4. Returns ranked results with entity hydration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from .models import SearchIndex, Snapshot, Episode, Memory


# =============================================================================
# Configuration
# =============================================================================

# Default weights for score blending
DEFAULT_VECTOR_WEIGHT = 0.6
DEFAULT_BM25_WEIGHT = 0.4

# Candidate multiplier (fetch more than K to account for filtering)
CANDIDATE_MULTIPLIER = 3


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class SearchResult:
    """A single search result with scores and entity data."""

    index_id: str
    entity_type: str  # 'snapshot', 'episode', 'memory'
    entity_id: str
    title: Optional[str]
    snippet: str  # Preview of search_text

    # Scores (0-1, higher is better)
    vector_score: float = 0.0
    bm25_score: float = 0.0
    final_score: float = 0.0

    # Metadata for filtering
    captured_at: Optional[int] = None
    app: Optional[str] = None

    # Hydrated entity (filled in after search)
    entity: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "index_id": self.index_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "title": self.title,
            "snippet": self.snippet,
            "score": round(self.final_score, 4),
            "vector_score": round(self.vector_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "captured_at": self.captured_at,
            "app": self.app,
            "entity": self.entity,
        }


# =============================================================================
# Hybrid Search Algorithm
# =============================================================================


def hybrid_search(
    session: Session,
    query: str,
    query_embedding: List[float],
    chroma_collection,
    k: int = 30,
    entity_types: Optional[List[str]] = None,
    app: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
) -> List[SearchResult]:
    """
    Perform hybrid search combining vector and BM25 ranking.

    Args:
        session: SQLAlchemy session
        query: Search query text
        query_embedding: Pre-computed embedding for the query
        chroma_collection: Chroma collection for vector search
        k: Number of results to return
        entity_types: Optional filter for entity types ['snapshot', 'episode', 'memory']
        app: Optional filter for app name
        start_time: Optional start time filter (unix ms)
        end_time: Optional end time filter (unix ms)
        vector_weight: Weight for vector similarity score (0-1)
        bm25_weight: Weight for BM25 score (0-1)

    Returns:
        List of SearchResult objects ranked by blended score
    """
    n_candidates = k * CANDIDATE_MULTIPLIER

    # 1. Get vector search candidates
    vector_results = _vector_search(
        chroma_collection,
        query_embedding,
        n_candidates,
        entity_types,
        app,
        start_time,
        end_time,
    )

    # 2. Get BM25 search candidates
    bm25_results = _bm25_search(
        session,
        query,
        n_candidates,
        entity_types,
        app,
        start_time,
        end_time,
    )

    # 3. Merge results
    merged = _merge_results(vector_results, bm25_results)

    # 4. Normalize scores
    _normalize_scores(merged)

    # 5. Blend scores
    _blend_scores(merged, vector_weight, bm25_weight)

    # 6. Sort by final score and take top K
    ranked = sorted(merged.values(), key=lambda r: r.final_score, reverse=True)[:k]

    # 7. Hydrate with full entity data
    _hydrate_results(session, ranked)

    return ranked


# =============================================================================
# Vector Search (Chroma)
# =============================================================================


def _vector_search(
    collection,
    query_embedding: List[float],
    n_results: int,
    entity_types: Optional[List[str]] = None,
    app: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> Dict[str, SearchResult]:
    """
    Query Chroma for vector similarity candidates.

    Returns dict of index_id -> SearchResult with vector_score populated.
    """
    # Build where filter
    where_filter = _build_chroma_filter(entity_types, app, start_time, end_time)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"Vector search error: {e}")
        return {}

    output = {}

    if results["ids"] and len(results["ids"]) > 0:
        for i in range(len(results["ids"][0])):
            index_id = results["ids"][0][i]
            distance = results["distances"][0][i]
            document = results["documents"][0][i]
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}

            # Convert distance to similarity (cosine distance: 0=identical, 2=opposite)
            similarity = 1 - (distance / 2)

            output[index_id] = SearchResult(
                index_id=index_id,
                entity_type=metadata.get("entity_type", ""),
                entity_id=metadata.get("entity_id", ""),
                title=metadata.get("title"),
                snippet=document[:200] if document else "",
                vector_score=max(0, similarity),  # Ensure non-negative
                captured_at=metadata.get("captured_at"),
                app=metadata.get("app"),
            )

    return output


def _build_chroma_filter(
    entity_types: Optional[List[str]] = None,
    app: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Build Chroma where filter."""
    conditions = []

    if entity_types and len(entity_types) < 3:
        if len(entity_types) == 1:
            conditions.append({"entity_type": entity_types[0]})
        else:
            conditions.append({"entity_type": {"$in": entity_types}})

    if app:
        conditions.append({"app": app})

    if start_time:
        conditions.append({"captured_at": {"$gte": start_time}})

    if end_time:
        conditions.append({"captured_at": {"$lte": end_time}})

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


# =============================================================================
# BM25 Search (FTS5)
# =============================================================================


def _bm25_search(
    session: Session,
    query: str,
    n_results: int,
    entity_types: Optional[List[str]] = None,
    app: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> Dict[str, SearchResult]:
    """
    Query FTS5 for BM25 candidates.

    Returns dict of index_id -> SearchResult with bm25_score populated.
    """
    # Build SQL query with filters
    # FTS5 bm25() returns negative scores (more negative = better match)

    # Escape query for FTS5 (handle special characters)
    safe_query = _escape_fts_query(query)

    if not safe_query:
        return {}

    # Build WHERE clauses for filtering
    where_clauses = ["search_index_fts MATCH :query"]
    params = {"query": safe_query, "limit": n_results}

    if entity_types:
        placeholders = ", ".join(f":type_{i}" for i in range(len(entity_types)))
        where_clauses.append(f"si.entity_type IN ({placeholders})")
        for i, t in enumerate(entity_types):
            params[f"type_{i}"] = t

    if app:
        where_clauses.append("si.app = :app")
        params["app"] = app

    if start_time:
        where_clauses.append("si.captured_at >= :start_time")
        params["start_time"] = start_time

    if end_time:
        where_clauses.append("si.captured_at <= :end_time")
        params["end_time"] = end_time

    where_sql = " AND ".join(where_clauses)

    sql = text(
        f"""
        SELECT 
            si.id,
            si.entity_type,
            si.entity_id,
            si.title,
            si.search_text,
            si.captured_at,
            si.app,
            bm25(search_index_fts) as bm25_score
        FROM search_index_fts
        JOIN search_index si ON si.rowid = search_index_fts.rowid
        WHERE {where_sql}
        ORDER BY bm25_score
        LIMIT :limit
    """
    )

    try:
        result = session.execute(sql, params)
        rows = result.fetchall()
    except Exception as e:
        print(f"BM25 search error: {e}")
        return {}

    output = {}

    for row in rows:
        index_id = row.id
        # BM25 scores are negative; convert to positive (more negative = better)
        # We'll normalize later, so just store the raw score inverted
        raw_bm25 = -row.bm25_score if row.bm25_score else 0

        output[index_id] = SearchResult(
            index_id=index_id,
            entity_type=row.entity_type,
            entity_id=row.entity_id,
            title=row.title,
            snippet=row.search_text[:200] if row.search_text else "",
            bm25_score=raw_bm25,
            captured_at=row.captured_at,
            app=row.app,
        )

    return output


def _escape_fts_query(query: str) -> str:
    """
    Escape a query string for FTS5.

    FTS5 has special syntax for operators. We escape them for literal search.
    """
    if not query:
        return ""

    # Remove or escape special FTS5 characters
    # For simple queries, we just quote the terms
    terms = query.split()
    escaped_terms = []

    for term in terms:
        # Remove special characters that break FTS5
        clean = "".join(c for c in term if c.isalnum() or c in "_-")
        if clean:
            escaped_terms.append(f'"{clean}"')

    if not escaped_terms:
        return ""

    # Join with OR for broader matching
    return " OR ".join(escaped_terms)


# =============================================================================
# Score Merging & Blending
# =============================================================================


def _merge_results(
    vector_results: Dict[str, SearchResult],
    bm25_results: Dict[str, SearchResult],
) -> Dict[str, SearchResult]:
    """
    Merge vector and BM25 results into a single dict.

    Results appearing in both get both scores populated.
    """
    merged = {}

    # Add all vector results
    for index_id, result in vector_results.items():
        merged[index_id] = result

    # Merge BM25 results
    for index_id, result in bm25_results.items():
        if index_id in merged:
            # Update existing with BM25 score
            merged[index_id].bm25_score = result.bm25_score
        else:
            # Add new result (vector score remains 0)
            merged[index_id] = result

    return merged


def _normalize_scores(results: Dict[str, SearchResult]) -> None:
    """
    Normalize vector and BM25 scores to 0-1 range.

    Vector scores use clamping (already 0-1 from cosine similarity).
    BM25 scores use max-based normalization (divide by max) to preserve
    relative ranking while ensuring scores stay in 0-1 range.

    Note: We avoid min-max normalization for BM25 because it maps the
    lowest matching score to 0.0, which loses the distinction between
    "matched poorly" and "didn't match at all".
    """
    if not results:
        return

    # Get score ranges
    vector_scores = [r.vector_score for r in results.values() if r.vector_score > 0]
    bm25_scores = [r.bm25_score for r in results.values() if r.bm25_score > 0]

    # Normalize vector scores (already 0-1 from cosine similarity)
    # But ensure they're in range
    for result in results.values():
        result.vector_score = max(0, min(1, result.vector_score))

    # Normalize BM25 scores using max-based normalization
    # This divides all scores by the maximum, so:
    # - Best match gets 1.0
    # - Other matches get proportional scores (e.g., 0.73, 0.65)
    # - Documents that didn't match stay at 0.0
    if bm25_scores:
        max_bm25 = max(bm25_scores)

        if max_bm25 > 0:
            for result in results.values():
                if result.bm25_score > 0:
                    result.bm25_score = result.bm25_score / max_bm25


def _blend_scores(
    results: Dict[str, SearchResult],
    vector_weight: float,
    bm25_weight: float,
) -> None:
    """
    Compute final blended score for each result.

    final_score = (vector_weight * vector_score) + (bm25_weight * bm25_score)

    Results with only one score type get boosted by the other weight
    to avoid penalizing them too much.
    """
    for result in results.values():
        has_vector = result.vector_score > 0
        has_bm25 = result.bm25_score > 0

        if has_vector and has_bm25:
            # Both scores available - blend normally
            result.final_score = (
                vector_weight * result.vector_score + bm25_weight * result.bm25_score
            )
        elif has_vector:
            # Only vector score - use it with reduced weight
            result.final_score = result.vector_score * (
                vector_weight + bm25_weight * 0.5
            )
        elif has_bm25:
            # Only BM25 score - use it with reduced weight
            result.final_score = result.bm25_score * (bm25_weight + vector_weight * 0.5)
        else:
            result.final_score = 0


# =============================================================================
# Result Hydration
# =============================================================================


def _hydrate_results(session: Session, results: List[SearchResult]) -> None:
    """
    Fetch full entity data for search results.

    Groups by entity type and batch fetches for efficiency.
    """
    # Group by entity type
    by_type: Dict[str, List[SearchResult]] = {
        "snapshot": [],
        "episode": [],
        "memory": [],
    }

    for result in results:
        if result.entity_type in by_type:
            by_type[result.entity_type].append(result)

    # Fetch snapshots
    if by_type["snapshot"]:
        snapshot_ids = [r.entity_id for r in by_type["snapshot"]]
        snapshots = {
            s.id: s
            for s in session.query(Snapshot).filter(Snapshot.id.in_(snapshot_ids)).all()
        }
        for result in by_type["snapshot"]:
            if result.entity_id in snapshots:
                result.entity = snapshots[result.entity_id].to_dict()

    # Fetch episodes
    if by_type["episode"]:
        episode_ids = [r.entity_id for r in by_type["episode"]]
        episodes = {
            e.id: e
            for e in session.query(Episode).filter(Episode.id.in_(episode_ids)).all()
        }
        for result in by_type["episode"]:
            if result.entity_id in episodes:
                result.entity = episodes[result.entity_id].to_dict()

    # Fetch memories
    if by_type["memory"]:
        memory_ids = [r.entity_id for r in by_type["memory"]]
        memories = {
            m.id: m
            for m in session.query(Memory).filter(Memory.id.in_(memory_ids)).all()
        }
        for result in by_type["memory"]:
            if result.entity_id in memories:
                result.entity = memories[result.entity_id].to_dict()


# =============================================================================
# Convenience Functions
# =============================================================================


def search(
    session: Session,
    query: str,
    chroma_collection,
    embedding_fn,
    k: int = 30,
    entity_types: Optional[List[str]] = None,
    app: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper for hybrid_search that handles embedding creation.

    Args:
        session: SQLAlchemy session
        query: Search query text
        chroma_collection: Chroma collection
        embedding_fn: Function to create embeddings (query) -> List[float]
        k: Number of results
        entity_types: Optional type filter
        app: Optional app filter
        start_time: Optional start time filter (unix ms)
        end_time: Optional end time filter (unix ms)
        vector_weight: Vector score weight
        bm25_weight: BM25 score weight

    Returns:
        List of result dictionaries ready for API response
    """
    # Create query embedding
    query_embedding = embedding_fn(query)
    if not query_embedding:
        return []

    # Run hybrid search
    results = hybrid_search(
        session=session,
        query=query,
        query_embedding=query_embedding,
        chroma_collection=chroma_collection,
        k=k,
        entity_types=entity_types,
        app=app,
        start_time=start_time,
        end_time=end_time,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )

    return [r.to_dict() for r in results]
