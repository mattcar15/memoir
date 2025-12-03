"""
FastAPI server for snapshot and memory retrieval endpoints.

Uses SQLite (via SQLAlchemy) as the source of truth with Chroma for semantic search.
"""

from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, Path as FastAPIPath
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session

from .retrieval import (
    get_snapshots_in_range,
    get_snapshots_in_range_legacy,
    filter_top_k_by_tokens,
    search_memories,
    search_snapshots,
    load_snapshot_data,
    get_oldest_snapshot_timestamp,
    get_snapshot_by_id,
    get_memory_by_id,
    get_episodes,
    get_episode_with_snapshots,
)
from ..storage.vector_store import VectorStore
from ..storage.database import (
    init_database,
    get_session,
    get_database_info,
    create_db_engine,
    create_session,
)
from ..storage import repository
from . import search as search_module


# Global state for the app
class AppState:
    """Application state container."""

    logs_dir: Path = None
    vector_store: VectorStore = None
    embedding_model: str = "embeddinggemma"
    use_sqlite: bool = True


_state = AppState()


def create_app(
    logs_dir: Path,
    vector_store: VectorStore,
    embedding_model: str = "embeddinggemma",
    use_sqlite: bool = True,
) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        logs_dir: Path to logs directory
        vector_store: VectorStore instance
        embedding_model: Embedding model name
        use_sqlite: Whether to use SQLite (True) or legacy JSON files (False)

    Returns:
        Configured FastAPI application
    """
    # Store in global state
    _state.logs_dir = logs_dir
    _state.vector_store = vector_store
    _state.embedding_model = embedding_model
    _state.use_sqlite = use_sqlite

    # Initialize SQLite database if using it
    if use_sqlite:
        db_path = init_database(logs_dir)
        print(f"📄 SQLite database initialized at {db_path}")

    app = FastAPI(
        title="Memoir API",
        description="API for retrieving screenshot memories, snapshots, and episodes",
        version="2.0.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "*",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Configure and include the unified search router
    from ..storage.search_index import get_search_index_collection_name
    import chromadb
    
    # Get or create the search index collection
    search_collection = vector_store.client.get_or_create_collection(
        name=get_search_index_collection_name(),
        metadata={"description": "Unified search index for hybrid BM25 + vector search"},
    )
    search_module.configure_search(search_collection, embedding_model)
    app.include_router(search_module.router)

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Memoir API",
            "version": "2.0.0",
            "storage": "sqlite" if _state.use_sqlite else "json",
            "endpoints": {
                "health": "/health",
                "me": "/me",
                "search": "/search (unified hybrid search)",
                "search_suggest": "/search/suggest",
                "search_similar": "/search/similar",
                "search_stats": "/search/stats",
                "snapshots_range": "/snapshots/range",
                "snapshots_search": "/snapshots/search",
                "memories_search": "/memories/search",
                "snapshot_by_id": "/snapshots/{snapshot_id}",
                "memory_by_id": "/memories/{memory_id}",
                "episodes": "/episodes",
                "episode_by_id": "/episodes/{episode_id}",
                "images": "/images/{filename}",
            },
        }

    # =========================================================================
    # Snapshot Endpoints
    # =========================================================================

    @app.get("/snapshots/range")
    async def get_snapshots_by_range(
        start_date: str = Query(
            ..., description="Start date in ISO format (e.g., 2025-10-10T00:00:00)"
        ),
        end_date: str = Query(
            ..., description="End date in ISO format (e.g., 2025-10-10T23:59:59)"
        ),
        k: int = Query(
            30, ge=1, le=100, description="Maximum number of snapshots to return"
        ),
        app_filter: Optional[str] = Query(
            None, alias="app", description="Filter by app name"
        ),
        include_stats: bool = Query(False, description="Include stats in response"),
        include_image: bool = Query(False, description="Include image URL in response"),
    ):
        """
        Get snapshots within a time range, prioritized by response token count.
        """
        try:
            if _state.use_sqlite:
                with get_session() as session:
                    snapshots = get_snapshots_in_range(
                        session, start_date, end_date, app_filter, limit=k * 3
                    )
            else:
                snapshots = get_snapshots_in_range_legacy(
                    start_date, end_date, _state.logs_dir
                )

            if not snapshots:
                return {
                    "snapshots": [],
                    "count": 0,
                    "message": "No snapshots found in time range",
                }

            top_snapshots = filter_top_k_by_tokens(snapshots, k)
            formatted_snapshots = [
                load_snapshot_data(snapshot, include_stats, include_image)
                for snapshot in top_snapshots
            ]

            return {
                "snapshots": formatted_snapshots,
                "count": len(formatted_snapshots),
                "total_found": len(snapshots),
                "time_range": {"start": start_date, "end": end_date},
            }

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )

    @app.get("/snapshots/search")
    async def search_snapshots_endpoint(
        query: str = Query(..., description="Search query text"),
        k: int = Query(
            30, ge=1, le=100, description="Maximum number of snapshots to return"
        ),
        threshold: float = Query(
            0.5, ge=0.0, le=1.0, description="Minimum similarity threshold"
        ),
        start_date: Optional[str] = Query(
            None, description="Optional start date filter in ISO format"
        ),
        end_date: Optional[str] = Query(
            None, description="Optional end date filter in ISO format"
        ),
        include_stats: bool = Query(False, description="Include stats in response"),
        include_image: bool = Query(False, description="Include image URL in response"),
    ):
        """
        Search snapshots using semantic similarity with optional time filtering.

        This is the legacy endpoint. For new integrations, use /memories/search.
        """
        try:
            # Use legacy JSON-based search for backward compatibility
            search_results = search_snapshots(
                query=query,
                vector_store=_state.vector_store,
                embedding_model=_state.embedding_model,
                k=k,
                threshold=threshold,
                start_date=start_date,
                end_date=end_date,
                logs_dir=_state.logs_dir,
            )

            if not search_results:
                return {
                    "snapshots": [],
                    "count": 0,
                    "message": f"No snapshots found matching query '{query}' above threshold {threshold}",
                }

            formatted_snapshots = [
                load_snapshot_data(snapshot, include_stats, include_image)
                for snapshot in search_results
            ]

            for i, snapshot in enumerate(search_results):
                if "similarity" in snapshot:
                    formatted_snapshots[i]["similarity"] = round(
                        snapshot["similarity"], 4
                    )

            return {
                "snapshots": formatted_snapshots,
                "count": len(formatted_snapshots),
                "query": query,
                "threshold": threshold,
                "time_filter": (
                    {"start": start_date, "end": end_date}
                    if start_date and end_date
                    else None
                ),
            }

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )

    @app.get("/snapshots/{snapshot_id}")
    async def get_snapshot(
        snapshot_id: str = FastAPIPath(..., description="Snapshot ID"),
        include_stats: bool = Query(False, description="Include stats in response"),
        include_image: bool = Query(False, description="Include image URL in response"),
    ):
        """Get a specific snapshot by ID."""
        if not _state.use_sqlite:
            raise HTTPException(
                status_code=501, detail="Snapshot by ID requires SQLite mode"
            )

        with get_session() as session:
            result = get_snapshot_by_id(
                session, snapshot_id, include_stats, include_image
            )
            if result is None:
                raise HTTPException(status_code=404, detail="Snapshot not found")
            return result

    # =========================================================================
    # Memory Endpoints
    # =========================================================================

    @app.get("/memories/search")
    async def search_memories_endpoint(
        query: str = Query(..., description="Search query text"),
        k: int = Query(
            30, ge=1, le=100, description="Maximum number of memories to return"
        ),
        threshold: float = Query(
            0.3, ge=0.0, le=1.0, description="Minimum similarity threshold"
        ),
        kind: Optional[str] = Query(
            None, description="Filter by memory kind: 'snapshot' or 'episode'"
        ),
        app_filter: Optional[str] = Query(
            None, alias="app", description="Filter by app name"
        ),
        start_date: Optional[str] = Query(
            None, description="Optional start date filter in ISO format"
        ),
        end_date: Optional[str] = Query(
            None, description="Optional end date filter in ISO format"
        ),
    ):
        """
        Search memories using hybrid vector + keyword search.

        Returns memories ranked by relevance with associated snapshot/episode data.
        """
        if not _state.use_sqlite:
            raise HTTPException(
                status_code=501, detail="Memory search requires SQLite mode"
            )

        try:
            with get_session() as session:
                results = search_memories(
                    session=session,
                    query=query,
                    vector_store=_state.vector_store,
                    embedding_model=_state.embedding_model,
                    k=k,
                    threshold=threshold,
                    start_date=start_date,
                    end_date=end_date,
                    kind=kind,
                    app=app_filter,
                )

            if not results:
                return {
                    "memories": [],
                    "count": 0,
                    "message": f"No memories found matching query '{query}' above threshold {threshold}",
                }

            return {
                "memories": results,
                "count": len(results),
                "query": query,
                "threshold": threshold,
                "filters": {
                    "kind": kind,
                    "app": app_filter,
                    "time_range": (
                        {"start": start_date, "end": end_date}
                        if start_date and end_date
                        else None
                    ),
                },
            }

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )

    @app.get("/memories/{memory_id}")
    async def get_memory(
        memory_id: str = FastAPIPath(..., description="Memory ID"),
    ):
        """Get a specific memory by ID with associated snapshot/episode data."""
        if not _state.use_sqlite:
            raise HTTPException(
                status_code=501, detail="Memory by ID requires SQLite mode"
            )

        with get_session() as session:
            result = get_memory_by_id(session, memory_id)
            if result is None:
                raise HTTPException(status_code=404, detail="Memory not found")
            return result

    # =========================================================================
    # Episode Endpoints
    # =========================================================================

    @app.get("/episodes")
    async def list_episodes(
        limit: int = Query(100, ge=1, le=1000, description="Maximum number to return"),
        offset: int = Query(0, ge=0, description="Pagination offset"),
    ):
        """List episodes with pagination."""
        if not _state.use_sqlite:
            raise HTTPException(status_code=501, detail="Episodes require SQLite mode")

        with get_session() as session:
            episodes = get_episodes(session, limit, offset)
            return {
                "episodes": episodes,
                "count": len(episodes),
                "limit": limit,
                "offset": offset,
            }

    @app.get("/episodes/{episode_id}")
    async def get_episode(
        episode_id: str = FastAPIPath(..., description="Episode ID"),
    ):
        """Get an episode with all its snapshots and memories."""
        if not _state.use_sqlite:
            raise HTTPException(status_code=501, detail="Episodes require SQLite mode")

        with get_session() as session:
            result = get_episode_with_snapshots(session, episode_id)
            if result is None:
                raise HTTPException(status_code=404, detail="Episode not found")
            return result

    @app.get("/episodes/{episode_id}/snapshots")
    async def get_episode_snapshots(
        episode_id: str = FastAPIPath(..., description="Episode ID"),
    ):
        """Get all snapshots for an episode."""
        if not _state.use_sqlite:
            raise HTTPException(status_code=501, detail="Episodes require SQLite mode")

        with get_session() as session:
            episode = repository.get_episode_by_id(session, episode_id)
            if episode is None:
                raise HTTPException(status_code=404, detail="Episode not found")

            snapshots = repository.get_snapshots_by_episode(session, episode_id)

            results = []
            for snapshot in snapshots:
                memory = repository.get_memory_by_snapshot(session, snapshot.id)
                results.append(
                    {
                        "snapshot_id": snapshot.id,
                        "captured_at": snapshot.captured_at,
                        "app": snapshot.app,
                        "window_title": snapshot.window_title,
                        "image_path": snapshot.image_path,
                        "memory": (
                            {
                                "id": memory.id,
                                "title": memory.title,
                                "summary": memory.summary,
                            }
                            if memory
                            else None
                        ),
                    }
                )

            return {
                "episode_id": episode_id,
                "snapshots": results,
                "count": len(results),
            }

    # =========================================================================
    # Utility Endpoints
    # =========================================================================

    @app.get("/images/{filename}")
    async def get_image(filename: str = FastAPIPath(..., description="Image filename")):
        """Serve image files from the screenshots directory."""
        screenshots_dir = _state.logs_dir / "screenshots"
        image_path = screenshots_dir / filename

        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        if not image_path.is_file():
            raise HTTPException(status_code=404, detail="Invalid file")

        valid_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
        if image_path.suffix.lower() not in valid_extensions:
            raise HTTPException(status_code=400, detail="Invalid image file")

        return FileResponse(
            path=str(image_path),
            media_type="image/png",
            filename=filename,
        )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        response = {
            "status": "healthy",
            "vector_store_count": _state.vector_store.count(),
            "logs_dir": str(_state.logs_dir),
            "storage_mode": "sqlite" if _state.use_sqlite else "json",
        }

        if _state.use_sqlite:
            try:
                with get_session() as session:
                    info = get_database_info(session)
                    response["database"] = info
            except Exception as e:
                response["database_error"] = str(e)

        return response

    @app.get("/me")
    async def get_user_info():
        """Get user information including the oldest snapshot timestamp."""
        try:
            if _state.use_sqlite:
                with get_session() as session:
                    oldest_timestamp = get_oldest_snapshot_timestamp(session)
                    info = get_database_info(session)
                    total_snapshots = info["counts"]["snapshots"]
                    total_memories = info["counts"]["memories"]
                    total_episodes = info["counts"]["episodes"]

                return {
                    "total_snapshots": total_snapshots,
                    "total_memories": total_memories,
                    "total_episodes": total_episodes,
                    "oldest_snapshot": oldest_timestamp,
                }
            else:
                # Legacy JSON mode
                total_snapshots = len(list(_state.logs_dir.glob("*.json")))
                return {
                    "total_snapshots": total_snapshots,
                    "oldest_snapshot": None,
                }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )

    @app.get("/stats")
    async def get_stats():
        """Get detailed statistics about the stored data."""
        if not _state.use_sqlite:
            raise HTTPException(status_code=501, detail="Stats require SQLite mode")

        with get_session() as session:
            apps = repository.get_apps(session)
            tags = repository.get_tags(session)
            entities = repository.get_entities(session)
            by_app = repository.count_snapshots_by_app(session)
            by_day = repository.count_snapshots_by_day(session)

            return {
                "apps": apps,
                "tags": tags[:50],  # Limit to top 50
                "entities": entities[:50],
                "snapshots_by_app": by_app,
                "snapshots_by_day": by_day[:30],  # Last 30 days
            }

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    logs_dir: Optional[Path] = None,
    vector_store: Optional[VectorStore] = None,
    embedding_model: str = "embeddinggemma",
    use_sqlite: bool = True,
):
    """
    Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        logs_dir: Path to logs directory
        vector_store: VectorStore instance
        embedding_model: Embedding model name
        use_sqlite: Whether to use SQLite (True) or legacy JSON files (False)
    """
    import uvicorn

    if logs_dir is None:
        logs_dir = Path("logs")

    if vector_store is None:
        vector_db_path = logs_dir / "vector_db"
        vector_store = VectorStore(persist_directory=str(vector_db_path))

    app = create_app(logs_dir, vector_store, embedding_model, use_sqlite)

    print(f"🚀 Starting Memoir API server on {host}:{port}")
    print(f"📁 Logs directory: {logs_dir.absolute()}")
    print(f"🗄️  Vector store: {vector_store.count()} memories")
    print(f"💾 Storage mode: {'SQLite' if use_sqlite else 'JSON files'}")
    print(f"📖 API docs available at: http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port)


def main():
    """CLI entry point for running the server standalone."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Memoir API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m memoir.api.server
    python -m memoir.api.server --logs-dir testing_logs
    python -m memoir.api.server --logs-dir testing_logs --port 8080
    python -m memoir.api.server --logs-dir testing_logs --legacy  # Use JSON files
        """,
    )

    parser.add_argument(
        "--logs-dir",
        "-l",
        type=Path,
        default=Path("logs"),
        help="Logs directory containing data (default: logs)",
    )

    parser.add_argument(
        "--vector-db",
        "-v",
        type=Path,
        default=None,
        help="Vector database directory (default: {logs-dir}/vector_db)",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="embeddinggemma",
        help="Embedding model name (default: embeddinggemma)",
    )

    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy JSON file storage instead of SQLite",
    )

    args = parser.parse_args()

    # Determine vector store path
    vector_db_path = args.vector_db or (args.logs_dir / "vector_db")

    # Initialize vector store
    vector_store = VectorStore(persist_directory=str(vector_db_path))

    # Run server
    run_server(
        host=args.host,
        port=args.port,
        logs_dir=args.logs_dir,
        vector_store=vector_store,
        embedding_model=args.embedding_model,
        use_sqlite=not args.legacy,
    )


if __name__ == "__main__":
    main()
