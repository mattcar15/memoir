"""
FastAPI server for snapshot retrieval endpoints.
"""

import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, Path as FastAPIPath
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .retrieval import (
    get_snapshots_in_range,
    filter_top_k_by_tokens,
    search_snapshots,
    load_snapshot_data,
    get_oldest_snapshot_timestamp,
)
from .vector_store import VectorStore
from .embeddings import create_embedding


def create_app(
    logs_dir: Path, vector_store: VectorStore, embedding_model: str = "embeddinggemma"
) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        logs_dir: Path to logs directory
        vector_store: VectorStore instance
        embedding_model: Embedding model name

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Memoir API",
        description="API for retrieving screenshot memories and snapshots",
        version="1.0.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "*",  # Allow all origins for development
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Memoir API",
            "version": "1.0.0",
            "endpoints": {
                "me": "/me",
                "range": "/snapshots/range",
                "search": "/snapshots/search",
                "images": "/images/{filename}",
            },
        }

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
        include_stats: bool = Query(False, description="Include stats in response"),
        include_image: bool = Query(False, description="Include image URL in response"),
    ):
        """
        Get snapshots within a time range, prioritized by response token count.
        """
        try:
            # Get snapshots in time range
            snapshots = get_snapshots_in_range(start_date, end_date, logs_dir)

            if not snapshots:
                return {
                    "snapshots": [],
                    "count": 0,
                    "message": "No snapshots found in time range",
                }

            # Filter to top K by token count
            top_snapshots = filter_top_k_by_tokens(snapshots, k)

            # Format response
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
        """
        try:
            # Search snapshots
            search_results = search_snapshots(
                query=query,
                vector_store=vector_store,
                embedding_model=embedding_model,
                k=k,
                threshold=threshold,
                start_date=start_date,
                end_date=end_date,
                logs_dir=logs_dir,
            )

            if not search_results:
                return {
                    "snapshots": [],
                    "count": 0,
                    "message": f"No snapshots found matching query '{query}' above threshold {threshold}",
                }

            # Format response
            formatted_snapshots = [
                load_snapshot_data(snapshot, include_stats, include_image)
                for snapshot in search_results
            ]

            # Add similarity scores if available
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

    @app.get("/images/{filename}")
    async def get_image(filename: str = FastAPIPath(..., description="Image filename")):
        """
        Serve image files from the screenshots directory.
        """
        screenshots_dir = logs_dir / "screenshots"
        image_path = screenshots_dir / filename

        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        if not image_path.is_file():
            raise HTTPException(status_code=404, detail="Invalid file")

        # Check if it's a valid image file
        valid_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
        if image_path.suffix.lower() not in valid_extensions:
            raise HTTPException(status_code=400, detail="Invalid image file")

        return FileResponse(
            path=str(image_path),
            media_type="image/png",  # Default to PNG, could be more sophisticated
            filename=filename,
        )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "vector_store_count": vector_store.count(),
            "logs_dir": str(logs_dir),
        }

    @app.get("/me")
    async def get_user_info():
        """
        Get user information including the oldest snapshot timestamp.
        """
        try:
            oldest_timestamp = get_oldest_snapshot_timestamp(logs_dir)
            total_snapshots = len(list(logs_dir.glob("*.json")))

            return {
                "total_snapshots": total_snapshots,
                "oldest_snapshot": oldest_timestamp,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    logs_dir: Optional[Path] = None,
    vector_store: Optional[VectorStore] = None,
    embedding_model: str = "embeddinggemma",
):
    """
    Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        logs_dir: Path to logs directory
        vector_store: VectorStore instance
        embedding_model: Embedding model name
    """
    import uvicorn

    if logs_dir is None:
        logs_dir = Path("logs")

    if vector_store is None:
        vector_store = VectorStore()

    app = create_app(logs_dir, vector_store, embedding_model)

    print(f"🚀 Starting Memoir API server on {host}:{port}")
    print(f"📁 Logs directory: {logs_dir.absolute()}")
    print(f"🗄️  Vector store: {vector_store.count()} memories")
    print(f"📖 API docs available at: http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port)
