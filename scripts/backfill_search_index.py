#!/usr/bin/env python3
"""
Backfill the search_index table from existing snapshots, episodes, and memories.

Usage:
    python scripts/backfill_search_index.py --logs-dir logs
    python scripts/backfill_search_index.py --logs-dir testing_logs --verbose
    python scripts/backfill_search_index.py --logs-dir logs --reset
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Backfill search_index table from existing data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--logs-dir",
        "-l",
        type=Path,
        default=Path("logs"),
        help="Logs directory containing memoir.db (default: logs)",
    )
    
    parser.add_argument(
        "--vector-db",
        type=Path,
        default=None,
        help="Vector DB directory (default: {logs-dir}/vector_db)",
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="embeddinggemma",
        help="Embedding model name (default: embeddinggemma)",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output",
    )
    
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing search index before backfilling",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding creation (default: 100)",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Search Index Backfill")
    print("=" * 70)
    print()
    
    # Validate logs directory
    if not args.logs_dir.exists():
        print(f"❌ Logs directory not found: {args.logs_dir}")
        return 1
    
    db_path = args.logs_dir / "memoir.db"
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("   Run migrations first: python -m memoir.storage.migrations upgrade head")
        return 1
    
    vector_db_path = args.vector_db or (args.logs_dir / "vector_db")
    
    print(f"📁 Logs directory: {args.logs_dir}")
    print(f"🗄️  Database: {db_path}")
    print(f"📊 Vector DB: {vector_db_path}")
    print(f"🧠 Embedding model: {args.embedding_model}")
    print()
    
    # Initialize database and vector store
    from memoir.storage.database import init_database, get_session
    from memoir.storage.vector_store import VectorStore
    from memoir.storage.embeddings import create_embedding
    from memoir.storage.search_index import (
        reindex_all_snapshots,
        get_search_index_collection_name,
    )
    from memoir.storage.models import SearchIndex
    
    print("🔧 Initializing database...")
    init_database(args.logs_dir)
    
    print("🔧 Initializing vector store...")
    vector_store = VectorStore(persist_directory=str(vector_db_path))
    
    # Get or create search index collection
    search_collection = vector_store.client.get_or_create_collection(
        name=get_search_index_collection_name(),
        metadata={"description": "Unified search index for hybrid BM25 + vector search"},
    )
    
    print(f"   Search index collection: {search_collection.count()} entries")
    
    # Reset if requested
    if args.reset:
        print()
        print("⚠️  Resetting search index...")
        
        with get_session() as session:
            # Clear SQLite search_index table
            session.execute(SearchIndex.__table__.delete())
            session.commit()
            print("   ✅ SQLite search_index cleared")
        
        # Clear Chroma collection
        try:
            vector_store.client.delete_collection(get_search_index_collection_name())
            search_collection = vector_store.client.get_or_create_collection(
                name=get_search_index_collection_name(),
                metadata={"description": "Unified search index for hybrid BM25 + vector search"},
            )
            print("   ✅ Chroma search_index collection cleared")
        except Exception as e:
            print(f"   ⚠️  Could not clear Chroma collection: {e}")
    
    print()
    print("🔄 Backfilling search index (snapshots only)...")
    
    # Create embedding function
    def embedding_fn(text: str):
        return create_embedding(text, args.embedding_model)
    
    with get_session() as session:
        # Only index snapshots for now (memories/episodes not fully built out)
        results = reindex_all_snapshots(
            session=session,
            chroma_collection=search_collection,
            embedding_fn=embedding_fn,
            verbose=args.verbose,
        )
        session.commit()
        results = {"snapshots": results, "episodes": 0, "memories": 0, "total": results}
    
    print()
    print("=" * 70)
    print("✅ BACKFILL COMPLETE")
    print("=" * 70)
    print()
    print(f"📊 Indexed:")
    print(f"   Snapshots: {results['snapshots']}")
    print(f"   Episodes:  {results['episodes']}")
    print(f"   Memories:  {results['memories']}")
    print(f"   Total:     {results['total']}")
    print()
    print(f"🗄️  Search index collection now has: {search_collection.count()} entries")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

