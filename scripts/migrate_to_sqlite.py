#!/usr/bin/env python3
"""
Migration script: Convert existing JSON files to the new SQLite database schema.

This script reads existing JSON snapshot files from a logs directory and:
1. Creates the SQLite database with the new schema (episodes, snapshots, memories)
2. Imports all data (all as kind='snapshot' memories with no episodes initially)
3. Optionally rebuilds the Chroma index with the new metadata schema

Usage:
    # Migrate existing logs to SQLite
    python scripts/migrate_to_sqlite.py --logs-dir logs

    # Migrate and rebuild Chroma index
    python scripts/migrate_to_sqlite.py --logs-dir testing_logs --rebuild-chroma

    # Dry run (show what would be migrated without making changes)
    python scripts/migrate_to_sqlite.py --logs-dir logs --dry-run
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from dateutil import parser as date_parser

from memoir.storage.database import init_database, get_connection
from memoir.storage.models import Snapshot, Memory
from memoir.storage.repository import upsert_memory_record
from memoir.storage.vector_store import VectorStore
from memoir.storage.embeddings import create_embedding


def parse_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract timestamp from memoir screenshot filename format.
    Format: YYYYMMDD_HHMMSS_mmm_AppName_WindowTitle.png
    """
    try:
        parts = filename.split("_")
        if len(parts) >= 3:
            date_str = parts[0]
            time_str = parts[1]
            ms_str = parts[2]

            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(time_str[:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            ms = int(ms_str)

            return datetime(year, month, day, hour, minute, second, ms * 1000)
    except (ValueError, IndexError):
        pass
    return None


def extract_app_from_filename(filename: str) -> Optional[str]:
    """Extract app name from memoir screenshot filename format."""
    try:
        parts = filename.split("_")
        if len(parts) >= 4:
            return parts[3]
    except (ValueError, IndexError):
        pass
    return None


def extract_window_title_from_filename(filename: str) -> Optional[str]:
    """Extract window title from memoir screenshot filename format."""
    try:
        parts = filename.split("_")
        if len(parts) >= 5:
            title = "_".join(parts[4:])
            if title.endswith(".png"):
                title = title[:-4]
            return title
    except (ValueError, IndexError):
        pass
    return None


def load_json_files(logs_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON files from the logs directory."""
    json_files = list(logs_dir.glob("*.json"))
    
    records = []
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            data["_source_file"] = json_file
            records.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue
    
    return records


def migrate_record(
    data: Dict[str, Any],
    conn,
    chroma_collection,
    embedding_model: str,
    rebuild_chroma: bool,
) -> bool:
    """
    Migrate a single JSON record to SQLite (and optionally Chroma).
    
    Returns True if successful, False otherwise.
    """
    try:
        # Extract data
        memory_id = data.get("memory_id")
        if not memory_id:
            # Try to derive from filename
            source_file = data.get("_source_file")
            if source_file:
                memory_id = source_file.stem
        
        if not memory_id:
            return False
        
        # Parse timestamp
        timestamp_str = data.get("timestamp")
        if timestamp_str:
            try:
                timestamp = date_parser.parse(timestamp_str)
            except:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()
        
        captured_at_ms = int(timestamp.timestamp() * 1000)
        
        # Extract fields
        title = data.get("title", "Untitled")
        summary = data.get("summary", "")
        bullets = data.get("bullets", [])
        tags = data.get("tags", [])
        entities = data.get("entities", [])
        stats = data.get("stats", {})
        screenshot_path = data.get("screenshot_path")
        
        # Try to extract app/window from screenshot path
        app = None
        window_title = None
        if screenshot_path:
            screenshot_name = Path(screenshot_path).name
            app = extract_app_from_filename(screenshot_name)
            window_title = extract_window_title_from_filename(screenshot_name)
        
        # Generate snapshot ID
        snapshot_id = f"snap_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Build search_text
        search_parts = [title or "", summary]
        search_parts.extend(bullets if isinstance(bullets, list) else [])
        search_parts.extend(tags if isinstance(tags, list) else [])
        search_parts.extend(entities if isinstance(entities, list) else [])
        search_text = " ".join(filter(None, search_parts))
        
        # Create Snapshot object
        snapshot = Snapshot(
            id=snapshot_id,
            captured_at=captured_at_ms,
            app=app,
            window_title=window_title,
            image_path=screenshot_path,
            extra=stats,
        )
        
        # Create Memory object
        memory = Memory(
            id=memory_id,
            kind="snapshot",
            snapshot_id=snapshot_id,
            title=title or "Untitled",
            summary=summary or "",
            bullets=bullets if isinstance(bullets, list) else [],
            tags=tags if isinstance(tags, list) else [],
            entities=entities if isinstance(entities, list) else [],
            search_text=search_text,
        )
        
        # Get or create embedding
        embedding = None
        if rebuild_chroma and search_text:
            embedding = create_embedding(search_text, embedding_model)
        
        if embedding is None:
            # Create a dummy embedding if not rebuilding Chroma
            # This allows SQLite migration without Chroma rebuild
            embedding = [0.0] * 768  # Common embedding dimension
        
        # Upsert to database
        upsert_memory_record(
            conn=conn,
            chroma_collection=chroma_collection,
            memory=memory,
            embedding=embedding,
            snapshot=snapshot,
        )
        
        return True
        
    except Exception as e:
        print(f"Error migrating record: {e}")
        return False


def migrate_to_sqlite(
    logs_dir: Path,
    rebuild_chroma: bool = False,
    embedding_model: str = "embeddinggemma",
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Migrate all JSON files to SQLite database.
    
    Args:
        logs_dir: Directory containing JSON files
        rebuild_chroma: Whether to rebuild Chroma index with new embeddings
        embedding_model: Embedding model to use (if rebuilding Chroma)
        dry_run: If True, only report what would be done
        
    Returns:
        Statistics dict with counts
    """
    print(f"📂 Scanning {logs_dir} for JSON files...")
    
    records = load_json_files(logs_dir)
    
    if not records:
        print("❌ No JSON files found")
        return {"error": "No JSON files found"}
    
    print(f"📊 Found {len(records)} JSON files to migrate")
    
    if dry_run:
        print("\n🔍 DRY RUN - No changes will be made\n")
        
        # Show sample of what would be migrated
        print("Sample records:")
        for record in records[:5]:
            print(f"  - {record.get('memory_id', 'unknown')}: {record.get('title', 'Untitled')[:50]}")
        
        if len(records) > 5:
            print(f"  ... and {len(records) - 5} more")
        
        return {
            "would_migrate": len(records),
            "dry_run": True,
        }
    
    # Initialize database
    db_path = init_database(logs_dir)
    print(f"📄 SQLite database initialized at {db_path}")
    
    # Initialize vector store
    vector_db_path = logs_dir / "vector_db"
    vector_store = VectorStore(persist_directory=str(vector_db_path))
    
    if rebuild_chroma:
        print("🔄 Resetting Chroma index for rebuild...")
        vector_store.reset()
    
    # Migrate records
    migrated = 0
    errors = 0
    
    with get_connection(db_path) as conn:
        pbar = tqdm(records, desc="Migrating", unit="record")
        
        for record in pbar:
            success = migrate_record(
                data=record,
                conn=conn,
                chroma_collection=vector_store.collection,
                embedding_model=embedding_model,
                rebuild_chroma=rebuild_chroma,
            )
            
            if success:
                migrated += 1
            else:
                errors += 1
            
            pbar.set_postfix_str(f"Migrated: {migrated}, Errors: {errors}")
    
    return {
        "total": len(records),
        "migrated": migrated,
        "errors": errors,
        "chroma_rebuilt": rebuild_chroma,
        "vector_store_count": vector_store.count(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing JSON files to SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Migrate existing logs to SQLite
    python scripts/migrate_to_sqlite.py --logs-dir logs

    # Migrate and rebuild Chroma index with new embeddings
    python scripts/migrate_to_sqlite.py --logs-dir testing_logs --rebuild-chroma

    # Dry run (show what would be migrated)
    python scripts/migrate_to_sqlite.py --logs-dir logs --dry-run
        """,
    )
    
    parser.add_argument(
        "--logs-dir",
        "-l",
        type=Path,
        default=Path("logs"),
        help="Logs directory containing JSON files (default: logs)",
    )
    
    parser.add_argument(
        "--rebuild-chroma",
        action="store_true",
        help="Rebuild Chroma index with new embeddings (slow, requires embedding model)",
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="embeddinggemma",
        help="Embedding model name (default: embeddinggemma)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔄 Memoir JSON to SQLite Migration")
    print("=" * 70)
    print()
    
    if not args.logs_dir.exists():
        print(f"❌ Logs directory not found: {args.logs_dir}")
        return 1
    
    print(f"📁 Logs directory: {args.logs_dir}")
    print(f"🔄 Rebuild Chroma: {args.rebuild_chroma}")
    print()
    
    stats = migrate_to_sqlite(
        logs_dir=args.logs_dir,
        rebuild_chroma=args.rebuild_chroma,
        embedding_model=args.embedding_model,
        dry_run=args.dry_run,
    )
    
    print()
    print("=" * 70)
    print("📊 MIGRATION COMPLETE")
    print("=" * 70)
    print()
    
    if "error" in stats:
        print(f"❌ Error: {stats['error']}")
        return 1
    
    if stats.get("dry_run"):
        print(f"📝 Would migrate: {stats['would_migrate']} records")
        print("\nRun without --dry-run to perform migration")
    else:
        print(f"✅ Migrated: {stats['migrated']}")
        print(f"❌ Errors: {stats['errors']}")
        print(f"📊 Total: {stats['total']}")
        print()
        if stats.get("chroma_rebuilt"):
            print(f"🗄️  Chroma index rebuilt with {stats['vector_store_count']} entries")
        print()
        print("Your data is now available via SQLite!")
        print(f"  Database: {args.logs_dir / 'memoir.db'}")
        print()
        print("To start the server with the new database:")
        print(f"  python -m memoir.api.server --logs-dir {args.logs_dir}")
    
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

