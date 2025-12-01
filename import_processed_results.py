#!/usr/bin/env python3
"""
Import processed batch results into the memoir system.

This script takes results from batch_process_images.py and:
1. Creates JSON snapshot files in the logs directory
2. Adds entries to the vector store for semantic search
3. Optionally copies screenshots to the appropriate location

Usage:
    # Import from a results JSON file
    python import_processed_results.py --results results.json --logs-dir testing_logs

    # Re-process images and import directly (simulates real-time processing)
    python import_processed_results.py --input-dir day_logs/staging --logs-dir testing_logs --reprocess

    # Specify custom vector store location
    python import_processed_results.py --results results.json --vector-db testing_logs/vector_db
"""

import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
from PIL import Image
from memoir.storage.vector_store import VectorStore
from memoir.storage.embeddings import create_embedding
from memoir.processing.pipeline.main import ImagePipeline


def parse_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract timestamp from memoir screenshot filename format.

    Format: YYYYMMDD_HHMMSS_mmm_AppName_WindowTitle.png
    Example: 20251122_170748_990_Cursor_main.py_—_memoir.png

    Args:
        filename: Screenshot filename

    Returns:
        datetime object or None if parsing fails
    """
    try:
        # Extract the timestamp part (first 18 characters: YYYYMMDD_HHMMSS_mmm)
        parts = filename.split("_")
        if len(parts) >= 3:
            date_str = parts[0]  # YYYYMMDD
            time_str = parts[1]  # HHMMSS
            ms_str = parts[2]  # mmm (milliseconds)

            # Parse components
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


def create_snapshot_json(
    memory_id: str,
    timestamp: datetime,
    summary: str,
    screenshot_path: Optional[Path],
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a snapshot JSON structure compatible with the memoir server.

    Args:
        memory_id: Unique identifier for the memory
        timestamp: Timestamp of the snapshot
        summary: VLM-generated summary
        screenshot_path: Path to screenshot file
        stats: Optional processing stats

    Returns:
        Dictionary ready to be saved as JSON
    """
    return {
        "memory_id": memory_id,
        "timestamp": timestamp.isoformat(),
        "summary": summary,
        "screenshot_path": str(screenshot_path) if screenshot_path else None,
        "stats": stats or {},
        "created_at": datetime.now().isoformat(),
    }


def import_from_results_file(
    results_file: Path,
    logs_dir: Path,
    vector_store: VectorStore,
    embedding_model: str = "embeddinggemma",
    copy_screenshots: bool = True,
) -> Dict[str, Any]:
    """
    Import processed results from a JSON file into the memoir system.

    Args:
        results_file: Path to batch processing results JSON
        logs_dir: Target logs directory
        vector_store: VectorStore instance
        embedding_model: Embedding model name
        copy_screenshots: Whether to copy screenshots to logs/screenshots

    Returns:
        Import statistics
    """
    print(f"📂 Loading results from {results_file}...")

    with open(results_file, "r") as f:
        batch_results = json.load(f)

    results = batch_results.get("results", [])

    if not results:
        print("❌ No results found in file")
        return {"error": "No results found"}

    print(f"📊 Found {len(results)} processed images to import")

    # Create directories
    logs_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = logs_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    imported = 0
    skipped = 0
    errors = 0

    pbar = tqdm(results, desc="Importing", unit="snapshot")

    for result in pbar:
        try:
            # Extract data from result
            summary = result.get("summary")
            source_file = result.get("source_file")
            vlm_stats = result.get("vlm_stats", {})
            result_timestamp = result.get("timestamp")

            if not summary or not source_file:
                skipped += 1
                continue

            source_path = Path(source_file)

            # Parse timestamp from filename FIRST (actual capture time)
            # Fall back to result timestamp (processing time) only if filename parsing fails
            timestamp = parse_timestamp_from_filename(source_path.name)

            if not timestamp and result_timestamp:
                try:
                    from dateutil import parser as date_parser

                    timestamp = date_parser.parse(result_timestamp)
                except:
                    pass

            if not timestamp:
                timestamp = datetime.now()

            # Generate memory_id from timestamp
            memory_id = timestamp.strftime("%Y-%m-%d_%H-%M-%S-%f")

            # Check if already exists
            json_file = logs_dir / f"{memory_id}.json"
            if json_file.exists():
                skipped += 1
                continue

            # Copy screenshot if requested
            screenshot_dest = None
            if copy_screenshots and source_path.exists():
                screenshot_dest = screenshots_dir / source_path.name
                if not screenshot_dest.exists():
                    shutil.copy2(source_path, screenshot_dest)

            # Create embedding for vector store
            pbar.set_postfix_str("Creating embedding...")
            embedding = create_embedding(summary, embedding_model)

            if not embedding:
                tqdm.write(f"⚠️  Failed to create embedding for {source_path.name}")
                errors += 1
                continue

            # Add to vector store
            metadata = {
                "timestamp": timestamp.isoformat(),
                "source_file": str(source_path),
                "screenshot_path": str(screenshot_dest) if screenshot_dest else None,
            }

            if vlm_stats:
                metadata["vlm_stats"] = vlm_stats

            vector_store.add_memory(
                memory_id=memory_id,
                summary=summary,
                embedding=embedding,
                metadata=metadata,
            )

            # Create and save JSON file
            snapshot_data = create_snapshot_json(
                memory_id=memory_id,
                timestamp=timestamp,
                summary=summary,
                screenshot_path=screenshot_dest,
                stats=vlm_stats,
            )

            with open(json_file, "w") as f:
                json.dump(snapshot_data, f, indent=2)

            imported += 1
            pbar.set_postfix_str(f"Imported: {imported}")

        except Exception as e:
            tqdm.write(
                f"❌ Error importing {result.get('source_file', 'unknown')}: {e}"
            )
            errors += 1

    pbar.close()

    return {
        "imported": imported,
        "skipped": skipped,
        "errors": errors,
        "total": len(results),
        "vector_store_count": vector_store.count(),
    }


def import_with_reprocessing(
    input_dir: Path,
    logs_dir: Path,
    vector_store: VectorStore,
    embedding_model: str = "embeddinggemma",
    copy_screenshots: bool = True,
    delay_between: float = 0.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Re-process images and import them as if processing in real-time.

    This simulates the real-time capture flow by processing images
    one at a time and adding them to the system.

    Args:
        input_dir: Directory containing images to process
        logs_dir: Target logs directory
        vector_store: VectorStore instance
        embedding_model: Embedding model name
        copy_screenshots: Whether to copy screenshots
        delay_between: Delay between processing each image (simulates real-time)
        verbose: Print detailed output

    Returns:
        Import statistics
    """
    print(f"📂 Processing images from {input_dir}...")

    # Get image files
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_dir.glob(f"*{ext}"))
        image_paths.extend(input_dir.glob(f"*{ext.upper()}"))

    image_paths = sorted(image_paths, key=lambda p: p.name)

    if not image_paths:
        print(f"❌ No images found in {input_dir}")
        return {"error": "No images found"}

    print(f"📊 Found {len(image_paths)} images to process and import")

    # Create directories
    logs_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = logs_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    # Initialize pipeline (without pHash dedup since we want all images)
    pipeline = ImagePipeline(
        similarity_threshold=0.7,
        memory_window_minutes=5,
        phash_threshold=0,  # Disable dedup for import
        embedding_model=embedding_model,
        verbose=verbose,
    )

    imported = 0
    skipped = 0
    errors = 0

    pbar = tqdm(image_paths, desc="Processing & Importing", unit="img")

    for image_path in pbar:
        pbar.set_postfix_str(
            image_path.name[:25] + "..."
            if len(image_path.name) > 25
            else image_path.name
        )

        try:
            # Parse timestamp from filename
            timestamp = parse_timestamp_from_filename(image_path.name)
            if not timestamp:
                timestamp = datetime.now()

            # Generate memory_id
            memory_id = timestamp.strftime("%Y-%m-%d_%H-%M-%S-%f")

            # Check if already exists
            json_file = logs_dir / f"{memory_id}.json"
            if json_file.exists():
                skipped += 1
                continue

            # Load and process image
            image = Image.open(image_path)
            image.load()

            # Process with VLM
            summary, vlm_stats = pipeline.process_with_vlm(
                pipeline.downscale_image(image, tier="balanced")
            )

            image.close()

            if not summary:
                tqdm.write(f"⚠️  Failed to generate summary for {image_path.name}")
                errors += 1
                continue

            # Create embedding
            embedding = create_embedding(summary, embedding_model)

            if not embedding:
                tqdm.write(f"⚠️  Failed to create embedding for {image_path.name}")
                errors += 1
                continue

            # Copy screenshot
            screenshot_dest = None
            if copy_screenshots:
                screenshot_dest = screenshots_dir / image_path.name
                if not screenshot_dest.exists():
                    shutil.copy2(image_path, screenshot_dest)

            # Add to vector store
            metadata = {
                "timestamp": timestamp.isoformat(),
                "source_file": str(image_path),
                "screenshot_path": str(screenshot_dest) if screenshot_dest else None,
            }

            if vlm_stats:
                metadata["vlm_stats"] = vlm_stats

            vector_store.add_memory(
                memory_id=memory_id,
                summary=summary,
                embedding=embedding,
                metadata=metadata,
            )

            # Create and save JSON file
            snapshot_data = create_snapshot_json(
                memory_id=memory_id,
                timestamp=timestamp,
                summary=summary,
                screenshot_path=screenshot_dest,
                stats=vlm_stats,
            )

            with open(json_file, "w") as f:
                json.dump(snapshot_data, f, indent=2)

            imported += 1

            # Optional delay to simulate real-time
            if delay_between > 0:
                time.sleep(delay_between)

        except Exception as e:
            tqdm.write(f"❌ Error processing {image_path.name}: {e}")
            errors += 1
            if verbose:
                import traceback

                traceback.print_exc()

    pbar.close()

    return {
        "imported": imported,
        "skipped": skipped,
        "errors": errors,
        "total": len(image_paths),
        "vector_store_count": vector_store.count(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Import processed batch results into the memoir system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Import from a results JSON file
    python import_processed_results.py --results results.json --logs-dir testing_logs

    # Re-process images and import (simulates real-time)
    python import_processed_results.py --input-dir day_logs/staging --logs-dir testing_logs --reprocess

    # Custom vector store location
    python import_processed_results.py --results results.json --vector-db testing_logs/vector_db

    # Simulate real-time with delay between images
    python import_processed_results.py --input-dir staging --logs-dir logs --reprocess --delay 1.0
        """,
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--results",
        "-r",
        type=Path,
        help="Path to batch processing results JSON file",
    )
    input_group.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        help="Directory with images to re-process and import",
    )

    parser.add_argument(
        "--logs-dir",
        "-l",
        type=Path,
        default=Path("logs"),
        help="Target logs directory for JSON files (default: logs)",
    )

    parser.add_argument(
        "--vector-db",
        "-v",
        type=Path,
        default=None,
        help="Vector database directory (default: {logs-dir}/vector_db)",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="embeddinggemma",
        help="Embedding model name (default: embeddinggemma)",
    )

    parser.add_argument(
        "--no-copy-screenshots",
        action="store_true",
        help="Don't copy screenshots to logs/screenshots",
    )

    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-process images with VLM (use with --input-dir)",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between processing images in seconds (simulates real-time)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )

    parser.add_argument(
        "--reset-vector-store",
        action="store_true",
        help="Clear the vector store before importing",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("📥 Memoir Results Importer")
    print("=" * 70)
    print()

    # Determine vector store path
    vector_db_path = args.vector_db or (args.logs_dir / "vector_db")

    print(f"📁 Logs directory: {args.logs_dir}")
    print(f"🗄️  Vector store: {vector_db_path}")
    print()

    # Initialize vector store
    print("🔧 Initializing vector store...")
    vector_store = VectorStore(persist_directory=str(vector_db_path))

    if args.reset_vector_store:
        print("⚠️  Resetting vector store...")
        vector_store.reset()

    print(f"   Current entries: {vector_store.count()}")
    print()

    # Run import
    if args.results:
        # Import from results file
        if not args.results.exists():
            print(f"❌ Results file not found: {args.results}")
            return 1

        stats = import_from_results_file(
            results_file=args.results,
            logs_dir=args.logs_dir,
            vector_store=vector_store,
            embedding_model=args.embedding_model,
            copy_screenshots=not args.no_copy_screenshots,
        )
    else:
        # Re-process and import
        if not args.input_dir.exists():
            print(f"❌ Input directory not found: {args.input_dir}")
            return 1

        stats = import_with_reprocessing(
            input_dir=args.input_dir,
            logs_dir=args.logs_dir,
            vector_store=vector_store,
            embedding_model=args.embedding_model,
            copy_screenshots=not args.no_copy_screenshots,
            delay_between=args.delay,
            verbose=args.verbose,
        )

    # Print summary
    print()
    print("=" * 70)
    print("📊 IMPORT COMPLETE")
    print("=" * 70)
    print()

    if "error" in stats:
        print(f"❌ Error: {stats['error']}")
        return 1

    print(f"✅ Imported: {stats['imported']}")
    print(f"⏭️  Skipped: {stats['skipped']}")
    print(f"❌ Errors: {stats['errors']}")
    print(f"📊 Total: {stats['total']}")
    print()
    print(f"🗄️  Vector store now has: {stats['vector_store_count']} entries")
    print()
    print("To start the server with these memories:")
    print(
        f"  python -m memoir.api.server --logs-dir {args.logs_dir} --vector-db {vector_db_path}"
    )
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
