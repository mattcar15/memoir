"""
CLI interface and orchestration for the Screenshot Memory System.
"""

import argparse
import signal
import sys
import time
from datetime import datetime

import schedule

from .capture import capture_screenshot
from .config import RESOLUTION_TIERS
from .processor import process_with_ollama, warmup_model
from .storage import setup_directories, save_log_entry, save_screenshot
from .vector_store import VectorStore
from .embeddings import create_embedding, warmup_embedding_model
from .live_reload import start_live_reload, stop_live_reload


# Global flag for graceful shutdown
running = True
# Global observer for live reload
live_reload_observer = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running, live_reload_observer
    print("\n\n🛑 Shutting down gracefully...")
    running = False
    if live_reload_observer:
        stop_live_reload(live_reload_observer)


def capture_and_process(
    resolution_tier,
    save_images,
    logs_dir,
    screenshots_dir,
    model_name,
    vector_store=None,
    embedding_model="embeddinggemma",
):
    """
    Main function to capture, process, and log a screenshot.

    Args:
        resolution_tier: Resolution tier to use
        save_images: Whether to save screenshot images
        logs_dir: Path to logs directory
        screenshots_dir: Path to screenshots directory
        model_name: Ollama model name
        vector_store: Optional VectorStore instance for semantic search
        embedding_model: Embedding model name

    Returns:
        Dictionary with timing stats or None if failed
    """
    total_start_time = time.time()

    print(
        f"\n📸 Capturing screenshot at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}..."
    )

    # Capture screenshot
    screenshot, capture_stats = capture_screenshot(resolution_tier)
    if screenshot is None:
        return None

    print(
        f"✅ Screenshot captured ({screenshot.size[0]}x{screenshot.size[1]}) in {capture_stats['capture_time_seconds']}s"
    )

    # Save screenshot if requested
    screenshot_path = None
    save_time = 0
    if save_images:
        save_start = time.time()
        screenshot_path = save_screenshot(screenshot, screenshots_dir)
        save_time = time.time() - save_start
        if screenshot_path:
            print(f"💾 Screenshot saved: {screenshot_path} ({round(save_time, 3)}s)")

    # Process with Ollama
    print(f"🤖 Processing with Ollama ({model_name})...")
    summary, ollama_stats = process_with_ollama(screenshot, model_name)

    if summary is None:
        print("⚠️  Processing failed, skipping log entry")
        return None

    print(
        f"⚡ Ollama processing completed in {ollama_stats['ollama_processing_time_seconds']}s"
    )
    print(
        f"📝 Summary: {summary[:100]}..."
        if len(summary) > 100
        else f"📝 Summary: {summary}"
    )

    # Combine all stats
    total_time = time.time() - total_start_time
    all_stats = {
        **capture_stats,
        **ollama_stats,
        "screenshot_save_time_seconds": round(save_time, 3) if save_images else None,
        "total_processing_time_seconds": round(total_time, 3),
    }

    # Save log entry (with vector store if enabled)
    log_file = save_log_entry(
        summary,
        resolution_tier,
        screenshot_path,
        logs_dir,
        all_stats,
        vector_store=vector_store,
        embedding_model=embedding_model,
    )
    if log_file:
        print(f"✅ Log saved: {log_file}")
        print(f"⏱️  Total time: {all_stats['total_processing_time_seconds']}s")

    return all_stats


def compare_resolution_tiers(
    save_images,
    logs_dir,
    screenshots_dir,
    model_name,
    vector_store=None,
    embedding_model="embeddinggemma",
):
    """
    Compare all three resolution tiers and display results.
    Only works in run-once mode.

    Args:
        save_images: Whether to save screenshot images
        logs_dir: Path to logs directory
        screenshots_dir: Path to screenshots directory
        model_name: Ollama model name
        vector_store: Optional VectorStore instance
        embedding_model: Embedding model name
    """
    print("\n" + "=" * 80)
    print("🔬 RESOLUTION TIER COMPARISON MODE")
    print("=" * 80)
    print("Testing all three resolution tiers with the same screenshot...\n")

    results = {}

    for tier in ["efficient", "balanced", "detailed"]:
        print(f"\n{'='*80}")
        print(
            f"Testing tier: {tier.upper()} ({RESOLUTION_TIERS[tier][0]}x{RESOLUTION_TIERS[tier][1]})"
        )
        print(f"{'='*80}")

        stats = capture_and_process(
            tier,
            save_images,
            logs_dir,
            screenshots_dir,
            model_name,
            vector_store=vector_store,
            embedding_model=embedding_model,
        )
        if stats:
            results[tier] = stats
        else:
            print(f"❌ Failed to process {tier} tier")

    # Display comparison table
    print("\n" + "=" * 80)
    print("📊 COMPARISON RESULTS")
    print("=" * 80)

    if not results:
        print("❌ No results to compare")
        return

    # Print comparison table
    print(f"\n{'Metric':<40} {'Efficient':<15} {'Balanced':<15} {'Detailed':<15}")
    print("-" * 85)

    metrics = [
        ("Original Resolution", "original_resolution"),
        ("Scaled Resolution", "scaled_resolution"),
        ("Image Size (KB)", "image_size_kb"),
        ("Base64 Size (KB)", "image_base64_size_kb"),
        ("Capture Time (s)", "capture_time_seconds"),
        ("Resize Time (s)", "resize_time_seconds"),
        ("Ollama Processing (s)", "ollama_processing_time_seconds"),
        ("Total Time (s)", "total_processing_time_seconds"),
        ("Response Length (chars)", "response_length_chars"),
        ("Response Word Count", "response_word_count"),
    ]

    for label, key in metrics:
        row = f"{label:<40}"
        for tier in ["efficient", "balanced", "detailed"]:
            if tier in results and key in results[tier]:
                value = results[tier][key]
                row += f" {str(value):<15}"
            else:
                row += f" {'N/A':<15}"
        print(row)

    # Token counts if available
    if any("prompt_eval_token_count" in results.get(tier, {}) for tier in results):
        print("\n" + "-" * 85)
        print("Token Counts (if available):")
        print(f"{'Metric':<40} {'Efficient':<15} {'Balanced':<15} {'Detailed':<15}")
        print("-" * 85)

        token_metrics = [
            ("Prompt Tokens", "prompt_eval_token_count"),
            ("Response Tokens", "response_token_count"),
        ]

        for label, key in token_metrics:
            row = f"{label:<40}"
            for tier in ["efficient", "balanced", "detailed"]:
                if tier in results and key in results[tier]:
                    value = results[tier][key]
                    row += f" {str(value):<15}"
                else:
                    row += f" {'N/A':<15}"
            print(row)

    # Speed comparison
    print("\n" + "=" * 80)
    print("⚡ SPEED COMPARISON")
    print("=" * 80)

    if all(tier in results for tier in ["efficient", "balanced", "detailed"]):
        fastest = min(
            results.keys(), key=lambda t: results[t]["total_processing_time_seconds"]
        )
        slowest = max(
            results.keys(), key=lambda t: results[t]["total_processing_time_seconds"]
        )

        fastest_time = results[fastest]["total_processing_time_seconds"]
        slowest_time = results[slowest]["total_processing_time_seconds"]
        speedup = slowest_time / fastest_time if fastest_time > 0 else 0

        print(f"⭐ Fastest: {fastest.upper()} ({fastest_time}s)")
        print(f"🐌 Slowest: {slowest.upper()} ({slowest_time}s)")
        print(f"📈 Speedup: {round(speedup, 2)}x faster")

        # Size comparison
        print("\n" + "=" * 80)
        print("💾 SIZE COMPARISON")
        print("=" * 80)

        for tier in ["efficient", "balanced", "detailed"]:
            size_kb = results[tier]["image_size_kb"]
            print(f"{tier.upper():<15} {size_kb} KB")


def search_memories(
    query_text, vector_store, embedding_model="embeddinggemma", n_results=5
):
    """
    Search memories using semantic search.

    Args:
        query_text: Search query
        vector_store: VectorStore instance
        embedding_model: Embedding model name
        n_results: Number of results to return
    """
    print(f'\n🔍 Searching memories for: "{query_text}"')
    print("=" * 80)

    # Create embedding for query
    print("🔄 Creating query embedding...")
    query_embedding = create_embedding(query_text, embedding_model)

    if not query_embedding:
        print("❌ Failed to create query embedding")
        return

    # Search vector store
    results = vector_store.search_by_text(query_text, query_embedding, n_results)

    if not results:
        print("📭 No results found")
        return

    # Display results
    print("\n" + "=" * 80)
    print(f"📊 Found {len(results)} similar memories:")
    print("=" * 80)

    for i, result in enumerate(results):
        print(f"\n{'='*80}")
        print(
            f"Result #{i+1} - Distance: {round(result['distance'], 4)} - ID: {result['id']}"
        )
        print(f"{'='*80}")
        print(f"Summary: {result['summary'][:300]}...")
        if "timestamp" in result["metadata"]:
            print(f"Timestamp: {result['metadata']['timestamp']}")
        print()


def run():
    """Main application entry point"""
    global running, live_reload_observer

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Screenshot Memory System - Capture and analyze screen activity with Ollama"
    )

    # Add subcommands (optional - for search)
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Search command
    search_parser = subparsers.add_parser(
        "search", help="Search memories using semantic search"
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query text",
    )
    search_parser.add_argument(
        "--results",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    search_parser.add_argument(
        "--embedding-model",
        type=str,
        default="embeddinggemma",
        help="Embedding model name (default: embeddinggemma)",
    )

    # Capture arguments (at top level for backward compatibility)
    parser.add_argument(
        "--resolution-tier",
        choices=["efficient", "balanced", "detailed"],
        default="detailed",
        help="Resolution tier for screenshots (default: detailed - recommended, no performance penalty)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save screenshot images alongside logs",
    )
    parser.add_argument(
        "--interval", type=int, default=1, help="Minutes between captures (default: 1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:4b",
        help="Ollama model name (default: gemma3:4b)",
    )
    parser.add_argument(
        "--run-once", action="store_true", help="Run once and exit (useful for testing)"
    )
    parser.add_argument(
        "--compare-tiers",
        action="store_true",
        help="Compare all resolution tiers (only works with --run-once)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip model warmup at startup (not recommended)",
    )
    parser.add_argument(
        "--disable-vectorization",
        action="store_true",
        help="Disable vectorization and vector database storage (vectorization is ON by default)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="embeddinggemma",
        help="Embedding model name (default: embeddinggemma)",
    )
    parser.add_argument(
        "--live-reload",
        action="store_true",
        help="Enable live reload on code changes",
    )

    # Parse arguments
    args = parser.parse_args()

    # Default to capture if no command specified
    if args.command is None:
        args.command = "capture"

    # Handle search command
    if args.command == "search":
        print("=" * 80)
        print("🔍 Memory Search")
        print("=" * 80)

        # Initialize vector store
        vector_store = VectorStore()

        if vector_store.count() == 0:
            print("📭 No memories in vector store yet!")
            print(
                "   Run with --enable-vectorization to start building your memory database"
            )
            return

        # Perform search
        search_memories(args.query, vector_store, args.embedding_model, args.results)
        return

    # Capture mode continues below
    # Validate arguments
    if args.compare_tiers and not args.run_once:
        print("❌ Error: --compare-tiers requires --run-once flag")
        sys.exit(1)

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Create necessary directories
    logs_dir, screenshots_dir = setup_directories()

    # Initialize vector store (enabled by default unless explicitly disabled)
    vector_store = None
    enable_vectorization = not args.disable_vectorization
    if enable_vectorization:
        vector_store = VectorStore()
        print(f"🗄️  Vector store enabled: {vector_store.count()} memories stored")
    else:
        print("⚠️  Vector database disabled - search will not be available")

    print("=" * 60)
    print("📷 Screenshot Memory System")
    print("=" * 60)
    if not args.compare_tiers:
        print(
            f"Resolution tier: {args.resolution_tier} ({RESOLUTION_TIERS[args.resolution_tier][0]}x{RESOLUTION_TIERS[args.resolution_tier][1]})"
        )
    print(f"Save images: {'Yes' if args.save_images else 'No'}")
    print(f"Vectorization: {'Yes ✓' if enable_vectorization else 'No (disabled)'}")
    print(f"Interval: {args.interval} minute(s)")
    print(f"Model: {args.model}")
    print(f"Logs directory: {logs_dir.absolute()}")
    print("=" * 60)

    # Warm up the models (unless disabled)
    if not args.no_warmup:
        print()
        warmup_model(args.model)

        # Warm up embedding model if vectorization is enabled
        if enable_vectorization:
            warmup_embedding_model(args.embedding_model)
        print()

    # Enable live reload if requested
    if args.live_reload:
        live_reload_observer = start_live_reload("memoir")
        print()

    if not args.run_once:
        print("Press Ctrl+C to stop\n")

    # Compare tiers mode
    if args.compare_tiers:
        compare_resolution_tiers(
            args.save_images,
            logs_dir,
            screenshots_dir,
            args.model,
            vector_store=vector_store,
            embedding_model=args.embedding_model,
        )
        return

    # Run once mode for testing
    if args.run_once:
        capture_and_process(
            args.resolution_tier,
            args.save_images,
            logs_dir,
            screenshots_dir,
            args.model,
            vector_store=vector_store,
            embedding_model=args.embedding_model,
        )
        print("\n✅ Single capture completed!")
        return

    # Schedule the job
    schedule.every(args.interval).minutes.do(
        capture_and_process,
        resolution_tier=args.resolution_tier,
        save_images=args.save_images,
        logs_dir=logs_dir,
        screenshots_dir=screenshots_dir,
        model_name=args.model,
        vector_store=vector_store,
        embedding_model=args.embedding_model,
    )

    # Run immediately on start
    capture_and_process(
        args.resolution_tier,
        args.save_images,
        logs_dir,
        screenshots_dir,
        args.model,
        vector_store=vector_store,
        embedding_model=args.embedding_model,
    )

    # Main loop
    while running:
        schedule.run_pending()
        time.sleep(1)

    print("👋 Goodbye!")
