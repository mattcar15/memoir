"""
CLI interface for the Screenshot Memory System with queue-based processing.
"""

import argparse
import signal
import sys
import time
from pathlib import Path

from ..config import (
    CAPTURE_STAGING_DIR,
    QUEUE_DB_PATH,
    DEBUG_MODE,
    PHASH_THRESHOLD_PER_TAB,
    WINDOW_POLL_INTERVAL,
    QUEUE_CHECK_INTERVAL,
)
from ..capture.window_monitor import WindowMonitor
from ..capture.queue import ProcessingQueue
from ..processing.queue_processor import QueueProcessor
from ..processing.pipeline.main import ImagePipeline
from ..storage.vector_store import VectorStore
from ..storage.embeddings import warmup_embedding_model
from .system_monitor import get_global_monitor


# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\n\n🛑 Shutting down gracefully...")
    running = False


def setup_signal_handler():
    """Set up signal handler for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)


def run_capture_only(args):
    """
    Run window monitor to capture frames (without processing).

    Args:
        args: Parsed command-line arguments
    """
    setup_signal_handler()

    # Create directories
    staging_dir = Path(args.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    queue_db = Path(args.queue_db)
    queue_db.parent.mkdir(parents=True, exist_ok=True)

    # Initialize queue
    queue = ProcessingQueue(queue_db)

    # Define capture callback
    def on_capture(frame_path, metadata):
        """Callback when a frame is captured."""
        print(f"✅ Captured: {frame_path.name}")
        queue.enqueue(frame_path, metadata)
        print(f"📥 Queued for processing (pending: {queue.get_pending_count()})")

    # Initialize window monitor
    monitor = WindowMonitor(
        capture_dir=staging_dir,
        on_capture_callback=on_capture,
        phash_threshold=args.phash_threshold,
        poll_interval=args.poll_interval,
        debug=args.debug,
    )

    print("=" * 80)
    print("📸 Window Monitor - Frame Capture")
    print("=" * 80)
    print(f"Staging directory: {staging_dir}")
    print(f"Queue database: {queue_db}")
    print(f"Debug mode: {args.debug}")
    print(f"pHash threshold: {args.phash_threshold}")
    print("=" * 80)
    print("Press Ctrl+C to stop\n")

    # Start monitoring
    monitor.start()

    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    monitor.stop()
    print("👋 Capture stopped")


def run_process_only(args):
    """
    Run queue processor to process captured frames.

    Args:
        args: Parsed command-line arguments
    """
    setup_signal_handler()

    # Create directories
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    queue_db = Path(args.queue_db)

    if not queue_db.exists():
        print(f"❌ Queue database not found: {queue_db}")
        print("Run 'memoir capture' first to start capturing frames")
        sys.exit(1)

    # Initialize queue
    queue = ProcessingQueue(queue_db)

    # Initialize pipeline
    pipeline = ImagePipeline(
        similarity_threshold=args.similarity_threshold,
        memory_window_minutes=args.memory_window,
        phash_threshold=args.phash_threshold,
        enable_llm=not args.no_llm,
        embedding_model=args.embedding_model,
        verbose=args.verbose,
    )

    # Initialize queue processor
    processor = QueueProcessor(
        queue=queue,
        pipeline=pipeline,
        output_dir=output_dir,
        debug=args.debug,
        check_interval=args.check_interval,
    )

    # Initialize system monitor
    sys_monitor = get_global_monitor(verbose=args.verbose)

    print("=" * 80)
    print("⚙️  Queue Processor - Background Processing")
    print("=" * 80)
    print(f"Queue database: {queue_db}")
    print(f"Output directory: {output_dir or 'None (debug mode off)'}")
    print(f"Debug mode: {args.debug}")
    print(f"LLM processing: {'Enabled' if not args.no_llm else 'Disabled'}")

    pending = queue.get_pending_count()
    print(f"\nPending items: {pending}")

    if pending > 0:
        sys_state = sys_monitor.get_system_state(use_cache=False)
        if sys_state:
            print(
                f"System state: {sys_state['recommendation']} "
                f"(battery: {sys_state['battery_pct']}%, "
                f"{'charging' if sys_state['charging'] else 'on battery'})"
            )

    print("=" * 80)
    print("Press Ctrl+C to stop\n")

    # Start processing
    processor.start()

    try:
        while running:
            time.sleep(5)
    except KeyboardInterrupt:
        pass

    processor.stop()
    processor.print_stats()
    print("👋 Processing stopped")


def run_both(args):
    """
    Run both window monitor and queue processor.

    Args:
        args: Parsed command-line arguments
    """
    from .runner import run_both as runner_run_both

    # Set up directories
    logs_dir = Path(args.logs_dir)
    staging_dir = Path(args.staging_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    logs_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize vector store if enabled
    vector_store = None
    if not args.disable_vectorization:
        vector_store = VectorStore(logs_dir / "vector_db")
        if not args.no_warmup:
            warmup_embedding_model(args.embedding_model)

    # Run both processes
    runner_run_both(
        logs_dir=logs_dir,
        staging_dir=staging_dir,
        output_dir=output_dir,
        queue_db=Path(args.queue_db),
        vector_store=vector_store,
        embedding_model=args.embedding_model,
        similarity_threshold=args.similarity_threshold,
        memory_window=args.memory_window,
        phash_threshold=args.phash_threshold,
        debug=args.debug,
        verbose=args.verbose,
        no_llm=args.no_llm,
        server_host=args.host if hasattr(args, "host") else "0.0.0.0",
        server_port=args.port if hasattr(args, "port") else 8000,
    )


def run_search(args):
    """
    Search memories using semantic search.

    Args:
        args: Parsed command-line arguments
    """
    logs_dir = Path(args.logs_dir)
    vector_store = VectorStore(logs_dir / "vector_db")

    if vector_store.count() == 0:
        print("No memories found in vector store")
        return

    print(f"🔍 Searching for: {args.query}")
    print(f"Total memories: {vector_store.count()}\n")

    results = vector_store.search(args.query, n_results=args.results)

    if not results:
        print("No results found")
        return

    print(f"Found {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['timestamp']}]")
        print(f"   {result['content']}")
        print(f"   Similarity: {result.get('similarity', 'N/A'):.3f}\n")


def run():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Memoir - Screenshot Memory System with MLX-based Processing"
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Capture command - window monitoring only
    capture_parser = subparsers.add_parser(
        "capture", help="Run window monitor to capture frames (queues for processing)"
    )
    capture_parser.add_argument(
        "--staging-dir",
        type=str,
        default=str(CAPTURE_STAGING_DIR),
        help=f"Directory for captured frames (default: {CAPTURE_STAGING_DIR})",
    )
    capture_parser.add_argument(
        "--queue-db",
        type=str,
        default=str(QUEUE_DB_PATH),
        help=f"Queue database path (default: {QUEUE_DB_PATH})",
    )
    capture_parser.add_argument(
        "--phash-threshold",
        type=int,
        default=PHASH_THRESHOLD_PER_TAB,
        help=f"pHash threshold for duplicate detection (default: {PHASH_THRESHOLD_PER_TAB})",
    )
    capture_parser.add_argument(
        "--poll-interval",
        type=float,
        default=WINDOW_POLL_INTERVAL,
        help=f"Window polling interval in seconds (default: {WINDOW_POLL_INTERVAL})",
    )
    capture_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (more verbose output)",
    )

    # Process command - queue processing only
    process_parser = subparsers.add_parser(
        "process", help="Run queue processor to process captured frames"
    )
    process_parser.add_argument(
        "--queue-db",
        type=str,
        default=str(QUEUE_DB_PATH),
        help=f"Queue database path (default: {QUEUE_DB_PATH})",
    )
    process_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save processed images (only in debug mode)",
    )
    process_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Cosine similarity threshold for memory consolidation (default: 0.7)",
    )
    process_parser.add_argument(
        "--memory-window",
        type=int,
        default=5,
        help="Memory consolidation window in minutes (default: 5)",
    )
    process_parser.add_argument(
        "--phash-threshold",
        type=int,
        default=10,
        help="pHash threshold for duplicate detection (default: 10)",
    )
    process_parser.add_argument(
        "--check-interval",
        type=float,
        default=QUEUE_CHECK_INTERVAL,
        help=f"Queue check interval in seconds (default: {QUEUE_CHECK_INTERVAL})",
    )
    process_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable MLX VLM processing (OCR only)",
    )
    process_parser.add_argument(
        "--embedding-model",
        type=str,
        default="embeddinggemma",
        help="Embedding model name (default: embeddinggemma)",
    )
    process_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (keep all captured frames)",
    )
    process_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed processing information",
    )

    # Both command - run everything (default)
    both_parser = subparsers.add_parser(
        "both", help="Run both capture and processing (default)"
    )
    both_parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Logs directory (default: logs)",
    )
    both_parser.add_argument(
        "--staging-dir",
        type=str,
        default=str(CAPTURE_STAGING_DIR),
        help=f"Directory for captured frames (default: {CAPTURE_STAGING_DIR})",
    )
    both_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save processed images (only in debug mode)",
    )
    both_parser.add_argument(
        "--queue-db",
        type=str,
        default=str(QUEUE_DB_PATH),
        help=f"Queue database path (default: {QUEUE_DB_PATH})",
    )
    both_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Cosine similarity threshold (default: 0.7)",
    )
    both_parser.add_argument(
        "--memory-window",
        type=int,
        default=5,
        help="Memory window in minutes (default: 5)",
    )
    both_parser.add_argument(
        "--phash-threshold",
        type=int,
        default=PHASH_THRESHOLD_PER_TAB,
        help=f"pHash threshold (default: {PHASH_THRESHOLD_PER_TAB})",
    )
    both_parser.add_argument(
        "--disable-vectorization",
        action="store_true",
        help="Disable vector database storage",
    )
    both_parser.add_argument(
        "--embedding-model",
        type=str,
        default="embeddinggemma",
        help="Embedding model (default: embeddinggemma)",
    )
    both_parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip model warmup",
    )
    both_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable MLX VLM processing",
    )
    both_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    both_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed processing information",
    )
    both_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    both_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )

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
        "--logs-dir",
        type=str,
        default="logs",
        help="Logs directory (default: logs)",
    )
    search_parser.add_argument(
        "--results",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Default to 'both' if no command specified
    if not args.command:
        args.command = "both"
        # Set default values for 'both' command
        args.logs_dir = "logs"
        args.staging_dir = str(CAPTURE_STAGING_DIR)
        args.output_dir = None
        args.queue_db = str(QUEUE_DB_PATH)
        args.similarity_threshold = 0.7
        args.memory_window = 5
        args.phash_threshold = PHASH_THRESHOLD_PER_TAB
        args.disable_vectorization = False
        args.embedding_model = "embeddinggemma"
        args.no_warmup = False
        args.no_llm = False
        args.debug = DEBUG_MODE
        args.verbose = False
        args.host = "0.0.0.0"
        args.port = 8000

    # Route to appropriate handler
    if args.command == "capture":
        run_capture_only(args)
    elif args.command == "process":
        run_process_only(args)
    elif args.command == "both":
        run_both(args)
    elif args.command == "search":
        run_search(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    run()
