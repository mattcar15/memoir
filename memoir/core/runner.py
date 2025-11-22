"""
Process management for running window monitor and queue processor concurrently.
"""

import signal
import threading
import time
from pathlib import Path
from typing import Optional

from ..capture.window_monitor import WindowMonitor
from ..capture.queue import ProcessingQueue
from ..processing.queue_processor import QueueProcessor
from ..processing.pipeline.main import ImagePipeline
from ..storage.vector_store import VectorStore
from ..api.server import run_server


# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\n\n🛑 Shutting down gracefully...")
    running = False


def run_window_monitor_process(
    staging_dir: Path,
    queue: ProcessingQueue,
    phash_threshold: int = 10,
    poll_interval: float = 0.5,
    debug: bool = False,
):
    """
    Run the window monitor process.
    
    Args:
        staging_dir: Directory to save captured frames
        queue: ProcessingQueue instance
        phash_threshold: pHash threshold for duplicate detection
        poll_interval: Seconds between window checks
        debug: If True, print debug messages
    """
    def on_capture(frame_path, metadata):
        """Callback when a frame is captured."""
        if debug:
            print(f"✅ Captured: {frame_path.name}")
        queue.enqueue(frame_path, metadata)
        if debug:
            print(f"📥 Queued (pending: {queue.get_pending_count()})")
    
    monitor = WindowMonitor(
        capture_dir=staging_dir,
        on_capture_callback=on_capture,
        phash_threshold=phash_threshold,
        poll_interval=poll_interval,
        debug=debug,
    )
    
    try:
        monitor.start()
        
        # Keep thread alive
        while running:
            time.sleep(1)
            
            # Periodic cleanup
            if int(time.time()) % 3600 == 0:  # Every hour
                monitor.cleanup_old_hashes(max_age_hours=24)
    except Exception as e:
        print(f"❌ Error in window monitor process: {e}")
    finally:
        monitor.stop()
        print("📷 Window monitor stopped")


def run_queue_processor_process(
    queue: ProcessingQueue,
    pipeline: ImagePipeline,
    output_dir: Optional[Path],
    debug: bool = False,
    check_interval: float = 2.0,
):
    """
    Run the queue processor process.
    
    Args:
        queue: ProcessingQueue instance
        pipeline: ImagePipeline instance
        output_dir: Directory to save processed images
        debug: If True, keep captured frames
        check_interval: Seconds between queue checks
    """
    processor = QueueProcessor(
        queue=queue,
        pipeline=pipeline,
        output_dir=output_dir,
        debug=debug,
        check_interval=check_interval,
    )
    
    try:
        processor.start()
        
        # Keep thread alive and run periodic maintenance
        last_maintenance = time.time()
        maintenance_interval = 3600  # 1 hour
        
        while running:
            time.sleep(60)  # Check every minute
            
            # Periodic maintenance
            if time.time() - last_maintenance > maintenance_interval:
                processor.run_maintenance()
                last_maintenance = time.time()
    except Exception as e:
        print(f"❌ Error in queue processor process: {e}")
    finally:
        processor.stop()
        processor.print_stats()
        print("⚙️  Queue processor stopped")


def run_server_process(
    logs_dir: Path,
    vector_store: Optional[VectorStore],
    embedding_model: str = "embeddinggemma",
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """
    Run the server process.
    
    Args:
        logs_dir: Path to logs directory
        vector_store: Optional VectorStore instance
        embedding_model: Embedding model name
        host: Host to bind to
        port: Port to bind to
    """
    try:
        run_server(
            host=host,
            port=port,
            logs_dir=logs_dir,
            vector_store=vector_store,
            embedding_model=embedding_model,
        )
    except KeyboardInterrupt:
        print("🌐 Server process stopped")
    except Exception as e:
        print(f"❌ Error in server process: {e}")


def run_both(
    logs_dir: Path,
    staging_dir: Path,
    output_dir: Optional[Path],
    queue_db: Path,
    vector_store: Optional[VectorStore] = None,
    embedding_model: str = "embeddinggemma",
    similarity_threshold: float = 0.7,
    memory_window: int = 5,
    phash_threshold: int = 10,
    debug: bool = False,
    verbose: bool = False,
    no_llm: bool = False,
    server_host: str = "0.0.0.0",
    server_port: int = 8000,
):
    """
    Run all processes concurrently: window monitor, queue processor, and server.
    
    Args:
        logs_dir: Path to logs directory
        staging_dir: Path to staging directory for captured frames
        output_dir: Path to output directory for processed images
        queue_db: Path to queue database
        vector_store: Optional VectorStore instance
        embedding_model: Embedding model name
        similarity_threshold: Cosine similarity threshold for memory consolidation
        memory_window: Memory window in minutes
        phash_threshold: pHash threshold for duplicate detection
        debug: If True, enable debug mode
        verbose: If True, show detailed processing information
        no_llm: If True, disable MLX VLM processing
        server_host: Host to bind server to
        server_port: Port to bind server to
    """
    global running
    
    # Set up signal handler for graceful shutdown
    try:
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        # Signal handler already set up, or not in main thread
        pass
    
    # Create directories
    staging_dir.mkdir(parents=True, exist_ok=True)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize queue
    queue = ProcessingQueue(queue_db)
    
    # Initialize pipeline
    pipeline = ImagePipeline(
        similarity_threshold=similarity_threshold,
        memory_window_minutes=memory_window,
        phash_threshold=phash_threshold,
        enable_llm=not no_llm,
        embedding_model=embedding_model,
        verbose=verbose,
    )
    
    print("=" * 80)
    print("🚀 Memoir - Intelligent Screenshot Memory System")
    print("=" * 80)
    print(f"📁 Logs: {logs_dir.absolute()}")
    print(f"📁 Staging: {staging_dir.absolute()}")
    print(f"📁 Queue DB: {queue_db.absolute()}")
    print(f"🗄️  Vector store: {vector_store.count() if vector_store else 0} memories")
    print(f"🌐 Server: http://{server_host}:{server_port}")
    print(f"🐛 Debug mode: {debug}")
    print(f"🤖 MLX VLM: {'Enabled' if not no_llm else 'Disabled'}")
    
    pending = queue.get_pending_count()
    if pending > 0:
        print(f"📥 Pending items in queue: {pending}")
    
    print("=" * 80)
    print("Press Ctrl+C to stop all processes\n")
    
    # Start window monitor thread
    monitor_thread = threading.Thread(
        target=run_window_monitor_process,
        args=(staging_dir, queue, phash_threshold, 0.5, debug),
        daemon=True,
    )
    monitor_thread.start()
    
    # Start queue processor thread
    processor_thread = threading.Thread(
        target=run_queue_processor_process,
        args=(queue, pipeline, output_dir, debug, 2.0),
        daemon=True,
    )
    processor_thread.start()
    
    # Start server thread
    server_thread = threading.Thread(
        target=run_server_process,
        args=(logs_dir, vector_store, embedding_model, server_host, server_port),
        daemon=True,
    )
    server_thread.start()
    
    try:
        # Wait for all threads with power-efficient polling
        while running and (
            monitor_thread.is_alive()
            or processor_thread.is_alive()
            or server_thread.is_alive()
        ):
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all processes...")
        running = False
        
        # Wait for threads to finish
        monitor_thread.join(timeout=5)
        processor_thread.join(timeout=10)
        server_thread.join(timeout=5)
        
        if monitor_thread.is_alive():
            print("⚠️  Window monitor did not stop gracefully")
        if processor_thread.is_alive():
            print("⚠️  Queue processor did not stop gracefully")
        if server_thread.is_alive():
            print("⚠️  Server did not stop gracefully")
    
    print("👋 All processes stopped")
