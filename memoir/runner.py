"""
Process management for running capture and server concurrently.
"""

import asyncio
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from .cli import run_capture_loop
from .server import run_server


# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\n\n🛑 Shutting down gracefully...")
    running = False


def run_capture_process(
    resolution_tier: str,
    save_images: bool,
    logs_dir: Path,
    screenshots_dir: Path,
    model_name: str,
    vector_store=None,
    embedding_model: str = "embeddinggemma",
    interval: int = 1,
    run_once: bool = False,
    compare_tiers: bool = False,
):
    """
    Run the capture process in a separate thread.

    Args:
        resolution_tier: Resolution tier to use
        save_images: Whether to save screenshot images
        logs_dir: Path to logs directory
        screenshots_dir: Path to screenshots directory
        model_name: Ollama model name
        vector_store: Optional VectorStore instance
        embedding_model: Embedding model name
        interval: Minutes between captures
        run_once: Whether to run once and exit
        compare_tiers: Whether to compare all resolution tiers
    """
    try:
        run_capture_loop(
            resolution_tier=resolution_tier,
            save_images=save_images,
            logs_dir=logs_dir,
            screenshots_dir=screenshots_dir,
            model_name=model_name,
            vector_store=vector_store,
            embedding_model=embedding_model,
            interval=interval,
            run_once=run_once,
            compare_tiers=compare_tiers,
        )
    except KeyboardInterrupt:
        print("📷 Capture process stopped")
    except Exception as e:
        print(f"❌ Error in capture process: {e}")


def run_server_process(
    logs_dir: Path,
    vector_store,
    embedding_model: str = "embeddinggemma",
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """
    Run the server process in a separate thread.

    Args:
        logs_dir: Path to logs directory
        vector_store: VectorStore instance
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
    resolution_tier: str,
    save_images: bool,
    logs_dir: Path,
    screenshots_dir: Path,
    model_name: str,
    vector_store=None,
    embedding_model: str = "embeddinggemma",
    interval: int = 1,
    run_once: bool = False,
    compare_tiers: bool = False,
    server_host: str = "0.0.0.0",
    server_port: int = 8000,
):
    """
    Run both capture and server processes concurrently.

    Args:
        resolution_tier: Resolution tier to use
        save_images: Whether to save screenshot images
        logs_dir: Path to logs directory
        screenshots_dir: Path to screenshots directory
        model_name: Ollama model name
        vector_store: Optional VectorStore instance
        embedding_model: Embedding model name
        interval: Minutes between captures
        run_once: Whether to run once and exit
        compare_tiers: Whether to compare all resolution tiers
        server_host: Host to bind server to
        server_port: Port to bind server to
    """
    global running

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 80)
    print("🚀 Memoir - Capture + API Server")
    print("=" * 80)
    print(f"📷 Capture: {resolution_tier} tier, {interval}min interval")
    print(f"🌐 Server: http://{server_host}:{server_port}")
    print(f"📁 Logs: {logs_dir.absolute()}")
    print(f"🗄️  Vector store: {vector_store.count() if vector_store else 0} memories")
    print("=" * 80)
    print("Press Ctrl+C to stop both processes\n")

    # Start capture process in a separate thread
    capture_thread = threading.Thread(
        target=run_capture_process,
        args=(
            resolution_tier,
            save_images,
            logs_dir,
            screenshots_dir,
            model_name,
            vector_store,
            embedding_model,
            interval,
            run_once,
            compare_tiers,
        ),
        daemon=True,
    )
    capture_thread.start()

    # Start server process in a separate thread
    server_thread = threading.Thread(
        target=run_server_process,
        args=(logs_dir, vector_store, embedding_model, server_host, server_port),
        daemon=True,
    )
    server_thread.start()

    try:
        # Wait for both threads to complete
        while running and (capture_thread.is_alive() or server_thread.is_alive()):
            time.sleep(1)

            # Check if capture process finished (for run_once mode)
            if run_once and not capture_thread.is_alive():
                print("📷 Capture process completed (run-once mode)")
                break

    except KeyboardInterrupt:
        print("\n🛑 Shutting down both processes...")
        running = False

        # Wait for threads to finish
        capture_thread.join(timeout=5)
        server_thread.join(timeout=5)

        if capture_thread.is_alive():
            print("⚠️  Capture process did not stop gracefully")
        if server_thread.is_alive():
            print("⚠️  Server process did not stop gracefully")

    print("👋 Both processes stopped")


async def run_both_async(
    resolution_tier: str,
    save_images: bool,
    logs_dir: Path,
    screenshots_dir: Path,
    model_name: str,
    vector_store=None,
    embedding_model: str = "embeddinggemma",
    interval: int = 1,
    run_once: bool = False,
    compare_tiers: bool = False,
    server_host: str = "0.0.0.0",
    server_port: int = 8000,
):
    """
    Async version of run_both for use with asyncio.

    Args:
        resolution_tier: Resolution tier to use
        save_images: Whether to save screenshot images
        logs_dir: Path to logs directory
        screenshots_dir: Path to screenshots directory
        model_name: Ollama model name
        vector_store: Optional VectorStore instance
        embedding_model: Embedding model name
        interval: Minutes between captures
        run_once: Whether to run once and exit
        compare_tiers: Whether to compare all resolution tiers
        server_host: Host to bind server to
        server_port: Port to bind server to
    """
    global running

    print("=" * 80)
    print("🚀 Memoir - Capture + API Server (Async)")
    print("=" * 80)
    print(f"📷 Capture: {resolution_tier} tier, {interval}min interval")
    print(f"🌐 Server: http://{server_host}:{server_port}")
    print(f"📁 Logs: {logs_dir.absolute()}")
    print(f"🗄️  Vector store: {vector_store.count() if vector_store else 0} memories")
    print("=" * 80)
    print("Press Ctrl+C to stop both processes\n")

    # Create tasks for both processes
    capture_task = asyncio.create_task(
        asyncio.to_thread(
            run_capture_process,
            resolution_tier,
            save_images,
            logs_dir,
            screenshots_dir,
            model_name,
            vector_store,
            embedding_model,
            interval,
            run_once,
            compare_tiers,
        )
    )

    server_task = asyncio.create_task(
        asyncio.to_thread(
            run_server_process,
            logs_dir,
            vector_store,
            embedding_model,
            server_host,
            server_port,
        )
    )

    try:
        # Wait for both tasks to complete
        await asyncio.gather(capture_task, server_task)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down both processes...")
        running = False

        # Cancel tasks
        capture_task.cancel()
        server_task.cancel()

        # Wait for cancellation to complete
        try:
            await asyncio.gather(capture_task, server_task, return_exceptions=True)
        except Exception:
            pass

    print("👋 Both processes stopped")
