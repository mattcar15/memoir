"""
Live reload functionality using watchdog to monitor code changes.
"""

import sys
import os
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class CodeChangeHandler(FileSystemEventHandler):
    """Handler for code file changes."""

    def __init__(self, callback):
        """
        Initialize the handler.

        Args:
            callback: Function to call when code changes are detected
        """
        self.callback = callback
        self.last_modified = time.time()
        self.debounce_seconds = 1.0  # Debounce to avoid multiple reloads

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        # Only watch Python files
        if not event.src_path.endswith(".py"):
            return

        # Debounce rapid changes
        current_time = time.time()
        if current_time - self.last_modified < self.debounce_seconds:
            return

        self.last_modified = current_time

        print(f"\n🔄 Code change detected: {event.src_path}")
        print("♻️  Restarting application...\n")

        # Call the callback
        self.callback()


def restart_program():
    """Restart the current program."""
    try:
        # Get the Python executable and arguments
        python = sys.executable
        os.execl(python, python, *sys.argv)
    except Exception as e:
        print(f"❌ Error restarting program: {e}")
        sys.exit(1)


def start_live_reload(watch_directory: str = "memoir"):
    """
    Start watching for code changes and enable live reload.

    Args:
        watch_directory: Directory to watch for changes

    Returns:
        Observer instance
    """
    watch_path = Path(watch_directory)

    if not watch_path.exists():
        print(f"⚠️  Watch directory not found: {watch_directory}")
        return None

    # Create event handler and observer
    event_handler = CodeChangeHandler(restart_program)
    observer = Observer()
    observer.schedule(event_handler, str(watch_path), recursive=True)
    observer.start()

    print(f"👀 Live reload enabled - watching: {watch_path.absolute()}")
    print("   Any .py file changes will trigger automatic restart")

    return observer


def stop_live_reload(observer):
    """
    Stop the live reload observer.

    Args:
        observer: Observer instance to stop
    """
    if observer:
        observer.stop()
        observer.join()
        print("🛑 Live reload stopped")
