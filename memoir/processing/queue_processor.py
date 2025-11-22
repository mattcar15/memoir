"""
Queue processor for asynchronous image processing.

This module processes queued images from the capture system,
using adaptive processing based on system resources.
"""

import time
import threading
from pathlib import Path
from typing import Optional
from PIL import Image

from ..capture.queue import ProcessingQueue
from ..core.system_monitor import get_global_monitor
from .mlx_processor import get_global_processor
from .pipeline.main import ImagePipeline


class QueueProcessor:
    """
    Background processor for queued images.
    
    Continuously processes images from the queue, adapting to system
    resources (battery, thermal state, etc.).
    """
    
    def __init__(
        self,
        queue: ProcessingQueue,
        pipeline: ImagePipeline,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        check_interval: float = 2.0,
    ):
        """
        Initialize the queue processor.
        
        Args:
            queue: ProcessingQueue instance
            pipeline: ImagePipeline instance for processing images
            output_dir: Directory to save processed images (if any)
            debug: If True, keep captured frames after processing
            check_interval: Seconds between queue checks when paused/slow
        """
        self.queue = queue
        self.pipeline = pipeline
        self.output_dir = Path(output_dir) if output_dir else None
        self.debug = debug
        self.check_interval = check_interval
        
        # System monitor
        self.monitor = get_global_monitor()
        
        # MLX processor (shared instance)
        self.mlx_processor = get_global_processor()
        
        # Control flags
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            "items_processed": 0,
            "items_failed": 0,
            "items_skipped": 0,
            "total_processing_time": 0.0,
        }
    
    def _log(self, message: str):
        """Log message with timestamp."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[QueueProcessor {timestamp}] {message}")
    
    def _should_keep_model_loaded(self) -> bool:
        """
        Determine if the MLX model should stay loaded.
        
        Returns:
            True if model should stay loaded
        """
        # Check queue and battery status
        pending_count = self.queue.get_pending_count()
        on_battery = self.monitor.is_on_battery()
        
        return self.mlx_processor.should_keep_loaded(
            queue_has_items=(pending_count > 0),
            on_battery=on_battery,
        )
    
    def _manage_model_lifecycle(self):
        """Manage MLX model loading/unloading based on queue and battery state."""
        should_keep = self._should_keep_model_loaded()
        
        if should_keep and not self.mlx_processor.is_loaded:
            self._log("Loading MLX model...")
            self.mlx_processor.load_model()
        elif not should_keep and self.mlx_processor.is_loaded:
            self._log("Unloading MLX model (queue empty or on battery)...")
            self.mlx_processor.unload_model()
    
    def _process_item(self, item: dict) -> bool:
        """
        Process a single queue item.
        
        Args:
            item: Queue item dictionary
            
        Returns:
            True if successful, False if failed
        """
        item_id = item["id"]
        image_path = item["image_path"]
        metadata = item["metadata"]
        
        try:
            self._log(f"Processing item {item_id}: {image_path.name}")
            
            # Load image
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = Image.open(image_path)
            
            # Process through pipeline
            start_time = time.time()
            result = self.pipeline.process_image(
                image,
                output_dir=self.output_dir,
                save_images=self.debug,  # Only save if debug mode
            )
            processing_time = time.time() - start_time
            
            if result:
                self._log(
                    f"Item {item_id} processed successfully in {processing_time:.2f}s "
                    f"(method: {result.get('processing_method', 'unknown')})"
                )
                
                # Mark as completed
                self.queue.mark_completed(item_id)
                
                # Delete image if not in debug mode
                if not self.debug:
                    try:
                        image_path.unlink()
                    except Exception as e:
                        self._log(f"Warning: Failed to delete {image_path}: {e}")
                
                # Update stats
                self.stats["items_processed"] += 1
                self.stats["total_processing_time"] += processing_time
                
                return True
            else:
                # Processing returned None (likely duplicate)
                self._log(f"Item {item_id} skipped (duplicate or error)")
                self.queue.mark_completed(item_id)  # Mark as completed anyway
                
                # Delete image
                if not self.debug:
                    try:
                        image_path.unlink()
                    except Exception as e:
                        self._log(f"Warning: Failed to delete {image_path}: {e}")
                
                self.stats["items_skipped"] += 1
                return True
                
        except Exception as e:
            self._log(f"Error processing item {item_id}: {e}")
            
            # Mark as failed (with retry if it's the first failure)
            retry = item.get("retry_count", 0) < 2
            self.queue.mark_failed(item_id, str(e), retry=retry)
            
            self.stats["items_failed"] += 1
            return False
    
    def _process_loop(self):
        """Main processing loop (runs in background thread)."""
        self._log("Queue processor started")
        
        # Reset any stale processing items on startup
        stale_count = self.queue.reset_stale_processing(timeout_minutes=30)
        if stale_count > 0:
            self._log(f"Reset {stale_count} stale processing items")
        
        while self.running:
            try:
                # Manage model lifecycle
                self._manage_model_lifecycle()
                
                # Get system recommendation
                recommendation = self.monitor.get_recommendation()
                
                if recommendation == "PAUSE":
                    # Don't process, just wait
                    self._log("System recommendation: PAUSE - waiting...")
                    time.sleep(self.check_interval)
                    continue
                
                # Try to get next item
                item = self.queue.dequeue()
                
                if not item:
                    # Queue is empty, wait a bit
                    time.sleep(self.check_interval)
                    continue
                
                # Process the item
                self._process_item(item)
                
                # If recommendation is SLOW, add delay between items
                if recommendation == "SLOW":
                    self._log("System recommendation: SLOW - adding delay...")
                    time.sleep(10.0)
                
                # Small delay even in RUN mode to prevent tight loop
                time.sleep(0.5)
                
            except Exception as e:
                self._log(f"Error in processing loop: {e}")
                time.sleep(self.check_interval)
        
        # Cleanup on exit
        if self.mlx_processor.is_loaded:
            self._log("Unloading MLX model on shutdown...")
            self.mlx_processor.unload_model()
        
        self._log("Queue processor stopped")
    
    def start(self):
        """Start the queue processor in a background thread."""
        if self.running:
            self._log("Queue processor already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        self._log("Queue processor thread started")
    
    def stop(self, timeout: float = 10.0):
        """
        Stop the queue processor.
        
        Args:
            timeout: Seconds to wait for thread to stop
        """
        if not self.running:
            return
        
        self._log("Stopping queue processor...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                self._log("Warning: Queue processor thread did not stop gracefully")
            else:
                self._log("Queue processor stopped successfully")
    
    def get_stats(self) -> dict:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with stats
        """
        queue_stats = self.queue.get_stats()
        
        return {
            **self.stats,
            "queue_pending": queue_stats.get("pending", 0),
            "queue_processing": queue_stats.get("processing", 0),
            "queue_completed": queue_stats.get("completed", 0),
            "queue_failed": queue_stats.get("failed", 0),
            "mlx_model_loaded": self.mlx_processor.is_loaded,
        }
    
    def print_stats(self):
        """Print processing statistics."""
        stats = self.get_stats()
        
        print("\n" + "=" * 80)
        print("Queue Processor Statistics")
        print("=" * 80)
        print(f"Items processed:     {stats['items_processed']}")
        print(f"Items failed:        {stats['items_failed']}")
        print(f"Items skipped:       {stats['items_skipped']}")
        
        if stats['items_processed'] > 0:
            avg_time = stats['total_processing_time'] / stats['items_processed']
            print(f"Avg processing time: {avg_time:.2f}s")
        
        print(f"\nQueue pending:       {stats['queue_pending']}")
        print(f"Queue processing:    {stats['queue_processing']}")
        print(f"Queue completed:     {stats['queue_completed']}")
        print(f"Queue failed:        {stats['queue_failed']}")
        
        print(f"\nMLX model loaded:    {stats['mlx_model_loaded']}")
        print("=" * 80 + "\n")
    
    def run_maintenance(self):
        """Run periodic maintenance tasks."""
        # Reset stale processing items
        stale_count = self.queue.reset_stale_processing(timeout_minutes=30)
        if stale_count > 0:
            self._log(f"Maintenance: Reset {stale_count} stale items")
        
        # Cleanup old completed items
        deleted_count = self.queue.cleanup_old_items()
        if deleted_count > 0:
            self._log(f"Maintenance: Deleted {deleted_count} old items")




