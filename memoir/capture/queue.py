"""
SQLite-based processing queue for captured frames.

This module manages a persistent queue of images to be processed,
allowing the system to capture frames quickly and process them
asynchronously based on system resources.
"""

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any


class ProcessingQueue:
    """SQLite-based persistent queue for image processing."""
    
    def __init__(self, db_path: Path, auto_cleanup_days: int = 7):
        """
        Initialize the processing queue.
        
        Args:
            db_path: Path to SQLite database file
            auto_cleanup_days: Auto-delete completed/failed items older than this many days
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.auto_cleanup_days = auto_cleanup_days
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queue_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    processed_at TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0
                )
            """)
            
            # Create indices for better performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status 
                ON queue_items(status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON queue_items(created_at)
            """)
            
            conn.commit()
    
    def enqueue(self, image_path: Path, metadata: Dict[str, Any]) -> int:
        """
        Add an item to the processing queue.
        
        Args:
            image_path: Path to the captured image
            metadata: Metadata dictionary (will be JSON serialized)
            
        Returns:
            Queue item ID
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO queue_items (image_path, metadata_json, status, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        str(image_path),
                        json.dumps(metadata),
                        "pending",
                        datetime.now().isoformat(),
                    )
                )
                conn.commit()
                return cursor.lastrowid
    
    def dequeue(self) -> Optional[Dict[str, Any]]:
        """
        Get the next pending item from the queue.
        
        Automatically marks the item as 'processing' to prevent
        duplicate processing.
        
        Returns:
            Dictionary with item details or None if queue is empty
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Get oldest pending item
                cursor = conn.execute(
                    """
                    SELECT id, image_path, metadata_json, created_at, retry_count
                    FROM queue_items
                    WHERE status = 'pending'
                    ORDER BY created_at ASC
                    LIMIT 1
                    """
                )
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                item_id = row["id"]
                
                # Mark as processing
                conn.execute(
                    """
                    UPDATE queue_items
                    SET status = 'processing'
                    WHERE id = ?
                    """,
                    (item_id,)
                )
                conn.commit()
                
                return {
                    "id": item_id,
                    "image_path": Path(row["image_path"]),
                    "metadata": json.loads(row["metadata_json"]),
                    "created_at": row["created_at"],
                    "retry_count": row["retry_count"],
                }
    
    def mark_completed(self, item_id: int):
        """
        Mark an item as successfully processed.
        
        Args:
            item_id: Queue item ID
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE queue_items
                    SET status = 'completed',
                        processed_at = ?
                    WHERE id = ?
                    """,
                    (datetime.now().isoformat(), item_id)
                )
                conn.commit()
    
    def mark_failed(self, item_id: int, error_message: str, retry: bool = False):
        """
        Mark an item as failed.
        
        Args:
            item_id: Queue item ID
            error_message: Error description
            retry: If True, mark as pending for retry; otherwise mark as failed
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                if retry:
                    conn.execute(
                        """
                        UPDATE queue_items
                        SET status = 'pending',
                            error_message = ?,
                            retry_count = retry_count + 1
                        WHERE id = ?
                        """,
                        (error_message, item_id)
                    )
                else:
                    conn.execute(
                        """
                        UPDATE queue_items
                        SET status = 'failed',
                            processed_at = ?,
                            error_message = ?
                        WHERE id = ?
                        """,
                        (datetime.now().isoformat(), error_message, item_id)
                    )
                conn.commit()
    
    def get_pending_count(self) -> int:
        """
        Get the number of pending items in the queue.
        
        Returns:
            Count of pending items
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM queue_items WHERE status = 'pending'"
            )
            return cursor.fetchone()[0]
    
    def get_processing_count(self) -> int:
        """
        Get the number of items currently being processed.
        
        Returns:
            Count of processing items
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM queue_items WHERE status = 'processing'"
            )
            return cursor.fetchone()[0]
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get queue statistics.
        
        Returns:
            Dictionary with counts by status
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT status, COUNT(*) as count
                FROM queue_items
                GROUP BY status
                """
            )
            
            stats = {row["status"]: row["count"] for row in cursor.fetchall()}
            
            # Ensure all statuses are present
            for status in ["pending", "processing", "completed", "failed"]:
                if status not in stats:
                    stats[status] = 0
            
            return stats
    
    def cleanup_old_items(self, days: Optional[int] = None):
        """
        Remove old completed and failed items from the queue.
        
        Args:
            days: Remove items older than this many days (default: use auto_cleanup_days)
        """
        if days is None:
            days = self.auto_cleanup_days
        
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Delete old completed items
                cursor = conn.execute(
                    """
                    DELETE FROM queue_items
                    WHERE status IN ('completed', 'failed')
                    AND processed_at < ?
                    """,
                    (cutoff_str,)
                )
                deleted_count = cursor.rowcount
                conn.commit()
                
                return deleted_count
    
    def reset_stale_processing(self, timeout_minutes: int = 30):
        """
        Reset items stuck in 'processing' state back to 'pending'.
        
        This handles cases where processing was interrupted.
        
        Args:
            timeout_minutes: Reset items processing for longer than this
            
        Returns:
            Number of items reset
        """
        cutoff = datetime.now() - timedelta(minutes=timeout_minutes)
        cutoff_str = cutoff.isoformat()
        
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # First, find items to reset
                cursor = conn.execute(
                    """
                    SELECT id FROM queue_items
                    WHERE status = 'processing'
                    AND created_at < ?
                    """,
                    (cutoff_str,)
                )
                
                stale_ids = [row[0] for row in cursor.fetchall()]
                
                if stale_ids:
                    # Reset to pending
                    placeholders = ",".join("?" * len(stale_ids))
                    conn.execute(
                        f"""
                        UPDATE queue_items
                        SET status = 'pending',
                            error_message = 'Reset from stale processing state'
                        WHERE id IN ({placeholders})
                        """,
                        stale_ids
                    )
                    conn.commit()
                
                return len(stale_ids)
    
    def get_all_pending(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all pending items (for debugging/monitoring).
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of item dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = """
                SELECT id, image_path, metadata_json, created_at, retry_count
                FROM queue_items
                WHERE status = 'pending'
                ORDER BY created_at ASC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor = conn.execute(query)
            
            return [
                {
                    "id": row["id"],
                    "image_path": Path(row["image_path"]),
                    "metadata": json.loads(row["metadata_json"]),
                    "created_at": row["created_at"],
                    "retry_count": row["retry_count"],
                }
                for row in cursor.fetchall()
            ]
    
    def clear_all(self):
        """Clear all items from the queue (for testing/reset)."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM queue_items")
                conn.commit()




