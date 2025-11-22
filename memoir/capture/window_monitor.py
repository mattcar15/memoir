"""
Window monitoring for intelligent frame capture.

This module tracks window/tab changes and captures screenshots when
the topmost application or window title changes. It uses perceptual hashing
to deduplicate similar frames per-tab.
"""

import time
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import imagehash
from PIL import Image

from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGNullWindowID,
    kCGWindowListOptionOnScreenOnly,
    kCGWindowListExcludeDesktopElements,
    kCGWindowBounds,
    kCGWindowLayer,
    kCGWindowOwnerName,
    kCGWindowOwnerPID,
    kCGWindowName,
    kCGWindowNumber,
    kCGWindowAlpha,
    kCGWindowSharingState,
    kCGWindowMemoryUsage,
)

from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
import objc


class WindowMonitor:
    """
    Monitor window changes and capture frames when app or title changes.
    
    Uses perceptual hashing (pHash) to deduplicate similar frames per-tab.
    """
    
    def __init__(
        self,
        capture_dir: Path,
        on_capture_callback=None,
        phash_threshold: int = 10,
        poll_interval: float = 0.5,
        debug: bool = False,
    ):
        """
        Initialize the window monitor.
        
        Args:
            capture_dir: Directory to save captured frames
            on_capture_callback: Callback function(frame_path, metadata) called when a frame is captured
            phash_threshold: pHash difference threshold for duplicate detection (0-64)
            poll_interval: Seconds between window snapshot checks
            debug: If True, print debug messages
        """
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        
        self.on_capture_callback = on_capture_callback
        self.phash_threshold = phash_threshold
        self.poll_interval = poll_interval
        self.debug = debug
        
        # Track last phash per tab (key: "app_name:window_title")
        self.tab_hashes: Dict[str, Tuple[str, datetime]] = {}  # tab_key -> (phash_str, timestamp)
        
        # Previous snapshot state
        self.prev_topmost: Optional[Dict] = None
        
        # Control flags
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Request accessibility permissions
        AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
        
        # Constants for window listing
        self.ONSCREEN_OPTS = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
    
    def _log(self, message: str):
        """Log debug message if debug is enabled."""
        if self.debug:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[WindowMonitor {timestamp}] {message}")
    
    def _take_snapshot(self) -> List[Dict]:
        """
        Take a snapshot of all visible windows.
        
        Returns:
            List of window info dictionaries
        """
        raw = CGWindowListCopyWindowInfo(self.ONSCREEN_OPTS, kCGNullWindowID) or []
        snap = []
        for order, w in enumerate(raw):
            layer = w.get(kCGWindowLayer, 0)
            # Only include layer 0 windows (normal windows)
            if layer != 0:
                continue
                
            b = w.get(kCGWindowBounds, {})
            snap.append({
                "order": order,  # 0 = topmost on screen
                "id": w.get(kCGWindowNumber),
                "pid": w.get(kCGWindowOwnerPID),
                "app": w.get(kCGWindowOwnerName, "") or "",
                "title": w.get(kCGWindowName, "") or "",
                "x": int(b.get("X", 0)),
                "y": int(b.get("Y", 0)),
                "w": int(b.get("Width", 0)),
                "h": int(b.get("Height", 0)),
                "layer": layer,
            })
        return snap
    
    def _get_topmost_window(self, snapshot: List[Dict]) -> Optional[Dict]:
        """
        Get the topmost window from a snapshot.
        
        Args:
            snapshot: List of window info dicts
            
        Returns:
            Topmost window dict or None
        """
        top_level = [w for w in snapshot if w["layer"] == 0]
        if not top_level:
            return None
        return min(top_level, key=lambda r: r["order"])
    
    def _capture_screenshot(self, window_info: Dict) -> Optional[Path]:
        """
        Capture a screenshot of the specified window.
        
        Args:
            window_info: Window information dictionary
            
        Returns:
            Path to saved screenshot or None if failed
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        app_name = window_info["app"].replace("/", "_").replace(" ", "_")
        title = window_info["title"].replace("/", "_").replace(" ", "_")[:50]
        filename = f"{timestamp}_{app_name}_{title}.png"
        
        filepath = self.capture_dir / filename
        window_id = window_info["id"]
        
        try:
            # Use screencapture to capture the window
            subprocess.run(
                ["screencapture", "-o", "-l", str(window_id), "-x", str(filepath)],
                check=True,
                capture_output=True,
                timeout=5,
            )
            return filepath
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self._log(f"Error capturing screenshot: {e}")
            return None
    
    def _compute_phash(self, image_path: Path) -> Optional[str]:
        """
        Compute perceptual hash of an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            pHash string or None if failed
        """
        try:
            img = Image.open(image_path)
            return str(imagehash.phash(img))
        except Exception as e:
            self._log(f"Error computing pHash: {e}")
            return None
    
    def _is_duplicate(self, tab_key: str, phash_str: str) -> bool:
        """
        Check if image is a duplicate of the last frame from this tab.
        
        Args:
            tab_key: Unique key for the tab (app:title)
            phash_str: pHash string of the current frame
            
        Returns:
            True if duplicate, False otherwise
        """
        if tab_key not in self.tab_hashes:
            return False
        
        prev_hash_str, _ = self.tab_hashes[tab_key]
        prev_hash = imagehash.hex_to_hash(prev_hash_str)
        curr_hash = imagehash.hex_to_hash(phash_str)
        
        diff = curr_hash - prev_hash
        return diff <= self.phash_threshold
    
    def _should_capture(self, curr_topmost: Dict) -> bool:
        """
        Determine if we should capture a screenshot.
        
        Args:
            curr_topmost: Current topmost window info
            
        Returns:
            True if should capture, False otherwise
        """
        if not self.prev_topmost:
            return True
        
        # Check if app or window changed
        if (
            self.prev_topmost["app"] != curr_topmost["app"]
            or self.prev_topmost["id"] != curr_topmost["id"]
            or self.prev_topmost["title"] != curr_topmost["title"]
        ):
            return True
        
        return False
    
    def _process_capture(self, window_info: Dict, frame_path: Path):
        """
        Process a captured frame with deduplication and callback.
        
        Args:
            window_info: Window information dictionary
            frame_path: Path to captured frame
        """
        # Compute pHash
        phash_str = self._compute_phash(frame_path)
        if not phash_str:
            self._log("Failed to compute pHash, skipping frame")
            frame_path.unlink()  # Delete the frame
            return
        
        # Create tab key
        tab_key = f"{window_info['app']}:{window_info['title']}"
        
        # Check for duplicate
        if self._is_duplicate(tab_key, phash_str):
            self._log(f"Duplicate frame detected for tab: {tab_key}")
            frame_path.unlink()  # Delete the duplicate
            return
        
        # Update hash tracking
        self.tab_hashes[tab_key] = (phash_str, datetime.now())
        
        # Prepare metadata
        metadata = {
            "app": window_info["app"],
            "title": window_info["title"],
            "window_id": window_info["id"],
            "pid": window_info["pid"],
            "timestamp": datetime.now().isoformat(),
            "phash": phash_str,
            "tab_key": tab_key,
        }
        
        self._log(f"Captured frame: {frame_path.name}")
        
        # Call callback if provided
        if self.on_capture_callback:
            try:
                self.on_capture_callback(frame_path, metadata)
            except Exception as e:
                self._log(f"Error in capture callback: {e}")
    
    def _monitor_loop(self):
        """Main monitoring loop (runs in background thread)."""
        self._log("Window monitor started")
        
        # Initialize with current snapshot
        snapshot = self._take_snapshot()
        self.prev_topmost = self._get_topmost_window(snapshot)
        
        while self.running:
            try:
                # Take new snapshot
                snapshot = self._take_snapshot()
                curr_topmost = self._get_topmost_window(snapshot)
                
                if curr_topmost and self._should_capture(curr_topmost):
                    self._log(
                        f"Change detected: {curr_topmost['app']} - {curr_topmost['title'][:50]}"
                    )
                    
                    # Capture screenshot
                    frame_path = self._capture_screenshot(curr_topmost)
                    if frame_path:
                        self._process_capture(curr_topmost, frame_path)
                
                # Update previous state
                self.prev_topmost = curr_topmost
                
                # Sleep before next check
                time.sleep(self.poll_interval)
                
            except Exception as e:
                self._log(f"Error in monitor loop: {e}")
                time.sleep(self.poll_interval)
        
        self._log("Window monitor stopped")
    
    def start(self):
        """Start the window monitor in a background thread."""
        if self.running:
            self._log("Window monitor already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        self._log("Window monitor thread started")
    
    def stop(self, timeout: float = 5.0):
        """
        Stop the window monitor.
        
        Args:
            timeout: Seconds to wait for thread to stop
        """
        if not self.running:
            return
        
        self._log("Stopping window monitor...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                self._log("Warning: Window monitor thread did not stop gracefully")
            else:
                self._log("Window monitor stopped successfully")
    
    def cleanup_old_hashes(self, max_age_hours: int = 24):
        """
        Clean up old tab hashes to prevent memory growth.
        
        Args:
            max_age_hours: Remove hashes older than this many hours
        """
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        keys_to_remove = [
            key for key, (_, timestamp) in self.tab_hashes.items()
            if timestamp < cutoff
        ]
        
        for key in keys_to_remove:
            del self.tab_hashes[key]
        
        if keys_to_remove:
            self._log(f"Cleaned up {len(keys_to_remove)} old tab hashes")




