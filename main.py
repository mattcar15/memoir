#!/usr/bin/env python3
"""
Screenshot Memory System
Captures screenshots at regular intervals, processes them with Ollama via LangChain,
and logs the results with timestamps.
"""

import argparse
import base64
import io
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import schedule
from PIL import Image, ImageGrab
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

# Resolution tier configurations
RESOLUTION_TIERS = {
    "efficient": (800, 600),
    "balanced": (1024, 768),
    "detailed": (1920, 1080),
}

# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\n\n🛑 Shutting down gracefully...")
    running = False


def setup_directories():
    """Create necessary directories for logs and screenshots"""
    logs_dir = Path("logs")
    screenshots_dir = logs_dir / "screenshots"

    logs_dir.mkdir(exist_ok=True)
    screenshots_dir.mkdir(exist_ok=True)

    return logs_dir, screenshots_dir


def capture_screenshot(resolution_tier="detailed"):
    """
    Capture a screenshot and scale it based on the resolution tier.

    Args:
        resolution_tier: One of 'efficient', 'balanced', or 'detailed'

    Returns:
        Tuple of (PIL Image object (scaled), stats dict) or (None, None) if capture fails
    """
    try:
        start_time = time.time()

        # Capture the entire screen
        screenshot = ImageGrab.grab()
        capture_time = time.time() - start_time

        # Get target resolution
        target_width, target_height = RESOLUTION_TIERS.get(
            resolution_tier, RESOLUTION_TIERS["detailed"]
        )

        # Calculate scaling to maintain aspect ratio
        original_width, original_height = screenshot.size
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        scale_ratio = min(width_ratio, height_ratio)

        # Calculate new dimensions
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)

        # Resize image
        resize_start = time.time()
        scaled_screenshot = screenshot.resize(
            (new_width, new_height), resample=Image.LANCZOS
        )
        resize_time = time.time() - resize_start

        # Calculate image size in memory
        img_buffer = io.BytesIO()
        scaled_screenshot.save(img_buffer, format="PNG")
        image_size_bytes = len(img_buffer.getvalue())

        stats = {
            "original_resolution": f"{original_width}x{original_height}",
            "scaled_resolution": f"{new_width}x{new_height}",
            "capture_time_seconds": round(capture_time, 3),
            "resize_time_seconds": round(resize_time, 3),
            "image_size_bytes": image_size_bytes,
            "image_size_kb": round(image_size_bytes / 1024, 2),
        }

        return scaled_screenshot, stats

    except Exception as e:
        print(f"❌ Error capturing screenshot: {e}")
        return None, None


def image_to_base64(image):
    """
    Convert PIL Image to base64 string for Ollama.

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def process_with_ollama(image, model_name="gemma3:4b"):
    """
    Process the screenshot with Ollama via LangChain.

    Args:
        image: PIL Image object
        model_name: Name of the Ollama model to use

    Returns:
        Tuple of (Model's response text, stats dict) or (None, None) if processing fails
    """
    try:
        start_time = time.time()

        # Initialize Ollama chat model with keep_alive to keep model loaded
        llm = ChatOllama(
            model=model_name,
            temperature=0.3,
            keep_alive="1h",  # Keep model loaded for 1 hour
        )

        # Convert image to base64
        image_base64 = image_to_base64(image)
        image_base64_size = len(image_base64)

        # Create message with image and prompt
        prompt_text = "Summarize the user's current activity"
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{image_base64}",
                },
                {"type": "text", "text": prompt_text},
            ],
        )

        # Get response from model
        response = llm.invoke([message])
        processing_time = time.time() - start_time

        # Extract token information if available
        stats = {
            "ollama_processing_time_seconds": round(processing_time, 3),
            "prompt_text": prompt_text,
            "prompt_length_chars": len(prompt_text),
            "image_base64_size_bytes": image_base64_size,
            "image_base64_size_kb": round(image_base64_size / 1024, 2),
            "response_length_chars": len(response.content),
            "response_word_count": len(response.content.split()),
        }

        # Try to extract token usage from response metadata if available
        if hasattr(response, "response_metadata"):
            metadata = response.response_metadata
            if "total_duration" in metadata:
                stats["model_total_duration_ns"] = metadata["total_duration"]
            if "load_duration" in metadata:
                stats["model_load_duration_ns"] = metadata["load_duration"]
            if "prompt_eval_count" in metadata:
                stats["prompt_eval_token_count"] = metadata["prompt_eval_count"]
            if "eval_count" in metadata:
                stats["response_token_count"] = metadata["eval_count"]

        return response.content, stats

    except Exception as e:
        print(f"❌ Error processing with Ollama: {e}")
        return None, None


def save_log_entry(summary, resolution_tier, screenshot_path, logs_dir, stats=None):
    """
    Save a log entry as an individual JSON file.

    Args:
        summary: Model's output text
        resolution_tier: Resolution tier used
        screenshot_path: Path to saved screenshot or None
        logs_dir: Path to logs directory
        stats: Dictionary of performance statistics

    Returns:
        Path to the created log file
    """
    timestamp = datetime.now()

    # Create filename with timestamp
    filename = timestamp.strftime("%Y-%m-%d_%H-%M-%S.json")
    filepath = logs_dir / filename

    # Create log entry
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "summary": summary,
        "resolution_tier": resolution_tier,
        "screenshot_path": str(screenshot_path) if screenshot_path else None,
    }

    # Add stats if provided
    if stats:
        log_entry["stats"] = stats

    # Write to file
    try:
        with open(filepath, "w") as f:
            json.dump(log_entry, f, indent=2)
        return filepath
    except Exception as e:
        print(f"❌ Error saving log entry: {e}")
        return None


def save_screenshot(image, screenshots_dir):
    """
    Save screenshot image to disk.

    Args:
        image: PIL Image object
        screenshots_dir: Path to screenshots directory

    Returns:
        Path to saved screenshot or None
    """
    timestamp = datetime.now()
    filename = timestamp.strftime("%Y-%m-%d_%H-%M-%S.png")
    filepath = screenshots_dir / filename

    try:
        image.save(filepath, format="PNG")
        return filepath
    except Exception as e:
        print(f"❌ Error saving screenshot: {e}")
        return None


def capture_and_process(
    resolution_tier, save_images, logs_dir, screenshots_dir, model_name
):
    """
    Main function to capture, process, and log a screenshot.

    Args:
        resolution_tier: Resolution tier to use
        save_images: Whether to save screenshot images
        logs_dir: Path to logs directory
        screenshots_dir: Path to screenshots directory
        model_name: Ollama model name

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

    # Save log entry
    log_file = save_log_entry(
        summary, resolution_tier, screenshot_path, logs_dir, all_stats
    )
    if log_file:
        print(f"✅ Log saved: {log_file}")
        print(f"⏱️  Total time: {all_stats['total_processing_time_seconds']}s")

    return all_stats


def warmup_model(model_name="gemma3:4b"):
    """
    Warm up the Ollama model by running a tiny test inference.
    This loads the model into memory to avoid cold start delays.

    Args:
        model_name: Name of the Ollama model to warm up

    Returns:
        True if warmup successful, False otherwise
    """
    try:
        print(f"🔥 Warming up Ollama model ({model_name})...")
        start_time = time.time()

        # Initialize Ollama with keep_alive
        llm = ChatOllama(
            model=model_name,
            temperature=0.3,
            keep_alive="1h",
        )

        # Send a minimal prompt to load the model
        message = HumanMessage(content="Hi")
        response = llm.invoke([message])

        warmup_time = time.time() - start_time
        print(f"✅ Model warmed up in {round(warmup_time, 2)}s")
        print(f"💡 Tip: Run 'ollama ps' in terminal to verify model is loaded")
        return True

    except Exception as e:
        print(f"⚠️  Model warmup failed: {e}")
        print("Continuing anyway - first capture may be slower")
        return False


def compare_resolution_tiers(save_images, logs_dir, screenshots_dir, model_name):
    """
    Compare all three resolution tiers and display results.
    Only works in run-once mode.

    Args:
        save_images: Whether to save screenshot images
        logs_dir: Path to logs directory
        screenshots_dir: Path to screenshots directory
        model_name: Ollama model name
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
            tier, save_images, logs_dir, screenshots_dir, model_name
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


def main():
    """Main application entry point"""
    global running

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Screenshot Memory System - Capture and analyze screen activity with Ollama"
    )
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

    args = parser.parse_args()

    # Validate arguments
    if args.compare_tiers and not args.run_once:
        print("❌ Error: --compare-tiers requires --run-once flag")
        sys.exit(1)

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Create necessary directories
    logs_dir, screenshots_dir = setup_directories()

    print("=" * 60)
    print("📷 Screenshot Memory System")
    print("=" * 60)
    if not args.compare_tiers:
        print(
            f"Resolution tier: {args.resolution_tier} ({RESOLUTION_TIERS[args.resolution_tier][0]}x{RESOLUTION_TIERS[args.resolution_tier][1]})"
        )
    print(f"Save images: {'Yes' if args.save_images else 'No'}")
    print(f"Interval: {args.interval} minute(s)")
    print(f"Model: {args.model}")
    print(f"Logs directory: {logs_dir.absolute()}")
    print("=" * 60)

    # Warm up the model (unless disabled)
    if not args.no_warmup:
        print()
        warmup_model(args.model)
        print()

    if not args.run_once:
        print("Press Ctrl+C to stop\n")

    # Compare tiers mode
    if args.compare_tiers:
        compare_resolution_tiers(
            args.save_images, logs_dir, screenshots_dir, args.model
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
    )

    # Run immediately on start
    capture_and_process(
        args.resolution_tier, args.save_images, logs_dir, screenshots_dir, args.model
    )

    # Main loop
    while running:
        schedule.run_pending()
        time.sleep(1)

    print("👋 Goodbye!")


if __name__ == "__main__":
    main()
