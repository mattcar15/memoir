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


# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\n\n🛑 Shutting down gracefully...")
    running = False


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


def run():
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
