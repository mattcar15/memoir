#!/usr/bin/env python3
"""
Batch image processing script for memoir logs.

Processes all images in a directory through the VLM pipeline with:
- Configurable throttling based on system resources (temperature, battery, etc.)
- Memory-efficient processing (loads one image at a time)
- Progress tracking and statistics

Usage:
    python batch_process_images.py                           # Process logs/staging with throttling
    python batch_process_images.py --input-dir /path/to/dir  # Custom directory
    python batch_process_images.py --no-throttle             # Disable throttling
    python batch_process_images.py --verbose                 # Detailed output
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Iterator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image
from tqdm import tqdm
from memoir.processing.pipeline.main import ImagePipeline
from memoir.core.system_monitor import SystemMonitor, get_global_monitor
from memoir.config import CAPTURE_STAGING_DIR


def get_image_paths(directory: Path) -> List[Path]:
    """
    Get all image files from directory, sorted by name (which typically includes timestamp).

    Args:
        directory: Directory to scan for images

    Returns:
        List of image file paths, sorted
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    images = []

    for ext in image_extensions:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(images, key=lambda p: p.name)


def load_image_lazy(path: Path) -> Optional[Image.Image]:
    """
    Load an image from disk. Returns None if loading fails.

    This function loads images one at a time to be memory efficient.

    Args:
        path: Path to image file

    Returns:
        PIL Image or None if loading failed
    """
    try:
        image = Image.open(path)
        # Force load the image data into memory (so we can close the file)
        image.load()
        return image
    except Exception as e:
        tqdm.write(f"⚠️  Failed to load {path.name}: {e}")
        return None


def wait_for_system_ready(
    monitor: SystemMonitor,
    verbose: bool = False,
    check_interval: float = 2.0,
    max_wait: float = 300.0,
) -> str:
    """
    Wait until system is ready for processing based on monitor recommendations.

    Args:
        monitor: SystemMonitor instance
        verbose: Print status updates
        check_interval: Seconds between checks
        max_wait: Maximum seconds to wait before proceeding anyway

    Returns:
        Final recommendation ("RUN", "SLOW", or "PAUSE")
    """
    start_time = time.time()
    last_status = None

    while True:
        elapsed = time.time() - start_time

        # Get fresh recommendation
        state = monitor.get_system_state(use_cache=False)
        recommendation = state["recommendation"] if state else "SLOW"

        # Only print if status changed or we're being verbose
        if recommendation != last_status:
            if recommendation == "RUN":
                if verbose:
                    tqdm.write(
                        f"✅ System ready: {state.get('gpu_temp', '?')}°C GPU, {state.get('battery_pct', '?')}% battery"
                    )
                return recommendation
            elif recommendation == "SLOW":
                if verbose:
                    tqdm.write(
                        f"⚡ System warm: {state.get('gpu_temp', '?')}°C GPU - proceeding at reduced pace"
                    )
                return recommendation
            else:  # PAUSE
                tqdm.write(
                    f"⏸️  Waiting for system cooldown: {state.get('gpu_temp', '?')}°C GPU, {state.get('battery_pct', '?')}% battery"
                )

        last_status = recommendation

        # Check timeout
        if elapsed > max_wait:
            tqdm.write(f"⏱️  Timeout after {elapsed:.0f}s, proceeding anyway")
            return recommendation

        # If paused, wait before checking again
        if recommendation == "PAUSE":
            time.sleep(check_interval)
        else:
            return recommendation


def get_throttle_delay(recommendation: str) -> float:
    """
    Get delay to add between image processing based on system recommendation.

    Args:
        recommendation: "RUN", "SLOW", or "PAUSE"

    Returns:
        Delay in seconds
    """
    if recommendation == "RUN":
        return 0.0
    elif recommendation == "SLOW":
        return 1.0  # Add 1 second between images when warm
    else:  # PAUSE
        return 5.0  # Add 5 seconds when system wants to pause


def process_batch(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    throttle: bool = True,
    verbose: bool = False,
    save_images: bool = False,
    skip_duplicates: bool = True,
    cooldown_interval: int = 10,
) -> Dict[str, Any]:
    """
    Process all images in a directory through the VLM pipeline.

    Args:
        input_dir: Directory containing images to process
        output_dir: Directory to save processed images (optional)
        throttle: Enable system-based throttling
        verbose: Print detailed output
        save_images: Save processed images to output_dir
        skip_duplicates: Use pHash to skip duplicate images
        cooldown_interval: Check system state every N images when throttling

    Returns:
        Dictionary with processing results and statistics
    """
    print("=" * 70)
    print("📷 Memoir Batch Image Processor")
    print("=" * 70)
    print()

    # Validate input directory
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return {"error": "Input directory not found"}

    if not input_dir.is_dir():
        print(f"❌ Path is not a directory: {input_dir}")
        return {"error": "Path is not a directory"}

    # Get image files
    image_paths = get_image_paths(input_dir)
    total_images = len(image_paths)

    if total_images == 0:
        print(f"❌ No images found in {input_dir}")
        return {"error": "No images found"}

    print(f"📁 Input directory: {input_dir}")
    print(f"🖼️  Found {total_images} images to process")
    print(f"⚙️  Throttling: {'enabled' if throttle else 'disabled'}")
    print(f"📝 Verbose: {'enabled' if verbose else 'disabled'}")
    print()

    # Initialize system monitor if throttling
    monitor = None
    if throttle:
        print("🔧 Initializing system monitor...")
        monitor = get_global_monitor(verbose=verbose)

        # Check initial system state
        state = monitor.get_system_state(use_cache=False)
        if state:
            print(
                f"   Battery: {state['battery_pct']}% {'(charging)' if state['charging'] else '(on battery)'}"
            )
            if state.get("gpu_temp"):
                print(f"   GPU Temp: {state['gpu_temp']:.1f}°C")
            if state.get("cpu_temp"):
                print(f"   CPU Temp: {state['cpu_temp']:.1f}°C")
        print()

    # Initialize pipeline
    print("🔧 Initializing VLM pipeline...")
    pipeline = ImagePipeline(
        similarity_threshold=0.7,
        memory_window_minutes=5,
        phash_threshold=10 if skip_duplicates else 0,
        verbose=verbose,
    )
    print("✅ Pipeline ready")
    print()

    # Processing stats
    results = []
    start_time = time.time()
    processed_count = 0
    skipped_count = 0
    error_count = 0
    throttle_wait_time = 0.0

    print("=" * 70)
    print("🚀 Starting batch processing...")
    print("=" * 70)
    print()

    # Create progress bar
    pbar = tqdm(
        enumerate(image_paths, 1),
        total=total_images,
        desc="Processing",
        unit="img",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for idx, image_path in pbar:
        # Update progress bar description with current file
        pbar.set_postfix_str(
            image_path.name[:30] + "..."
            if len(image_path.name) > 30
            else image_path.name
        )

        # Throttling check
        if throttle and monitor:
            # Check system state periodically
            if (idx - 1) % cooldown_interval == 0:
                wait_start = time.time()
                recommendation = wait_for_system_ready(monitor, verbose=verbose)
                throttle_wait_time += time.time() - wait_start

                # Add delay based on system state
                delay = get_throttle_delay(recommendation)
                if delay > 0:
                    if verbose:
                        tqdm.write(f"   Adding {delay:.1f}s throttle delay")
                    time.sleep(delay)
                    throttle_wait_time += delay

        if verbose:
            tqdm.write(f"\n[{idx}/{total_images}] Processing: {image_path.name}")

        # Load image (memory efficient - one at a time)
        image = load_image_lazy(image_path)
        if image is None:
            error_count += 1
            continue

        try:
            # Process through pipeline
            result = pipeline.process_image(
                image,
                output_dir=output_dir,
                save_images=save_images,
            )

            if result:
                result["source_file"] = str(image_path)
                results.append(result)
                processed_count += 1
            else:
                skipped_count += 1
                if verbose:
                    tqdm.write(f"   ⏭️  Skipped (duplicate or processing failed)")

        except Exception as e:
            tqdm.write(f"   ❌ Error processing {image_path.name}: {e}")
            error_count += 1
            if verbose:
                import traceback

                traceback.print_exc()
        finally:
            # Explicitly close image to free memory
            image.close()
            del image

    pbar.close()

    # Calculate totals
    total_time = time.time() - start_time
    processing_time = total_time - throttle_wait_time

    # Print summary
    print()
    print("=" * 70)
    print("📊 BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print()
    print(f"📁 Input: {input_dir}")
    print(f"🖼️  Total images: {total_images}")
    print(f"✅ Processed: {processed_count}")
    print(f"⏭️  Skipped (duplicates): {skipped_count}")
    print(f"❌ Errors: {error_count}")
    print()
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"⏱️  Processing time: {processing_time:.1f}s")
    if throttle:
        print(f"⏱️  Throttle wait time: {throttle_wait_time:.1f}s")
    if processed_count > 0:
        print(f"⏱️  Avg per image: {processing_time / processed_count:.2f}s")
    print()

    # Print pipeline statistics
    pipeline.print_statistics()

    # Compile final results
    summary = {
        "input_dir": str(input_dir),
        "total_images": total_images,
        "processed": processed_count,
        "skipped": skipped_count,
        "errors": error_count,
        "total_time_seconds": round(total_time, 2),
        "processing_time_seconds": round(processing_time, 2),
        "throttle_wait_seconds": round(throttle_wait_time, 2),
        "throttle_enabled": throttle,
        "pipeline_stats": pipeline.get_statistics(),
        "results": results,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Batch process images through the memoir VLM pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process default logs/staging directory with throttling
    python batch_process_images.py

    # Process custom directory
    python batch_process_images.py --input-dir /path/to/images

    # Disable throttling for maximum speed
    python batch_process_images.py --no-throttle

    # Verbose output with saved results
    python batch_process_images.py --verbose --output results.json
        """,
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=CAPTURE_STAGING_DIR,
        help=f"Directory containing images to process (default: {CAPTURE_STAGING_DIR})",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory to save processed images (optional)",
    )

    parser.add_argument(
        "--output",
        "-O",
        type=Path,
        default=None,
        help="Save results to JSON file",
    )

    parser.add_argument(
        "--no-throttle",
        action="store_true",
        help="Disable system-based throttling (process at full speed)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed processing information",
    )

    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save processed images to output directory",
    )

    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable duplicate detection (process all images)",
    )

    parser.add_argument(
        "--cooldown-interval",
        type=int,
        default=10,
        help="Check system state every N images (default: 10)",
    )

    args = parser.parse_args()

    # Run batch processing
    results = process_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        throttle=not args.no_throttle,
        verbose=args.verbose,
        save_images=args.save_images,
        skip_duplicates=not args.no_dedup,
        cooldown_interval=args.cooldown_interval,
    )

    # Save results to JSON if requested
    if args.output and "error" not in results:
        print(f"\n💾 Saving results to {args.output}...")
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✅ Results saved")

    # Return appropriate exit code
    if "error" in results:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
