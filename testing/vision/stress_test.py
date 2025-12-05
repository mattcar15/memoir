#!/usr/bin/env python3
"""
Stress test script for OCR with sidebar filtering.
Processes all images in the test_images folder repeatedly.
"""

import argparse
import time
from pathlib import Path

from ocr_with_sidebar_filter import (
    ocr_with_sidebar_filter,
    annotate_with_sidebars,
)


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def find_test_images(test_dir: Path) -> list[Path]:
    """Find all image files in the test directory."""
    images = []
    for ext in SUPPORTED_EXTENSIONS:
        images.extend(test_dir.glob(f"*{ext}"))
    # Filter out generated files
    images = [
        img
        for img in images
        if not img.stem.endswith("_cropped")
        and not img.stem.endswith("_sidebar_filtered")
        and not img.stem.endswith("_annotated")
    ]
    return sorted(images)


def run_stress_test(
    test_dir: Path,
    iterations: int = 1,
    save_output: bool = False,
    verbose: bool = True,
):
    """
    Run OCR with sidebar filtering on all test images.

    Args:
        test_dir: Directory containing test images
        iterations: Number of times to process all images
        save_output: Whether to save annotated output images
        verbose: Print detailed output
    """
    images = find_test_images(test_dir)

    if not images:
        print(f"No images found in {test_dir}")
        return

    print("=" * 70)
    print("OCR Sidebar Filter Stress Test")
    print("=" * 70)
    print(f"Test directory: {test_dir}")
    print(f"Images found: {len(images)}")
    print(f"Iterations: {iterations}")
    print(f"Total runs: {len(images) * iterations}")
    print("=" * 70)

    for img in images:
        print(f"  - {img.name}")
    print()

    total_time = 0
    total_runs = 0
    results_summary = []

    for iteration in range(1, iterations + 1):
        if iterations > 1:
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{iterations}")
            print("=" * 70)

        for img_path in images:
            print(f"\nProcessing: {img_path.name}")
            print("-" * 50)

            start_time = time.perf_counter()

            try:
                result = ocr_with_sidebar_filter(str(img_path), return_all=True)
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
                total_runs += 1

                # Summary
                width, height = result["image_size"]
                crop = result["crop_region"]
                content_count = len(result["content_detections"])
                sidebar_count = len(result.get("sidebar_detections", []))

                menu_status = {}
                for region in result["menu_results"]:
                    name = region["name"]
                    status = region.get("status") or "none"
                    score = region.get("score", 0.0)
                    menu_status[name] = (status, score)

                if verbose:
                    print(f"  Size: {width}x{height}")
                    rw, rh = result.get("resized_size", (width, height))
                    scale = result.get("scale", 1.0)
                    if scale < 1.0:
                        print(f"  Resized: {rw}x{rh} (scale={scale:.3f})")
                    print(f"  Crop: ({crop[0]}, {crop[1]}) → ({crop[2]}, {crop[3]})")
                    print(f"  Content: {content_count} | Sidebar: {sidebar_count}")
                    for name, (status, score) in menu_status.items():
                        print(f"    {name}: {status} ({score:.2f})")

                print(f"  Time: {elapsed:.2f}s")

                results_summary.append(
                    {
                        "image": img_path.name,
                        "iteration": iteration,
                        "time": elapsed,
                        "content": content_count,
                        "sidebar": sidebar_count,
                        "menus": menu_status,
                    }
                )

                if save_output:
                    paths = annotate_with_sidebars(str(img_path), save_cropped=True)
                    print(f"  Saved: {Path(paths['annotated']).name}")

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                print(f"  ERROR: {e}")
                import traceback

                traceback.print_exc()
                results_summary.append(
                    {
                        "image": img_path.name,
                        "iteration": iteration,
                        "time": elapsed,
                        "error": str(e),
                    }
                )

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total runs: {total_runs}")
    print(f"Total time: {total_time:.2f}s")
    if total_runs > 0:
        avg_time = total_time / total_runs
        print(f"Average time per image: {avg_time:.2f}s")
        print(f"Images per second: {1/avg_time:.2f}")

    # Show any errors
    errors = [r for r in results_summary if "error" in r]
    if errors:
        print(f"\nErrors: {len(errors)}")
        for err in errors:
            print(f"  - {err['image']}: {err['error']}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Stress test OCR with sidebar filtering on all test images"
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "test_images",
        help="Directory containing test images",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations (default: 1)",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save annotated output images"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Less verbose output"
    )

    args = parser.parse_args()

    run_stress_test(
        test_dir=args.test_dir.resolve(),
        iterations=args.iterations,
        save_output=args.save,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
