#!/usr/bin/env python3
"""
OCR Pipeline with Menu Detection
Processes images through OCR, detects menus, and generates annotated output.
"""

import argparse
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import iter_image_paths, load_and_resize_image
from ocr_processing import init_ocr, run_ocr
from layout_analysis import filter_chrome, split_columns_dynamic, group_paragraphs
from menu_detection import detect_menus
from visualization import create_combined_output


def process_image(ocr_model, image_path, output_dir):
    """Process a single image through the complete pipeline."""
    print(f"\nProcessing: {image_path.name}")
    print("=" * 80)

    start_time = time.perf_counter()

    # Step 1: Load and resize
    image_bgr, image_rgb, original_rgb, original_size, resized_size, scale = (
        load_and_resize_image(image_path)
    )
    print(f"Original size: {original_size[0]}x{original_size[1]}")
    if scale < 1.0:
        print(f"Resized to: {resized_size[0]}x{resized_size[1]} (scale={scale:.3f})")

    load_time = time.perf_counter() - start_time
    print(f"Load time: {load_time:.2f}s")

    # Step 2: Run OCR
    ocr_start = time.perf_counter()
    boxes, texts, scores = run_ocr(ocr_model, image_bgr)
    ocr_time = time.perf_counter() - ocr_start
    print(f"OCR time: {ocr_time:.2f}s")
    print(f"Detected {len(boxes)} text regions")

    if not boxes:
        print("No text detected, skipping...")
        return

    # Step 3: Layout analysis - filter chrome and group paragraphs
    H, W = image_rgb.shape[:2]
    keep_indices, drop_indices = filter_chrome(
        image_rgb, boxes, texts, scores, left_ratio=0.12, top_ratio=0.06, min_conf=0.65
    )

    # Split into columns and group into paragraphs
    columns = split_columns_dynamic(
        list(range(len(boxes))), boxes, W, eps_ratio=0.06, min_points=4
    )
    groups = []
    for col in columns:
        groups.extend(group_paragraphs(col, boxes, gap_factor=1.5))

    print(f"Paragraph groups: {len(groups)}")
    print(f"Dropped (chrome): {len(drop_indices)}")

    # Step 4: Menu detection
    menu_start = time.perf_counter()
    menu_results = detect_menus(boxes, texts, (W, H))
    menu_time = time.perf_counter() - menu_start
    print(f"Menu detection time: {menu_time:.2f}s")

    print("\nMenu Detection Results:")
    for name, status, score in menu_results:
        status_str = status or "none"
        print(f"  {name:>6s}: {status_str:>6s} (score: {score:6.2f})")

    # Step 5: Visualize and save
    viz_start = time.perf_counter()
    output_image = create_combined_output(
        image_rgb, boxes, texts, scores, groups, drop_indices, menu_results, (W, H)
    )

    # Save output
    output_path = output_dir / f"{image_path.stem}_processed.png"
    output_image.save(output_path)
    viz_time = time.perf_counter() - viz_start
    print(f"Visualization time: {viz_time:.2f}s")
    print(f"Saved to: {output_path}")

    total_time = time.perf_counter() - start_time
    print(f"TOTAL TIME: {total_time:.2f}s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process images with OCR and menu detection"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).parent.parent / "test_images",
        help="Directory containing input images (default: ../test_images)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "out",
        help="Directory for output images (default: ./out)",
    )

    args = parser.parse_args()

    # Resolve paths
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    print("OCR Pipeline with Menu Detection")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Initialize OCR model
    print("\nInitializing OCR model...")
    ocr_model = init_ocr()
    print("Model ready.")

    # Find images
    image_paths = list(iter_image_paths([input_dir]))
    if not image_paths:
        print(f"ERROR: No images found in {input_dir}")
        return

    print(f"\nFound {len(image_paths)} images to process:")
    for idx, path in enumerate(image_paths, 1):
        print(f"  {idx:>2d}. {path.name}")

    # Process each image
    overall_start = time.perf_counter()
    for image_path in image_paths:
        try:
            process_image(ocr_model, image_path, output_dir)
        except Exception as e:
            print(f"ERROR processing {image_path.name}: {e}")
            import traceback

            traceback.print_exc()

    overall_time = time.perf_counter() - overall_start
    print("\n" + "=" * 80)
    print(f"Processed {len(image_paths)} images in {overall_time:.2f}s")
    print(f"Average: {overall_time/len(image_paths):.2f}s per image")
    print("=" * 80)


if __name__ == "__main__":
    main()
