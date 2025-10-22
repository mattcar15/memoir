#!/usr/bin/env python3
"""
OCR Pipeline with Menu Detection
Processes images through OCR, detects menus, and generates annotated output.
"""

import argparse
import json
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image

from utils import iter_image_paths, load_and_resize_image
from ocr_processing import init_ocr, run_ocr
from layout_analysis import filter_chrome, split_columns_dynamic, group_paragraphs
from menu_detection import detect_menus
from visualization import (
    create_combined_output,
    draw_divider_edges,
    draw_menu_stage,
    draw_ocr_boxes,
)


def process_image(get_ocr_model, image_path, output_dir, reuse_ocr=False):
    """Process a single image through the complete pipeline."""
    print(f"\nProcessing: {image_path.name}")
    print("=" * 80)

    start_time = time.perf_counter()

    image_output_dir = output_dir / image_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)
    ocr_json_path = image_output_dir / "ocr.json"

    # Step 1: Load and resize
    image_bgr, image_rgb, original_rgb, original_size, resized_size, scale = (
        load_and_resize_image(image_path)
    )
    print(f"Original size: {original_size[0]}x{original_size[1]}")
    if scale < 1.0:
        print(f"Resized to: {resized_size[0]}x{resized_size[1]} (scale={scale:.3f})")

    load_time = time.perf_counter() - start_time
    print(f"Load time: {load_time:.2f}s")

    # Step 2: Run OCR (or reuse cached results)
    boxes, texts, scores = [], [], []
    loaded_from_cache = False
    if reuse_ocr:
        reuse_start = time.perf_counter()
        try:
            with ocr_json_path.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            boxes = cached.get("boxes") or []
            texts = cached.get("texts") or []
            scores = cached.get("scores") or []
            if len(texts) < len(boxes):
                texts.extend([""] * (len(boxes) - len(texts)))
            if len(scores) < len(boxes):
                scores.extend([None] * (len(boxes) - len(scores)))
            loaded_from_cache = True
            reuse_time = time.perf_counter() - reuse_start
            print(
                f"OCR reuse enabled; loaded cached results from {ocr_json_path.name} "
                f"in {reuse_time:.2f}s"
            )
        except FileNotFoundError:
            print(
                f"OCR reuse enabled but {ocr_json_path.name} was not found; running OCR..."
            )
        except json.JSONDecodeError as exc:
            print(
                f"Cached OCR data in {ocr_json_path.name} is invalid ({exc}); running OCR..."
            )
        except Exception as exc:
            print(
                f"Failed to load cached OCR data ({exc}); running OCR with fresh results..."
            )

    if not loaded_from_cache:
        ocr_model = get_ocr_model()
        ocr_start = time.perf_counter()
        boxes, texts, scores = run_ocr(ocr_model, image_bgr)
        ocr_time = time.perf_counter() - ocr_start
        print(f"OCR time: {ocr_time:.2f}s")
        try:
            with ocr_json_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {"boxes": boxes, "texts": texts, "scores": scores}, f, indent=2
                )
            if reuse_ocr:
                print(f"Saved OCR results for reuse to: {ocr_json_path}")
        except Exception as exc:
            print(f"WARNING: Failed to save OCR data for reuse ({exc})")
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
    menu_results = detect_menus(image_rgb, boxes, texts)
    menu_time = time.perf_counter() - menu_start
    print(f"Menu detection time: {menu_time:.2f}s")

    print("\nMenu Detection Results:")
    for region in menu_results:
        name = region["name"]
        final_status = region.get("status") or "none"
        final_score = region.get("score", 0.0)
        initial_status = region.get("initial_status") or "none"
        initial_score = region.get("initial_score", 0.0)
        print(
            f"  {name:>6s}: final={final_status:>6s} ({final_score:6.2f}) | initial={initial_status:>6s} ({initial_score:6.2f})"
        )
        divider = region.get("divider")
        if divider and region.get("status"):
            print(
                f"           divider at ({divider[0]}, {divider[1]}) → ({divider[2]}, {divider[3]})"
            )
        if region.get("notes"):
            print(f"           note: {region['notes']}")

    # Step 5: Visualize and save
    viz_start = time.perf_counter()
    output_image = create_combined_output(
        image_rgb, boxes, texts, scores, groups, drop_indices, menu_results
    )

    # Save output steps
    step_images = [
        ("step0_original.png", Image.fromarray(image_rgb)),
        ("step1_ocr_boxes.png", draw_ocr_boxes(image_rgb, boxes, texts, scores)),
        ("step2_initial_menus.png", draw_menu_stage(image_rgb, menu_results, stage="initial")),
        ("step3_refined_menus.png", draw_menu_stage(image_rgb, menu_results, stage="final")),
        ("step4_divider_edges.png", draw_divider_edges(image_rgb, boxes, menu_results)),
        ("step5_combined.png", output_image),
    ]

    for filename, pil_image in step_images:
        save_path = image_output_dir / filename
        pil_image.save(save_path)

    viz_time = time.perf_counter() - viz_start
    final_path = image_output_dir / "step5_combined.png"
    print(f"Visualization time: {viz_time:.2f}s")
    print(f"Saved processing steps to: {image_output_dir}")
    print(f"Final combined output: {final_path}")

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
    parser.add_argument(
        "--reuse-ocr",
        action="store_true",
        help="Reuse cached OCR results when available",
    )

    args = parser.parse_args()

    # Resolve paths
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    print("OCR Pipeline with Menu Detection")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if args.reuse_ocr:
        print("OCR reuse: enabled (will load cached detections when available)")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    ocr_model = None

    def get_ocr_model():
        nonlocal ocr_model
        if ocr_model is None:
            print("\nInitializing OCR model...")
            ocr_model = init_ocr()
            print("Model ready.")
        return ocr_model

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
            process_image(get_ocr_model, image_path, output_dir, reuse_ocr=args.reuse_ocr)
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
