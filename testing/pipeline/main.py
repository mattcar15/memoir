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

from utils import iter_image_paths, load_and_resize_image, box_bounds
from ocr_processing import init_ocr, run_ocr
from layout_analysis import filter_chrome, split_columns_dynamic, group_paragraphs
from menu_detection import detect_menus
from visualization import (
    create_combined_output,
    draw_divider_edges,
    draw_menu_stage,
    draw_ocr_boxes,
)
from elements import (
    draw_element_ocr_boxes,
    create_text_removed_image,
    detect_lines_and_cuts,
)


def adjust_menu_rectangles_for_display(menu_results, width, height):
    """
    Adjust menu rectangles to span full height/width aligned to dividers.
    This is the same logic used in create_combined_output visualization.
    """

    def _clamp(a, lo, hi):
        return max(lo, min(hi, a))

    adjusted = []
    for region in menu_results or []:
        r = dict(region)
        name = r.get("name")
        status = r.get("status")
        base_rect = r.get("rect_aligned") or r.get("rect")

        # Only adjust for confident/maybe menus and when a base rect exists
        if status in ("menu", "maybe") and base_rect:
            x0, y0, x1, y1 = [int(round(v)) for v in base_rect]

            # Prefer divider line as the alignment target when present
            divider = r.get("divider")
            if name == "left":
                target_x = None
                if divider:
                    target_x = int(round(min(divider[0], divider[2])))
                else:
                    # Fallback to current aligned inner edge
                    target_x = x1
                target_x = _clamp(target_x, 0, width)
                # Grow/shrink so the inner (right) edge aligns; span full height
                new_rect = (x0, 0, max(x0 + 1, target_x), height)
            elif name == "right":
                target_x = None
                if divider:
                    target_x = int(round(max(divider[0], divider[2])))
                else:
                    target_x = x0
                target_x = _clamp(target_x, 0, width)
                # Grow/shrink so the inner (left) edge aligns; span full height
                new_rect = (min(target_x, x1 - 1), 0, x1, height)
            elif name == "top":
                target_y = None
                if divider:
                    target_y = int(round(max(divider[1], divider[3])))
                else:
                    target_y = y1
                target_y = _clamp(target_y, 0, height)
                # Grow/shrink so the bottom edge aligns; span full width
                new_rect = (0, 0, width, max(1, target_y))
            else:
                new_rect = base_rect

            # Ensure valid ordering
            nx0, ny0, nx1, ny1 = new_rect
            if nx1 <= nx0:
                nx1 = nx0 + 1
            if ny1 <= ny0:
                ny1 = ny0 + 1
            r["rect_display"] = (int(nx0), int(ny0), int(nx1), int(ny1))

        adjusted.append(r)

    return adjusted


def process_image(
    get_ocr_model,
    image_path,
    output_dir,
    reuse_ocr=False,
    run_menu=True,
    run_elements=True,
):
    """Process a single image through the complete pipeline."""
    print(f"\nProcessing: {image_path.name}")
    print("=" * 80)

    start_time = time.perf_counter()

    image_output_dir = output_dir / image_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    menu_dir = image_output_dir / "menu_segmentation"
    element_dir = image_output_dir / "element_segmentation"

    if run_menu:
        menu_dir.mkdir(parents=True, exist_ok=True)
    if run_elements:
        element_dir.mkdir(parents=True, exist_ok=True)

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
    # Try to find cached OCR in multiple locations
    boxes, texts, scores = [], [], []
    loaded_from_cache = False
    ocr_json_path = None

    if reuse_ocr:
        reuse_start = time.perf_counter()
        # Check for cached OCR in order of preference
        possible_paths = [
            menu_dir / "ocr.json",  # Saved with menu segmentation
            image_output_dir / "ocr.json",  # Legacy location
        ]

        for path in possible_paths:
            if path.exists():
                ocr_json_path = path
                break

        if ocr_json_path:
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
                    f"OCR reuse enabled; loaded cached results from {ocr_json_path.relative_to(output_dir)} "
                    f"in {reuse_time:.2f}s"
                )
            except json.JSONDecodeError as exc:
                print(
                    f"Cached OCR data in {ocr_json_path.name} is invalid ({exc}); running OCR..."
                )
            except Exception as exc:
                print(
                    f"Failed to load cached OCR data ({exc}); running OCR with fresh results..."
                )
        else:
            print("OCR reuse enabled but no cached OCR found; running OCR...")

    if not loaded_from_cache:
        ocr_model = get_ocr_model()
        ocr_start = time.perf_counter()
        boxes, texts, scores = run_ocr(ocr_model, image_bgr)
        ocr_time = time.perf_counter() - ocr_start
        print(f"OCR time: {ocr_time:.2f}s")

        # Determine where to save OCR results
        if run_menu:
            ocr_json_path = menu_dir / "ocr.json"
        else:
            ocr_json_path = image_output_dir / "ocr.json"

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

    # These are needed for both pipelines
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

    # Step 4: Menu detection (run if either pipeline needs it)
    menu_results = None
    if run_menu or run_elements:
        menu_start = time.perf_counter()
        menu_results = detect_menus(image_rgb, boxes, texts)
        menu_time = time.perf_counter() - menu_start
        print(f"Menu detection time: {menu_time:.2f}s")

    if menu_results:
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

    # Step 5: Menu Segmentation - Visualize, extract text, and save
    if run_menu and menu_results:
        viz_start = time.perf_counter()
        output_image = create_combined_output(
            image_rgb, boxes, texts, scores, groups, drop_indices, menu_results
        )

        # Save output steps to menu_segmentation folder
        step_images = [
            ("step0_original.png", Image.fromarray(image_rgb)),
            ("step1_ocr_boxes.png", draw_ocr_boxes(image_rgb, boxes, texts, scores)),
            (
                "step2_initial_menus.png",
                draw_menu_stage(image_rgb, menu_results, stage="initial"),
            ),
            (
                "step3_refined_menus.png",
                draw_menu_stage(image_rgb, menu_results, stage="final"),
            ),
            (
                "step4_divider_edges.png",
                draw_divider_edges(image_rgb, boxes, menu_results),
            ),
            ("step5_combined.png", output_image),
        ]

        for filename, pil_image in step_images:
            save_path = menu_dir / filename
            pil_image.save(save_path)

        viz_time = time.perf_counter() - viz_start
        final_path = menu_dir / "step5_combined.png"
        print(f"Menu visualization time: {viz_time:.2f}s")
        print(f"Saved menu segmentation to: {menu_dir}")
        print(f"Final combined output: {final_path}")

        # Extract text groups excluding menu text
        extract_start = time.perf_counter()

        menu_indices = set()
        for region in menu_results:
            if region.get("status") in ("menu", "maybe"):
                menu_indices.update(region.get("indices", []))

        # Collect text from groups, excluding menu items
        text_groups = []
        for group_indices in groups:
            group_texts = []
            for idx in group_indices:
                if idx not in menu_indices and idx < len(texts):
                    text = texts[idx]
                    if text and text.strip():
                        group_texts.append(text.strip())

            if group_texts:
                # Join texts in this group with spaces
                group_text = " ".join(group_texts)
                text_groups.append(group_text)

        # Save as ordered list to menu_segmentation folder
        text_output_path = menu_dir / "extracted_text.txt"
        with text_output_path.open("w", encoding="utf-8") as f:
            for i, text in enumerate(text_groups, 1):
                f.write(f"{i}. {text}\n\n")

        print(f"Text extraction time: {time.perf_counter() - extract_start:.2f}s")
        print(
            f"Extracted {len(text_groups)} text groups (excluding {len(menu_indices)} menu items)"
        )
        print(f"Saved text groups to: {text_output_path}")

    # Step 6: Element Segmentation - Crop image excluding menu regions
    if run_elements and menu_results:
        crop_start = time.perf_counter()

        # Get the same adjusted rectangles used in the visualization
        adjusted_menu_results = adjust_menu_rectangles_for_display(menu_results, W, H)

        # Calculate crop boundaries and menu indices based on display rectangles
        crop_x0, crop_y0, crop_x1, crop_y1 = 0, 0, W, H
        display_menu_indices = set()

        for region in adjusted_menu_results:
            if region.get("status") in ("menu", "maybe"):
                name = region["name"]
                rect = region.get("rect_display")
                if rect:
                    rx0, ry0, rx1, ry1 = rect

                    # Determine crop boundaries
                    if name == "top":
                        # Crop from bottom of top menu
                        crop_y0 = max(crop_y0, ry1)
                    elif name == "left":
                        # Crop from right edge of left menu
                        crop_x0 = max(crop_x0, rx1)
                    elif name == "right":
                        # Crop to left edge of right menu
                        crop_x1 = min(crop_x1, rx0)

                    # Find boxes within the display menu region
                    for i, box in enumerate(boxes):
                        if not box:
                            continue
                        bx0, by0, bx1, by1 = box_bounds(box)
                        # Check if box overlaps with display menu region
                        if bx1 >= rx0 and bx0 <= rx1 and by1 >= ry0 and by0 <= ry1:
                            display_menu_indices.add(i)

        # Ensure valid crop region
        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            print("Warning: Invalid crop region, skipping crop")
        else:
            # Crop the image
            cropped_image = image_rgb[crop_y0:crop_y1, crop_x0:crop_x1]

            # Filter and adjust OCR boxes for cropped region
            cropped_boxes = []
            cropped_texts = []
            cropped_scores = []

            for i, box in enumerate(boxes):
                if not box or i in display_menu_indices:
                    continue

                # Get box bounds
                bx0, by0, bx1, by1 = box_bounds(box)

                # Check if box is within cropped region
                if (
                    bx0 >= crop_x0
                    and by0 >= crop_y0
                    and bx1 <= crop_x1
                    and by1 <= crop_y1
                ):
                    # Adjust coordinates relative to cropped region
                    if isinstance(box, list) and len(box) == 4:
                        # Polygon format [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
                        adjusted_box = [
                            [pt[0] - crop_x0, pt[1] - crop_y0] for pt in box
                        ]
                    else:
                        # Handle other formats if needed
                        adjusted_box = box

                    cropped_boxes.append(adjusted_box)
                    cropped_texts.append(texts[i] if i < len(texts) else "")
                    cropped_scores.append(scores[i] if i < len(scores) else None)

            # Save cropped OCR JSON to element_segmentation folder
            cropped_ocr_path = element_dir / "cropped_ocr.json"
            with cropped_ocr_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "boxes": cropped_boxes,
                        "texts": cropped_texts,
                        "scores": cropped_scores,
                        "crop_region": {
                            "x0": int(crop_x0),
                            "y0": int(crop_y0),
                            "x1": int(crop_x1),
                            "y1": int(crop_y1),
                        },
                        "original_size": {"width": W, "height": H},
                        "cropped_size": {
                            "width": int(crop_x1 - crop_x0),
                            "height": int(crop_y1 - crop_y0),
                        },
                    },
                    f,
                    indent=2,
                )

            # Generate element segmentation steps
            element_viz_start = time.perf_counter()

            # Step 1: Cropped image
            step1_cropped = Image.fromarray(cropped_image)

            # Step 2: OCR box overlay
            step2_ocr_overlay = draw_element_ocr_boxes(
                cropped_image, cropped_boxes, cropped_texts, cropped_scores
            )

            # Step 3: Text-removed B&W image
            step3_text_removed = create_text_removed_image(cropped_image, cropped_boxes)
            step3_text_removed_pil = Image.fromarray(step3_text_removed)

            # Step 4: Edge detection with Hough transform and XY-cut
            segmentation_results = detect_lines_and_cuts(cropped_image, cropped_boxes)
            step4_segmentation = segmentation_results["visualization"]

            # Save all element segmentation steps
            element_steps = [
                ("step1_cropped_image.png", step1_cropped),
                ("step2_ocr_overlay.png", step2_ocr_overlay),
                ("step3_text_removed_bw.png", step3_text_removed_pil),
                ("step4_segmentation.png", step4_segmentation),
            ]

            for filename, pil_image in element_steps:
                save_path = element_dir / filename
                pil_image.save(save_path)

            # Also save segmentation data as JSON
            segmentation_data_path = element_dir / "segmentation_data.json"
            with segmentation_data_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "horizontal_lines": segmentation_results["lines"]["horizontal"],
                        "vertical_lines": segmentation_results["lines"]["vertical"],
                        "horizontal_cuts": segmentation_results["cut_lines"][
                            "horizontal"
                        ],
                        "vertical_cuts": segmentation_results["cut_lines"]["vertical"],
                    },
                    f,
                    indent=2,
                )

            element_viz_time = time.perf_counter() - element_viz_start
            crop_time = time.perf_counter() - crop_start

            print(f"Crop time: {crop_time:.2f}s")
            print(f"Crop region: ({crop_x0}, {crop_y0}) → ({crop_x1}, {crop_y1})")
            print(f"Cropped size: {crop_x1-crop_x0}x{crop_y1-crop_y0}")
            print(
                f"Cropped OCR boxes: {len(cropped_boxes)} (excluded {len(display_menu_indices)} "
                f"menu boxes from {len(boxes)} original)"
            )
            print(f"Element visualization time: {element_viz_time:.2f}s")
            print(
                f"Detected {len(segmentation_results['lines']['horizontal'])} horizontal lines, "
                f"{len(segmentation_results['lines']['vertical'])} vertical lines"
            )
            print(
                f"Found {len(segmentation_results['cut_lines']['horizontal'])} horizontal cuts, "
                f"{len(segmentation_results['cut_lines']['vertical'])} vertical cuts"
            )
            print(f"Saved element segmentation to: {element_dir}")
            print(f"Saved cropped OCR to: {cropped_ocr_path}")
            print(f"Saved segmentation data to: {segmentation_data_path}")

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
    parser.add_argument(
        "--run-menu",
        action="store_true",
        help="Run only menu segmentation",
    )
    parser.add_argument(
        "--run-elements",
        action="store_true",
        help="Run only element segmentation",
    )

    args = parser.parse_args()

    # Determine which pipelines to run
    run_menu = args.run_menu
    run_elements = args.run_elements

    # If neither flag is set, run both (default behavior)
    if not run_menu and not run_elements:
        run_menu = True
        run_elements = True

    # Resolve paths
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    print("OCR Pipeline with Menu Detection")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if args.reuse_ocr:
        print("OCR reuse: enabled (will load cached detections when available)")

    print(
        f"Running: Menu={'Yes' if run_menu else 'No'}, Elements={'Yes' if run_elements else 'No'}"
    )

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
            process_image(
                get_ocr_model,
                image_path,
                output_dir,
                reuse_ocr=args.reuse_ocr,
                run_menu=run_menu,
                run_elements=run_elements,
            )
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
