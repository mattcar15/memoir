"""
OCR with Sidebar Filtering using Apple Vision and Menu Detection

Combines Apple Vision's OCR with the pipeline's sidebar detection to filter out
menu/sidebar text and return only the important content area OCR.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

# Import Apple Vision OCR functions
from apple_vision import (
    recognize_text_with_boxes,
    _normalized_to_pixel_box,
)

# Import local modules
from geometry import vision_box_to_polygon, box_in_region
from menu_utils import adjust_menu_rectangles_for_display, resize_image_for_detection
from visualization import (
    save_cropped_content,
    save_annotated_with_sidebars,
    print_ocr_summary,
)

# Add pipeline directory to path for menu detection imports
pipeline_dir = Path(__file__).resolve().parent.parent / "pipeline"
sys.path.insert(0, str(pipeline_dir))

from menu_detection import detect_menus


def ocr_with_sidebar_filter(path: str, return_all: bool = False) -> Dict:
    """
    Run OCR on an image and filter out sidebar/menu text, segmenting by region.

    Args:
        path: Path to the input image
        return_all: If True, also return sidebar text separately

    Returns:
        Dict with:
        - 'content_text': OCR text from the main content area (joined)
        - 'content_detections': List of detections from main content (body)
        - 'left_sidebar_text': OCR text from left sidebar
        - 'left_sidebar_detections': List of detections from left sidebar
        - 'right_sidebar_text': OCR text from right sidebar
        - 'right_sidebar_detections': List of detections from right sidebar
        - 'top_bar_text': OCR text from top bar
        - 'top_bar_detections': List of detections from top bar
        - 'sidebar_text': Combined OCR text from all sidebars (if return_all=True)
        - 'sidebar_detections': Combined list of detections from sidebars (if return_all=True)
        - 'menu_results': Raw menu detection results
        - 'crop_region': The crop region used (x0, y0, x1, y1) in original image coordinates
        - 'image_size': (width, height) of the original image
    """
    path = os.path.abspath(path)

    # Load original image to get dimensions
    original_image = Image.open(path)
    orig_width, orig_height = original_image.size

    # Run Apple Vision OCR on original image (Vision handles its own scaling)
    detections = recognize_text_with_boxes(path)

    if not detections:
        return {
            "content_text": "",
            "content_detections": [],
            "left_sidebar_text": "",
            "left_sidebar_detections": [],
            "right_sidebar_text": "",
            "right_sidebar_detections": [],
            "top_bar_text": "",
            "top_bar_detections": [],
            "menu_results": [],
            "crop_region": (0, 0, orig_width, orig_height),
            "image_size": (orig_width, orig_height),
        }

    # Resize image for menu detection (matching pipeline behavior)
    resized_image, scale = resize_image_for_detection(original_image.convert("RGB"))
    resized_width, resized_height = resized_image.size
    image_rgb = np.array(resized_image)

    # Convert Vision boxes to pipeline polygon format at RESIZED scale
    boxes = []
    texts = []
    scores = []

    for det in detections:
        # Vision boxes are normalized, convert to resized pixel coordinates
        polygon = vision_box_to_polygon(det["bbox"], resized_width, resized_height)
        boxes.append(polygon)
        texts.append(det["text"])
        scores.append(None)

    # Run menu detection on resized image
    menu_results = detect_menus(image_rgb, boxes, texts)

    # Get adjusted menu rectangles (in resized coordinates)
    adjusted_menu_results = adjust_menu_rectangles_for_display(
        menu_results, resized_width, resized_height
    )

    # Build region rectangles and track boxes per region
    region_rects = {}  # name -> rect
    region_boxes = {"left": set(), "right": set(), "top": set()}

    # Calculate crop boundaries in resized coordinates
    crop_x0, crop_y0, crop_x1, crop_y1 = 0, 0, resized_width, resized_height

    for region in adjusted_menu_results:
        if region.get("status") in ("menu", "maybe"):
            name = region["name"]
            rect = region.get("rect_display")
            if rect:
                region_rects[name] = rect
                rx0, ry0, rx1, ry1 = rect

                # Update crop boundaries
                if name == "top":
                    crop_y0 = max(crop_y0, ry1)
                elif name == "left":
                    crop_x0 = max(crop_x0, rx1)
                elif name == "right":
                    crop_x1 = min(crop_x1, rx0)

                # Find boxes within this menu region
                for i, box in enumerate(boxes):
                    if box_in_region(box, rect):
                        region_boxes[name].add(i)

    # Handle overlap between regions - assign to the more prominent region
    _resolve_region_overlaps(region_boxes)

    # All sidebar indices (combined for backward compatibility)
    all_sidebar_indices = (
        region_boxes["left"] | region_boxes["right"] | region_boxes["top"]
    )

    # Scale crop region back to original image coordinates
    if scale < 1.0:
        orig_crop_x0 = int(round(crop_x0 / scale))
        orig_crop_y0 = int(round(crop_y0 / scale))
        orig_crop_x1 = int(round(crop_x1 / scale))
        orig_crop_y1 = int(round(crop_y1 / scale))
    else:
        orig_crop_x0, orig_crop_y0, orig_crop_x1, orig_crop_y1 = (
            crop_x0,
            crop_y0,
            crop_x1,
            crop_y1,
        )

    # Clamp to image bounds
    orig_crop_x0 = max(0, min(orig_width, orig_crop_x0))
    orig_crop_y0 = max(0, min(orig_height, orig_crop_y0))
    orig_crop_x1 = max(0, min(orig_width, orig_crop_x1))
    orig_crop_y1 = max(0, min(orig_height, orig_crop_y1))

    # Separate detections by region
    content_detections = []
    left_sidebar_detections = []
    right_sidebar_detections = []
    top_bar_detections = []
    sidebar_detections = []

    for i, det in enumerate(detections):
        if i in region_boxes["left"]:
            left_sidebar_detections.append(det)
            sidebar_detections.append(det)
        elif i in region_boxes["right"]:
            right_sidebar_detections.append(det)
            sidebar_detections.append(det)
        elif i in region_boxes["top"]:
            top_bar_detections.append(det)
            sidebar_detections.append(det)
        else:
            # Check if box is within the crop region (using original coords)
            pixel_box = _normalized_to_pixel_box(det["bbox"], orig_width, orig_height)
            bx0, by0, bx1, by1 = pixel_box
            if (
                bx0 >= orig_crop_x0
                and by0 >= orig_crop_y0
                and bx1 <= orig_crop_x1
                and by1 <= orig_crop_y1
            ):
                content_detections.append(det)

    # Build result with segmented text
    content_text = "\n".join(det["text"] for det in content_detections if det["text"])
    left_sidebar_text = "\n".join(
        det["text"] for det in left_sidebar_detections if det["text"]
    )
    right_sidebar_text = "\n".join(
        det["text"] for det in right_sidebar_detections if det["text"]
    )
    top_bar_text = "\n".join(det["text"] for det in top_bar_detections if det["text"])

    result = {
        "content_text": content_text,
        "content_detections": content_detections,
        "left_sidebar_text": left_sidebar_text,
        "left_sidebar_detections": left_sidebar_detections,
        "right_sidebar_text": right_sidebar_text,
        "right_sidebar_detections": right_sidebar_detections,
        "top_bar_text": top_bar_text,
        "top_bar_detections": top_bar_detections,
        "menu_results": menu_results,
        "crop_region": (orig_crop_x0, orig_crop_y0, orig_crop_x1, orig_crop_y1),
        "image_size": (orig_width, orig_height),
        "resized_size": (resized_width, resized_height),
        "scale": scale,
        "region_rects": region_rects,
    }

    if return_all:
        sidebar_text = "\n".join(
            det["text"] for det in sidebar_detections if det["text"]
        )
        result["sidebar_text"] = sidebar_text
        result["sidebar_detections"] = sidebar_detections

    return result


def _resolve_region_overlaps(region_boxes: Dict[str, set]) -> None:
    """
    Handle overlap between regions - assign to the more prominent region.
    Modifies region_boxes in place.
    """
    # Check for overlap between left sidebar and top bar
    left_top_overlap = region_boxes["left"] & region_boxes["top"]
    if left_top_overlap:
        left_count = len(region_boxes["left"])
        top_count = len(region_boxes["top"])

        if left_count >= top_count:
            # Left sidebar is more prominent, assign overlap to left
            region_boxes["top"] -= left_top_overlap
        else:
            # Top bar is more prominent, assign overlap to top
            region_boxes["left"] -= left_top_overlap

    # Check for overlap between right sidebar and top bar
    right_top_overlap = region_boxes["right"] & region_boxes["top"]
    if right_top_overlap:
        right_count = len(region_boxes["right"])
        top_count = len(region_boxes["top"])

        if right_count >= top_count:
            # Right sidebar is more prominent, assign overlap to right
            region_boxes["top"] -= right_top_overlap
        else:
            # Top bar is more prominent, assign overlap to top
            region_boxes["right"] -= right_top_overlap


# Convenience wrappers that use the main OCR function
def crop_content(
    path: str,
    output_path: Optional[str] = None,
    result: Optional[Dict] = None,
) -> str:
    """Save the cropped content area (excluding sidebars) as a separate image."""
    return save_cropped_content(
        path, output_path=output_path, result=result, ocr_func=ocr_with_sidebar_filter
    )


def annotate_with_sidebars(
    path: str,
    output_path: Optional[str] = None,
    save_cropped: bool = True,
    **kwargs,
) -> Dict[str, str]:
    """Create an annotated image showing content vs sidebar/top bar regions."""
    return save_annotated_with_sidebars(
        path,
        ocr_func=ocr_with_sidebar_filter,
        normalized_to_pixel_func=_normalized_to_pixel_box,
        output_path=output_path,
        save_cropped=save_cropped,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run OCR with sidebar filtering using Apple Vision"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to image file (default: sample test image)",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Save an annotated image showing content vs sidebar regions",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output path for annotated image"
    )

    args = parser.parse_args()

    # Default to a test image if none provided
    if args.image is None:
        sample_image = (
            Path(__file__).resolve().parent.parent
            / "test_images"
            / "20251021_123325_226_Spotify_Taylor_Swift_-_The_Fate_of_Ophelia.png"
        )
        if not sample_image.exists():
            # Try another test image
            test_images_dir = Path(__file__).resolve().parent.parent / "test_images"
            if test_images_dir.exists():
                images = list(test_images_dir.glob("*.png"))
                if images:
                    sample_image = images[0]
                else:
                    print("No test images found")
                    sys.exit(1)
    else:
        sample_image = Path(args.image)

    print("=" * 60)
    print("OCR with Sidebar Filtering")
    print("=" * 60)
    print(f"Processing: {sample_image}")

    # Run OCR with sidebar filtering
    result = ocr_with_sidebar_filter(str(sample_image), return_all=True)
    print_ocr_summary(result)

    # Optionally save annotated image and cropped content
    if args.annotate:
        paths = annotate_with_sidebars(
            str(sample_image), output_path=args.output, save_cropped=True
        )
        print(f"\nAnnotated image saved: {paths['annotated']}")
        if "cropped" in paths:
            print(f"Cropped content saved: {paths['cropped']}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
