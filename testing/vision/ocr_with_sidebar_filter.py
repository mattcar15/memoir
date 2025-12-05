"""
OCR with Sidebar Filtering using Apple Vision and Menu Detection

Combines Apple Vision's OCR with the pipeline's sidebar detection to filter out
menu/sidebar text and return only the important content area OCR.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import Apple Vision OCR functions
from apple_vision import (
    recognize_text_with_boxes,
    _normalized_to_pixel_box,
    _measure_text,
    load_image_as_cgimage,
)

# Add pipeline directory to path for menu detection imports
pipeline_dir = Path(__file__).resolve().parent.parent / "pipeline"
sys.path.insert(0, str(pipeline_dir))

from menu_detection import detect_menus

# Match the pipeline's resize behavior
MAX_LONG_SIDE = 1080


def vision_box_to_polygon(norm_box: Tuple[float, float, float, float], width: int, height: int) -> List[List[float]]:
    """
    Convert Vision's normalized box (origin bottom-left) to a 4-point polygon in pixel coordinates.
    
    Vision format: (x, y, w, h) where origin is bottom-left, values 0-1
    Pipeline format: [[x0,y0], [x1,y0], [x1,y1], [x0,y1]] in pixel coordinates
    """
    x, y, w, h = norm_box
    # Convert to pixel coordinates (flip Y axis since Vision uses bottom-left origin)
    px0 = x * width
    py0 = (1.0 - y - h) * height  # top-left Y
    px1 = (x + w) * width
    py1 = (1.0 - y) * height  # bottom-left Y
    
    return [[px0, py0], [px1, py0], [px1, py1], [px0, py1]]


def adjust_menu_rectangles_for_display(
    menu_results: List[Dict], width: int, height: int
) -> List[Dict]:
    """
    Adjust menu rectangles to span full height/width aligned to dividers.
    Adapted from main.py in the pipeline.
    """
    def _clamp(a, lo, hi):
        return max(lo, min(hi, a))

    adjusted = []
    for region in menu_results or []:
        r = dict(region)
        name = r.get("name")
        status = r.get("status")
        base_rect = r.get("rect_aligned") or r.get("rect")

        if status in ("menu", "maybe") and base_rect:
            x0, y0, x1, y1 = [int(round(v)) for v in base_rect]
            divider = r.get("divider")
            
            if name == "left":
                target_x = None
                if divider:
                    target_x = int(round(min(divider[0], divider[2])))
                else:
                    target_x = x1
                target_x = _clamp(target_x, 0, width)
                new_rect = (x0, 0, max(x0 + 1, target_x), height)
            elif name == "right":
                target_x = None
                if divider:
                    target_x = int(round(max(divider[0], divider[2])))
                else:
                    target_x = x0
                target_x = _clamp(target_x, 0, width)
                new_rect = (min(target_x, x1 - 1), 0, x1, height)
            elif name == "top":
                target_y = None
                if divider:
                    target_y = int(round(max(divider[1], divider[3])))
                else:
                    target_y = y1
                target_y = _clamp(target_y, 0, height)
                new_rect = (0, 0, width, max(1, target_y))
            else:
                new_rect = base_rect

            nx0, ny0, nx1, ny1 = new_rect
            if nx1 <= nx0:
                nx1 = nx0 + 1
            if ny1 <= ny0:
                ny1 = ny0 + 1
            r["rect_display"] = (int(nx0), int(ny0), int(nx1), int(ny1))

        adjusted.append(r)

    return adjusted


def box_bounds(box: List[List[float]]) -> Tuple[float, float, float, float]:
    """Get bounding box (x0, y0, x1, y1) from a 4-point polygon."""
    xs = [pt[0] for pt in box]
    ys = [pt[1] for pt in box]
    return min(xs), min(ys), max(xs), max(ys)


def box_in_region(box: List[List[float]], rect: Tuple[int, int, int, int]) -> bool:
    """Check if a box overlaps with a rectangle region."""
    rx0, ry0, rx1, ry1 = rect
    bx0, by0, bx1, by1 = box_bounds(box)
    return bx1 >= rx0 and bx0 <= rx1 and by1 >= ry0 and by0 <= ry1


def resize_image_for_detection(image: Image.Image, max_long_side: int = MAX_LONG_SIDE) -> Tuple[Image.Image, float]:
    """
    Resize image proportionally if it exceeds max_long_side.
    Returns (resized_image, scale_factor).
    """
    width, height = image.size
    max_dim = max(width, height)
    
    if max_dim > max_long_side:
        scale = max_long_side / max_dim
        new_size = (int(round(width * scale)), int(round(height * scale)))
        resized = image.resize(new_size, Image.LANCZOS)
        return resized, scale
    
    return image, 1.0


def ocr_with_sidebar_filter(
    path: str,
    return_all: bool = False
) -> Dict:
    """
    Run OCR on an image and filter out sidebar/menu text.
    
    Args:
        path: Path to the input image
        return_all: If True, also return sidebar text separately
        
    Returns:
        Dict with:
        - 'content_text': OCR text from the main content area (joined)
        - 'content_detections': List of detections from main content
        - 'sidebar_text': OCR text from detected sidebars (if return_all=True)
        - 'sidebar_detections': List of detections from sidebars (if return_all=True)
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
    adjusted_menu_results = adjust_menu_rectangles_for_display(menu_results, resized_width, resized_height)
    
    # Calculate crop boundaries in resized coordinates
    crop_x0, crop_y0, crop_x1, crop_y1 = 0, 0, resized_width, resized_height
    sidebar_indices = set()
    
    for region in adjusted_menu_results:
        if region.get("status") in ("menu", "maybe"):
            name = region["name"]
            rect = region.get("rect_display")
            if rect:
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
                        sidebar_indices.add(i)
    
    # Scale crop region back to original image coordinates
    if scale < 1.0:
        orig_crop_x0 = int(round(crop_x0 / scale))
        orig_crop_y0 = int(round(crop_y0 / scale))
        orig_crop_x1 = int(round(crop_x1 / scale))
        orig_crop_y1 = int(round(crop_y1 / scale))
    else:
        orig_crop_x0, orig_crop_y0, orig_crop_x1, orig_crop_y1 = crop_x0, crop_y0, crop_x1, crop_y1
    
    # Clamp to image bounds
    orig_crop_x0 = max(0, min(orig_width, orig_crop_x0))
    orig_crop_y0 = max(0, min(orig_height, orig_crop_y0))
    orig_crop_x1 = max(0, min(orig_width, orig_crop_x1))
    orig_crop_y1 = max(0, min(orig_height, orig_crop_y1))
    
    # Separate content and sidebar detections using original coordinates
    content_detections = []
    sidebar_detections = []
    
    for i, det in enumerate(detections):
        if i in sidebar_indices:
            sidebar_detections.append(det)
        else:
            # Check if box is within the crop region (using original coords)
            pixel_box = _normalized_to_pixel_box(det["bbox"], orig_width, orig_height)
            bx0, by0, bx1, by1 = pixel_box
            if bx0 >= orig_crop_x0 and by0 >= orig_crop_y0 and bx1 <= orig_crop_x1 and by1 <= orig_crop_y1:
                content_detections.append(det)
    
    # Build result
    content_text = "\n".join(det["text"] for det in content_detections if det["text"])
    
    result = {
        "content_text": content_text,
        "content_detections": content_detections,
        "menu_results": menu_results,
        "crop_region": (orig_crop_x0, orig_crop_y0, orig_crop_x1, orig_crop_y1),
        "image_size": (orig_width, orig_height),
        "resized_size": (resized_width, resized_height),
        "scale": scale,
    }
    
    if return_all:
        sidebar_text = "\n".join(det["text"] for det in sidebar_detections if det["text"])
        result["sidebar_text"] = sidebar_text
        result["sidebar_detections"] = sidebar_detections
    
    return result


def save_cropped_content(
    path: str,
    output_path: Optional[str] = None,
    result: Optional[Dict] = None,
) -> str:
    """
    Save the cropped content area (excluding sidebars) as a separate image.
    
    Args:
        path: Path to the input image
        output_path: Path to save cropped image (default: adds _cropped suffix)
        result: Pre-computed result from ocr_with_sidebar_filter (optional)
        
    Returns:
        Path to the saved cropped image
    """
    if result is None:
        result = ocr_with_sidebar_filter(path, return_all=True)
    
    image_path = Path(path)
    image = Image.open(image_path).convert("RGB")
    
    # Get crop region
    crop = result["crop_region"]
    x0, y0, x1, y1 = crop
    
    # Crop the image
    cropped = image.crop((x0, y0, x1, y1))
    
    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_cropped.png")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(output_path)
    
    return str(output_path)


def save_annotated_with_sidebars(
    path: str,
    output_path: Optional[str] = None,
    content_color: str = "green",
    sidebar_color: str = "red",
    crop_color: str = "blue",
    save_cropped: bool = True,
) -> Dict[str, str]:
    """
    Create an annotated image showing content vs sidebar regions.
    
    Content text boxes are drawn in green, sidebar text in red,
    and the crop region is outlined in blue.
    
    Args:
        path: Path to the input image
        output_path: Path to save annotated image
        content_color: Color for content text boxes
        sidebar_color: Color for sidebar text boxes
        crop_color: Color for crop region outline
        save_cropped: Also save the cropped content area
        
    Returns:
        Dict with paths: {'annotated': path, 'cropped': path}
    """
    result = ocr_with_sidebar_filter(path, return_all=True)
    
    image_path = Path(path)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    stroke_width = max(2, int(min(image.size) * 0.002))
    
    width, height = image.size
    crop = result["crop_region"]
    
    # Draw semi-transparent overlay on sidebar regions (areas outside crop)
    overlay_color = (255, 0, 0, 80)  # Red with alpha
    
    # Left sidebar overlay
    if crop[0] > 0:
        draw.rectangle([0, 0, crop[0], height], fill=overlay_color)
    # Right sidebar overlay  
    if crop[2] < width:
        draw.rectangle([crop[2], 0, width, height], fill=overlay_color)
    # Top bar overlay
    if crop[1] > 0:
        draw.rectangle([crop[0], 0, crop[2], crop[1]], fill=overlay_color)
    
    # Draw sidebar detections (red boxes)
    for det in result.get("sidebar_detections", []):
        rect = _normalized_to_pixel_box(det["bbox"], width, height)
        draw.rectangle(rect, outline=sidebar_color, width=stroke_width)
    
    # Draw content detections (green boxes)
    for det in result["content_detections"]:
        rect = _normalized_to_pixel_box(det["bbox"], width, height)
        draw.rectangle(rect, outline=content_color, width=stroke_width)
    
    # Draw crop region (thick blue border)
    crop_rect = (crop[0], crop[1], crop[2], crop[3])
    draw.rectangle(crop_rect, outline=crop_color, width=stroke_width + 2)
    
    # Draw legend in top-right corner (avoiding content)
    legend_y = 10
    legend_x = width - 150
    legend_items = [
        (content_color, "Content"),
        (sidebar_color, "Sidebar"),
        (crop_color, "Crop Region"),
    ]
    
    # Legend background
    draw.rectangle(
        [legend_x - 5, legend_y - 5, width - 5, legend_y + len(legend_items) * 18 + 5],
        fill=(0, 0, 0, 180)
    )
    
    for color, label in legend_items:
        draw.rectangle([legend_x, legend_y, legend_x + 20, legend_y + 12], fill=color)
        draw.text((legend_x + 25, legend_y), label, fill="white", font=font)
        legend_y += 18
    
    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_sidebar_filtered.png")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    
    paths = {"annotated": str(output_path)}
    
    # Also save cropped content area
    if save_cropped:
        cropped_path = image_path.with_name(f"{image_path.stem}_cropped.png")
        cropped_path = save_cropped_content(path, output_path=cropped_path, result=result)
        paths["cropped"] = cropped_path
    
    return paths


def print_ocr_summary(result: Dict) -> None:
    """Print a summary of OCR results with sidebar filtering."""
    width, height = result["image_size"]
    crop = result["crop_region"]
    
    print(f"\nOriginal image size: {width}x{height}")
    if "resized_size" in result:
        rw, rh = result["resized_size"]
        scale = result.get("scale", 1.0)
        print(f"Resized for detection: {rw}x{rh} (scale={scale:.3f})")
    print(f"Crop region: ({crop[0]}, {crop[1]}) → ({crop[2]}, {crop[3]})")
    print(f"Crop size: {crop[2]-crop[0]}x{crop[3]-crop[1]}")
    
    print(f"\nContent detections: {len(result['content_detections'])}")
    if "sidebar_detections" in result:
        print(f"Sidebar detections: {len(result['sidebar_detections'])}")
    
    print("\nMenu Detection Results:")
    for region in result["menu_results"]:
        name = region["name"]
        status = region.get("status") or "none"
        score = region.get("score", 0.0)
        print(f"  {name:>6s}: {status:>6s} (score={score:.2f})")
    
    print("\n" + "=" * 60)
    print("CONTENT TEXT (filtered)")
    print("=" * 60)
    print(result["content_text"] or "(no text detected)")
    
    if "sidebar_text" in result:
        print("\n" + "-" * 60)
        print("SIDEBAR TEXT (excluded)")
        print("-" * 60)
        print(result["sidebar_text"] or "(no sidebar text)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run OCR with sidebar filtering using Apple Vision"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to image file (default: sample test image)"
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Save an annotated image showing content vs sidebar regions"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for annotated image"
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
        paths = save_annotated_with_sidebars(
            str(sample_image),
            output_path=args.output,
            save_cropped=True
        )
        print(f"\nAnnotated image saved: {paths['annotated']}")
        if "cropped" in paths:
            print(f"Cropped content saved: {paths['cropped']}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

