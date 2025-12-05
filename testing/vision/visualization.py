"""
Visualization utilities for OCR results - annotation, cropping, and printing.
"""

from pathlib import Path
from typing import Callable, Dict, Optional

from PIL import Image, ImageDraw, ImageFont


def save_cropped_content(
    path: str,
    output_path: Optional[str] = None,
    result: Optional[Dict] = None,
    ocr_func: Optional[Callable] = None,
) -> str:
    """
    Save the cropped content area (excluding sidebars) as a separate image.

    Args:
        path: Path to the input image
        output_path: Path to save cropped image (default: adds _cropped suffix)
        result: Pre-computed result from ocr_with_sidebar_filter (optional)
        ocr_func: OCR function to use if result not provided

    Returns:
        Path to the saved cropped image
    """
    if result is None:
        if ocr_func is None:
            raise ValueError("Either result or ocr_func must be provided")
        result = ocr_func(path, return_all=True)

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
    ocr_func: Callable,
    normalized_to_pixel_func: Callable,
    output_path: Optional[str] = None,
    content_color: str = "green",
    left_sidebar_color: str = "red",
    right_sidebar_color: str = "orange",
    top_bar_color: str = "purple",
    crop_color: str = "blue",
    save_cropped: bool = True,
) -> Dict[str, str]:
    """
    Create an annotated image showing content vs sidebar/top bar regions.

    Content text boxes are drawn in green, left sidebar in red, right sidebar in orange,
    top bar in purple, and the crop region is outlined in blue.

    Args:
        path: Path to the input image
        ocr_func: OCR function to call (ocr_with_sidebar_filter)
        normalized_to_pixel_func: Function to convert normalized bbox to pixels
        output_path: Path to save annotated image
        content_color: Color for content (body) text boxes
        left_sidebar_color: Color for left sidebar text boxes
        right_sidebar_color: Color for right sidebar text boxes
        top_bar_color: Color for top bar text boxes
        crop_color: Color for crop region outline
        save_cropped: Also save the cropped content area

    Returns:
        Dict with paths: {'annotated': path, 'cropped': path}
    """
    result = ocr_func(path, return_all=True)

    image_path = Path(path)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    stroke_width = max(2, int(min(image.size) * 0.002))

    width, height = image.size
    crop = result["crop_region"]

    # Draw semi-transparent overlays on each region with distinct colors
    left_overlay = (255, 0, 0, 60)  # Red with alpha
    right_overlay = (255, 165, 0, 60)  # Orange with alpha
    top_overlay = (128, 0, 128, 60)  # Purple with alpha

    # Left sidebar overlay
    if crop[0] > 0:
        draw.rectangle([0, 0, crop[0], height], fill=left_overlay)
    # Right sidebar overlay
    if crop[2] < width:
        draw.rectangle([crop[2], 0, width, height], fill=right_overlay)
    # Top bar overlay (only the portion not covered by sidebars)
    if crop[1] > 0:
        draw.rectangle([crop[0], 0, crop[2], crop[1]], fill=top_overlay)

    # Draw left sidebar detections
    for det in result.get("left_sidebar_detections", []):
        rect = normalized_to_pixel_func(det["bbox"], width, height)
        draw.rectangle(rect, outline=left_sidebar_color, width=stroke_width)

    # Draw right sidebar detections
    for det in result.get("right_sidebar_detections", []):
        rect = normalized_to_pixel_func(det["bbox"], width, height)
        draw.rectangle(rect, outline=right_sidebar_color, width=stroke_width)

    # Draw top bar detections
    for det in result.get("top_bar_detections", []):
        rect = normalized_to_pixel_func(det["bbox"], width, height)
        draw.rectangle(rect, outline=top_bar_color, width=stroke_width)

    # Draw content detections (green boxes)
    for det in result["content_detections"]:
        rect = normalized_to_pixel_func(det["bbox"], width, height)
        draw.rectangle(rect, outline=content_color, width=stroke_width)

    # Draw crop region (thick blue border)
    crop_rect = (crop[0], crop[1], crop[2], crop[3])
    draw.rectangle(crop_rect, outline=crop_color, width=stroke_width + 2)

    # Draw legend in top-right corner (avoiding content)
    legend_y = 10
    legend_x = width - 150
    legend_items = [
        (content_color, "Body"),
        (left_sidebar_color, "Left Sidebar"),
        (right_sidebar_color, "Right Sidebar"),
        (top_bar_color, "Top Bar"),
        (crop_color, "Crop Region"),
    ]

    # Legend background
    draw.rectangle(
        [legend_x - 5, legend_y - 5, width - 5, legend_y + len(legend_items) * 18 + 5],
        fill=(0, 0, 0, 180),
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
        cropped_path = save_cropped_content(
            path, output_path=cropped_path, result=result
        )
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

    # Show detection counts per region
    print(f"\nDetection counts by region:")
    print(f"  Body:          {len(result['content_detections'])}")
    print(f"  Left Sidebar:  {len(result.get('left_sidebar_detections', []))}")
    print(f"  Right Sidebar: {len(result.get('right_sidebar_detections', []))}")
    print(f"  Top Bar:       {len(result.get('top_bar_detections', []))}")
    if "sidebar_detections" in result:
        print(f"  Total Sidebar: {len(result['sidebar_detections'])}")

    print("\nMenu Detection Results:")
    for region in result["menu_results"]:
        name = region["name"]
        status = region.get("status") or "none"
        score = region.get("score", 0.0)
        print(f"  {name:>6s}: {status:>6s} (score={score:.2f})")

    # Body (content) text
    print("\n" + "=" * 60)
    print("BODY TEXT")
    print("=" * 60)
    print(result["content_text"] or "(no body text detected)")

    # Left sidebar text
    if result.get("left_sidebar_text"):
        print("\n" + "-" * 60)
        print("LEFT SIDEBAR TEXT")
        print("-" * 60)
        print(result["left_sidebar_text"])

    # Right sidebar text
    if result.get("right_sidebar_text"):
        print("\n" + "-" * 60)
        print("RIGHT SIDEBAR TEXT")
        print("-" * 60)
        print(result["right_sidebar_text"])

    # Top bar text
    if result.get("top_bar_text"):
        print("\n" + "-" * 60)
        print("TOP BAR TEXT")
        print("-" * 60)
        print(result["top_bar_text"])
