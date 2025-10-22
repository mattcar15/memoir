"""
Visualization functions for OCR results, text groups, and menu regions.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import box_bounds


def draw_ocr_boxes(
    image_rgb, boxes, texts=None, scores=None, drop_score=0.5, font_path=None
):
    """
    Draw OCR bounding boxes on image with optional text labels.
    """
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, 16) if font_path else None

    for i, box in enumerate(boxes):
        score = scores[i] if scores and i < len(scores) else None
        if score is not None and score < drop_score:
            continue

        poly = [(float(x), float(y)) for x, y in (box or [])]
        if not poly:
            continue
        draw.line(poly + [poly[0]], fill="lime", width=2)

        if texts and i < len(texts):
            label = str(texts[i])
            text_width = (
                draw.textlength(label, font=font)
                if hasattr(draw, "textlength")
                else 8 * len(label)
            )
            text_height = font.size if font else 16
            text_pos = (poly[0][0], max(0, poly[0][1] - text_height - 2))
            background = [
                (text_pos[0], text_pos[1]),
                (text_pos[0] + text_width + 4, text_pos[1] + text_height + 4),
            ]
            draw.rectangle(background, fill="black")
            draw.text(
                (text_pos[0] + 2, text_pos[1] + 2), label, fill="white", font=font
            )

    return pil_image


def draw_grouped_regions(
    image_rgb, boxes, keep_groups, dropped_indices=None, font_path=None
):
    """
    Draw paragraph groups with colored overlays.
    """
    pil = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(pil, "RGBA")
    palette = [
        (31, 119, 180, 90),
        (255, 127, 14, 90),
        (44, 160, 44, 90),
        (214, 39, 40, 90),
        (148, 103, 189, 90),
        (140, 86, 75, 90),
        (227, 119, 194, 90),
        (127, 127, 127, 90),
        (188, 189, 34, 90),
        (23, 190, 207, 90),
    ]
    if dropped_indices:
        for i in dropped_indices:
            if not boxes[i]:
                continue
            poly = [(float(x), float(y)) for x, y in boxes[i]]
            draw.polygon(poly, outline=(180, 180, 180, 180))
    for group_idx, group in enumerate(keep_groups):
        color = palette[group_idx % len(palette)]
        outline = (color[0], color[1], color[2], 220)
        for i in group:
            poly = [(float(x), float(y)) for x, y in boxes[i]]
            draw.polygon(poly, fill=color, outline=outline)
        first = group[0]
        x0, y0, _, _ = box_bounds(boxes[first])
        label = f"G{group_idx + 1}"
        try:
            font = ImageFont.truetype(font_path, 16) if font_path else None
        except Exception:
            font = None
        text_width = (
            draw.textlength(label, font=font)
            if hasattr(draw, "textlength")
            else 8 * len(label)
        )
        text_height = font.size if font else 16
        y_label = max(0, y0 - text_height - 4)
        draw.rectangle(
            [(x0, y_label), (x0 + text_width + 6, y_label + text_height + 6)],
            fill=(0, 0, 0, 180),
        )
        draw.text((x0 + 3, y_label + 3), label, fill=(255, 255, 255, 255), font=font)
    return pil


def draw_menu_regions(image_rgb, menu_results, img_size):
    """
    Draw menu detection strips with color-coded status.
    Red = detected menu, Yellow = uncertain, Gray = not a menu
    """
    pil = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(pil, "RGBA")

    W, H = img_size
    strips = {
        "top": (0, 0, W, int(0.16 * H)),
        "left": (0, 0, int(0.18 * W), H),
        "right": (int(0.82 * W), 0, W, H),
    }

    status_colors = {
        "menu": (255, 0, 0, 60),  # Red
        "maybe": (255, 255, 0, 60),  # Yellow
        None: (128, 128, 128, 30),  # Gray
    }

    font = None
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        pass

    for name, status, score in menu_results:
        rect = strips[name]
        color = status_colors.get(status, (128, 128, 128, 30))

        # Draw rectangle
        draw.rectangle(
            rect, fill=color, outline=(color[0], color[1], color[2], 200), width=3
        )

        # Draw label
        label = f"{name.upper()}: {status or 'none'} ({score:.2f})"

        # Position label inside the strip
        if name == "top":
            text_pos = (10, 10)
        elif name == "left":
            text_pos = (10, H // 2)
        else:  # right
            text_pos = (int(0.83 * W), H // 2)

        text_width = (
            draw.textlength(label, font=font)
            if hasattr(draw, "textlength") and font
            else 10 * len(label)
        )
        text_height = font.size if font else 20

        # Background for text
        draw.rectangle(
            [text_pos, (text_pos[0] + text_width + 8, text_pos[1] + text_height + 4)],
            fill=(0, 0, 0, 200),
        )
        draw.text(
            (text_pos[0] + 4, text_pos[1] + 2),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )

    return pil


def create_combined_output(
    image_rgb, boxes, texts, scores, groups, dropped_indices, menu_results, img_size
):
    """
    Create a composite visualization with OCR boxes, paragraph groups, and menu regions.
    """
    # Start with grouped regions
    result = draw_grouped_regions(
        image_rgb, boxes, groups, dropped_indices, font_path=None
    )

    # Overlay menu regions
    result_array = np.array(result)
    result = draw_menu_regions(result_array, menu_results, img_size)

    return result
