"""
Visualization functions for OCR results, text groups, and menu regions.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
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


def draw_menu_regions(image_rgb, menu_results):
    """
    Draw menu detection strips with color-coded status.
    Red = detected menu, Yellow = uncertain, Gray = not a menu
    """
    pil = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(pil, "RGBA")

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

    for region in menu_results:
        name = region["name"]
        status = region.get("status")
        score = region.get("score", 0.0)
        rect = region.get("rect")

        if not rect:
            continue

        x0, y0, x1, y1 = [int(round(v)) for v in rect]
        color = status_colors.get(status, (128, 128, 128, 30))

        # Draw rectangle
        draw.rectangle(
            [(x0, y0), (x1, y1)],
            fill=color,
            outline=(color[0], color[1], color[2], 200),
            width=3,
        )

        # Draw label
        label = f"{name.upper()}: {status or 'none'} ({score:.2f})"
        if region.get("notes"):
            label += f" [{region['notes']}]"

        # Position label inside the strip
        if name == "top":
            text_pos = (10, 10)
        elif name == "left":
            text_pos = (x0 + 10, y0 + (y1 - y0) // 2)
        else:  # right
            text_pos = (max(0, x1 - 10 - len(label) * 10), y0 + (y1 - y0) // 2)

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

        divider = region.get("divider")
        if divider and status:
            draw.line(
                [(divider[0], divider[1]), (divider[2], divider[3])],
                fill=(0, 255, 255, 220),
                width=3,
            )

    return pil


def create_combined_output(
    image_rgb, boxes, texts, scores, groups, dropped_indices, menu_results
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
    result = draw_menu_regions(result_array, menu_results)

    return result


def draw_menu_stage(image_rgb, menu_results, stage="initial"):
    """
    Draw menu regions for a specific processing stage ("initial" or "final").
    """
    pil = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(pil, "RGBA")

    colors = {
        "menu": (255, 0, 0, 60),
        "maybe": (255, 255, 0, 60),
        None: (128, 128, 128, 30),
    }

    font = None
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        pass

    for region in menu_results:
        if stage == "initial":
            rect = region.get("initial_rect")
            status = region.get("initial_status")
            score = region.get("initial_score", 0.0)
        else:
            rect = region.get("rect")
            status = region.get("status")
            score = region.get("score", 0.0)

        if not rect:
            continue

        x0, y0, x1, y1 = [int(round(v)) for v in rect]
        color = colors.get(status, (128, 128, 128, 30))

        draw.rectangle(
            [(x0, y0), (x1, y1)],
            fill=color,
            outline=(color[0], color[1], color[2], 200),
            width=3,
        )

        label = (
            f"{region['name'].upper()} {stage}: {status or 'none'} ({score:.2f})"
        )
        text_width = (
            draw.textlength(label, font=font)
            if hasattr(draw, "textlength") and font
            else 10 * len(label)
        )
        text_height = font.size if font else 20
        draw.rectangle(
            [(x0 + 10, y0 + 10), (x0 + 10 + text_width + 8, y0 + 10 + text_height + 4)],
            fill=(0, 0, 0, 200),
        )
        draw.text(
            (x0 + 14, y0 + 12),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )

        if stage == "final":
            divider = region.get("divider")
            if divider and status:
                draw.line(
                    [(divider[0], divider[1]), (divider[2], divider[3])],
                    fill=(0, 255, 255, 220),
                    width=3,
                )

    return pil


def draw_divider_edges(image_rgb, boxes, menu_results):
    """
    Show the blurred grayscale image used for Hough-based divider detection, highlighting edge responses and line candidates.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_blurred = gray.copy()
    height, width = gray.shape

    for box in boxes:
        if not box:
            continue
        bx0, by0, bx1, by1 = box_bounds(box)
        x0 = max(0, int(round(bx0)))
        y0 = max(0, int(round(by0)))
        x1 = min(width, int(round(bx1)))
        y1 = min(height, int(round(by1)))
        if x1 - x0 <= 1 or y1 - y0 <= 1:
            continue
        region = gray_blurred[y0:y1, x0:x1]
        if region.size == 0:
            continue
        kernel_w = 5 if (x1 - x0) >= 5 else 3
        kernel_h = 5 if (y1 - y0) >= 5 else 3
        if kernel_w < 3 or kernel_h < 3:
            continue
        blurred = cv2.GaussianBlur(region, (kernel_w, kernel_h), 0)
        gray_blurred[y0:y1, x0:x1] = blurred

    overlay_bgr = cv2.cvtColor(gray_blurred, cv2.COLOR_GRAY2BGR)

    for region in menu_results:
        rect = region.get("rect")
        status = region.get("status")
        if not rect:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in rect]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image_rgb.shape[1], x1), min(image_rgb.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            continue

        edges = region.get("divider_edges")
        if edges is not None:
            edges_arr = np.asarray(edges)
            region_h, region_w = edges_arr.shape[:2]
            target_h = y1 - y0
            target_w = x1 - x0
            if region_h != target_h or region_w != target_w:
                edges_arr = cv2.resize(
                    edges_arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST
                )
            mask = edges_arr > 0
            if np.any(mask):
                overlay_region = overlay_bgr[y0:y1, x0:x1]
                overlay_region[mask] = [0, 255, 255]  # Yellow in BGR
                overlay_bgr[y0:y1, x0:x1] = overlay_region

        candidates = region.get("divider_candidates") or []
        for cand in candidates:
            xA, yA, xB, yB = cand
            xA = int(round(xA))
            yA = int(round(yA))
            xB = int(round(xB))
            yB = int(round(yB))
            cv2.line(
                overlay_bgr,
                (xA, yA),
                (xB, yB),
                (255, 200, 0),
                1,
            )

        divider = region.get("divider")
        if divider and status:
            cv2.line(
                overlay_bgr,
                (int(divider[0]), int(divider[1])),
                (int(divider[2]), int(divider[3])),
                (255, 0, 0),  # Blue in BGR
                2,
            )

    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb)
