# %% [markdown]
# # PaddleOCR Quickstart
#
# This notebook demonstrates running PaddleOCR on an image and visualizing its quadrilateral detections.

# %%
from pathlib import Path
from collections.abc import Mapping
from numbers import Number


import cv2
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path("/Users/mattcarroll/code/memoir")
DEFAULT_IMAGE_PATH = (
    PROJECT_ROOT
    / "logs"
    / "benchmark_output"
    / "downscaled_2025-10-13_16-56-56-636984.png"
)
OUTPUT_IMAGE_PATH = DEFAULT_IMAGE_PATH.with_name(
    DEFAULT_IMAGE_PATH.stem + "_annotated" + DEFAULT_IMAGE_PATH.suffix
)

print(f"Default image path: {DEFAULT_IMAGE_PATH}")

# %%
# Initialize PaddleOCR model once
ocr = PaddleOCR(use_angle_cls=True, lang="en")
print("Model ready.")

# %%
# Load image and display
image_path = DEFAULT_IMAGE_PATH
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

image = cv2.imread(str(image_path))
if image is None:
    raise ValueError(f"Unable to read image: {image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(image_rgb)
plt.axis("off")
plt.title("Original Image")

# %%
# Run OCR
ocr_result = ocr.ocr(str(image_path))
print(f"Detected {len(ocr_result[0])} text regions.")


# Use the same image representation PaddleOCR used for inference when available.
viz_image_rgb = image_rgb
primary_result = ocr_result[0] if ocr_result else None
if isinstance(primary_result, Mapping):
    doc_pre = primary_result.get("doc_preprocessor_res") or {}
    output_img = doc_pre.get("output_img")
    if output_img is not None:
        viz_image_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        print(
            "Using doc-preprocessor output for visualization to match OCR coordinates."
        )


# %%
# --- 1) Normalizer: handles 2.x/3.x shapes, batched lists, and dict-style results ---
def normalize_ocr_result(ocr_result):
    """
    Returns (boxes, texts, scores).
    Works with:
      - [[ [x,y]x4, (text, score) ], ...]
      - [[ [x,y]x4, "text" ], ...]
      - {'data': [{'text_region':..., 'text':..., 'confidence':...}, ...]}  (3.x predict)
      - Batched list: [ [...], ... ]
    """

    # Newer PaddleOCR 3.x `OCRResult` objects behave like mappings.
    def _ensure_box_list(raw_box):
        if raw_box is None:
            return None
        if hasattr(raw_box, "reshape"):
            coords = raw_box.reshape(-1, 2)
            return [[float(x), float(y)] for x, y in coords]
        if hasattr(raw_box, "tolist"):
            coords = raw_box.tolist()
        else:
            coords = raw_box
        if not coords:
            return None
        if isinstance(coords[0], (list, tuple)):
            return [[float(x), float(y)] for x, y in coords]
        # flatten list like [x1, y1, x2, y2, ...]
        pts = list(zip(coords[::2], coords[1::2]))
        return [[float(x), float(y)] for x, y in pts]

    def _ensure_score(value):
        if value is None:
            return None
        if isinstance(value, Number):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if isinstance(ocr_result, list):
        if not ocr_result:
            return [], [], []
        if isinstance(ocr_result[0], Mapping):
            boxes, texts, scores = [], [], []
            for item in ocr_result:
                b, t, s = normalize_ocr_result(item)
                boxes.extend(b)
                texts.extend(t)
                if s:
                    scores.extend(s)
                else:
                    scores.extend([None] * len(b))
            if len(scores) < len(boxes):
                scores.extend([None] * (len(boxes) - len(scores)))
            return boxes, texts, scores
        # Legacy nested batch structure [[...], [...]] should fall through

    if isinstance(ocr_result, Mapping):
        data = ocr_result.get("data")
        if data:
            boxes, texts, scores = [], [], []
            for item in data:
                boxes.append(_ensure_box_list(item.get("text_region")))
                texts.append(item.get("text", ""))
                scores.append(_ensure_score(item.get("confidence")))
            return boxes, texts, scores

        boxes = (
            ocr_result.get("rec_polys")
            or ocr_result.get("rec_boxes")
            or ocr_result.get("dt_polys")
        )
        texts = ocr_result.get("rec_texts")
        scores = ocr_result.get("rec_scores")
        if boxes is not None and texts is not None:
            box_list = [_ensure_box_list(box) for box in boxes]
            score_list = [_ensure_score(score) for score in (scores or [])]
            if len(score_list) < len(box_list):
                score_list.extend([None] * (len(box_list) - len(score_list)))
            return box_list, list(texts), score_list

    # Possibly batched: take first batch if needed
    lines = ocr_result
    if (
        isinstance(lines, list)
        and lines
        and isinstance(lines[0], list)
        and isinstance(lines[0][0], (list, tuple))
    ):
        # Typical .ocr(...) returns [ [line, line, ...] ]
        lines = lines[0]

    boxes, texts, scores = [], [], []
    for line in lines or []:
        if not line:
            continue

        # Some formats provide dicts
        if isinstance(line, Mapping) and "text_region" in line:
            box = _ensure_box_list(line.get("text_region"))
            text = line.get("text", "")
            score = _ensure_score(line.get("confidence"))
            boxes.append(box)
            texts.append(text)
            scores.append(score)
            continue

        if isinstance(line, Mapping):
            b, t, s = normalize_ocr_result(line)
            boxes.extend(b)
            texts.extend(t)
            scores.extend(s)
            continue

        box = _ensure_box_list(line[0])
        info = line[1] if len(line) > 1 else ""

        if isinstance(info, (list, tuple)):
            # Expected (text, score)
            text = str(info[0]) if len(info) > 0 else ""
            score = _ensure_score(info[1] if len(info) > 1 else None)
        elif isinstance(info, dict):
            # Rare: dict with keys
            text = info.get("text", "")
            score = _ensure_score(info.get("confidence"))
        elif isinstance(info, str):
            text, score = info, None
        else:
            text, score = str(info), None

        boxes.append(box)
        texts.append(text)
        scores.append(score)

    return boxes, texts, scores


def draw_ocr_min(
    image_rgb, boxes, texts=None, scores=None, drop_score=0.5, font_path=None
):
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


# %%
# Draw quadrilateral boxes
boxes, texts, scores = normalize_ocr_result(ocr_result)

visualized = draw_ocr_min(
    viz_image_rgb, boxes, texts, scores, drop_score=0.5, font_path=None
)
plt.figure(figsize=(12, 8))
plt.imshow(visualized)
plt.axis("off")
plt.title("PaddleOCR Detections")
plt.tight_layout()

visualized.save(OUTPUT_IMAGE_PATH)
print(f"Annotated image saved to {OUTPUT_IMAGE_PATH}")

# %%
# Print out OCR results
if isinstance(ocr_result, list) and ocr_result and isinstance(ocr_result[0], Mapping):
    for idx, (text, score, box) in enumerate(
        zip(texts, scores or [None] * len(texts), boxes), start=1
    ):
        score_display = f"{score:.2f}" if isinstance(score, Number) else "--"
        print(f"{idx:02d}. {text} (confidence: {score_display})")
        print(f"    Box: {box}")
else:
    for idx, line in enumerate(ocr_result[0]):
        box, (text, score) = line
        print(f"{idx + 1:02d}. {text} (confidence: {score:.2f})")
        print(f"    Box: {box}")
