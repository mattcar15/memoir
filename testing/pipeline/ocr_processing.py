"""
OCR processing module for PaddleOCR integration.
"""

from collections.abc import Mapping
from numbers import Number
from paddleocr import PaddleOCR
import os
import platform

print(platform.machine())


def init_ocr():
    """Initialize PaddleOCR model with optimized settings."""
    ocr = PaddleOCR(
        lang="en",
        use_textline_orientation=False,  # new in 3.x; replaces use_angle_cls
        use_doc_unwarping=False,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_detection_model_dir=os.path.expanduser(
            "~/.paddlex/official_models/PP-OCRv5_mobile_det"
        ),
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        text_recognition_model_dir=os.path.expanduser(
            "~/.paddlex/official_models/en_PP-OCRv5_mobile_rec"
        ),
        text_det_limit_type="max",  # enforce a hard size limit on detector input
        text_det_limit_side_len=1080,  # detector won't process larger than this
        text_recognition_batch_size=4,  # smaller batches → lower peak memory
    )
    return ocr


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


def run_ocr(ocr_model, image_bgr):
    """Run OCR on image and return normalized results."""
    ocr_result = ocr_model.predict(image_bgr)
    boxes, texts, scores = normalize_ocr_result(ocr_result)
    return boxes, texts, scores
