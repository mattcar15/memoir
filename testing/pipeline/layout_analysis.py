"""
Layout analysis for text grouping and chrome filtering.
"""

import numpy as np
from utils import box_bounds, poly_height, x_center, y_top


def filter_chrome(
    image_rgb, boxes, texts, scores, left_ratio=0.12, top_ratio=0.06, min_conf=0.65
):
    """
    Filter out browser/app chrome from OCR results.
    Returns (keep_indices, drop_indices).
    """
    H, W = image_rgb.shape[:2]
    menu_tokens = {
        "home",
        "file",
        "edit",
        "view",
        "help",
        "settings",
        "sign",
        "login",
        "account",
        "share",
        "about",
    }
    keep, drop = [], []
    for i, box in enumerate(boxes):
        if not box:
            drop.append(i)
            continue
        conf = scores[i] if scores and i < len(scores) else None
        if conf is not None and conf < min_conf:
            drop.append(i)
            continue
        x0, y0, x1, y1 = box_bounds(box)
        if x1 <= left_ratio * W or y1 <= top_ratio * H:
            drop.append(i)
            continue
        text_lower = (texts[i] or "").lower()
        if any(tok in text_lower for tok in menu_tokens):
            if x0 < 0.25 * W or y0 < 0.12 * H:
                drop.append(i)
                continue
        keep.append(i)
    return keep, drop


def split_columns_dynamic(indices, boxes, W, eps_ratio=0.06, min_points=4):
    """
    Split text boxes into columns using DBSCAN clustering.
    Falls back to projection method if DBSCAN fails.
    """
    try:
        return _split_columns_dbscan(
            indices, boxes, W, eps_ratio=eps_ratio, min_points=min_points
        )
    except Exception:
        return _split_columns_projection(indices, boxes, W)


def _split_columns_dbscan(indices, boxes, W, eps_ratio=0.06, min_points=4):
    """N-column detection using DBSCAN clustering."""
    from sklearn.cluster import DBSCAN

    xs = np.array([x_center(boxes[i]) for i in indices], dtype=float)
    X = (xs / float(W)).reshape(-1, 1)
    db = DBSCAN(eps=eps_ratio, min_samples=max(3, min_points)).fit(X)
    labels = db.labels_
    cols = {}
    for idx, label in zip(indices, labels):
        if label == -1:
            continue
        cols.setdefault(label, []).append(idx)
    major = [c for c in cols.values() if len(c) >= min_points]
    if not major:
        return [sorted(indices, key=lambda i: y_top(boxes[i]))]

    def _mx(col):
        return np.mean([x_center(boxes[i]) for i in col])

    major.sort(key=_mx)
    rest = [i for i in indices if all(i not in c for c in major)]
    if rest:
        means = [_mx(c) for c in major]
        for i in rest:
            x = x_center(boxes[i])
            j = int(np.argmin([abs(x - m) for m in means]))
            major[j].append(i)
    for col in major:
        col.sort(key=lambda i: y_top(boxes[i]))
    return major


def _split_columns_projection(
    indices, boxes, W, bins=96, min_gutter_ratio=0.02, min_col_items=4
):
    """Column detection using horizontal projection."""
    hist = np.zeros(bins, dtype=float)
    to_bin = lambda x: int(np.clip(np.floor(x / W * bins), 0, bins - 1))
    for i in indices:
        x0 = box_bounds(boxes[i])[0]
        x1 = box_bounds(boxes[i])[2]
        b0, b1 = to_bin(x0), to_bin(x1)
        hist[b0 : b1 + 1] += 1.0
    threshold = np.percentile(hist, 25)
    min_gutter = max(1, int(min_gutter_ratio * bins))
    cuts, run = [], []
    for idx, val in enumerate(hist):
        if val <= threshold:
            run.append(idx)
        else:
            if len(run) >= min_gutter:
                cuts.append((run[0], run[-1]))
            run = []
    if len(run) >= min_gutter:
        cuts.append((run[0], run[-1]))
    cut_xs = [((a + b) / 2.0 / bins) * W for a, b in cuts]
    bands, edges = [], [0.0] + cut_xs + [float(W)]
    for a, b in zip(edges[:-1], edges[1:]):
        band = [i for i in indices if a <= x_center(boxes[i]) < b]
        if len(band) >= min_col_items:
            band.sort(key=lambda i: y_top(boxes[i]))
            bands.append(band)
    if not bands:
        return [sorted(indices, key=lambda i: y_top(boxes[i]))]
    return bands


def group_paragraphs(col_indices, boxes, gap_factor=1.5):
    """
    Group text boxes in a column into paragraphs based on vertical gaps.
    """
    if not col_indices:
        return []
    heights = [poly_height(boxes[i]) for i in col_indices]
    med_h = float(np.median(heights)) if heights else 16.0
    max_gap = gap_factor * med_h
    groups, cur, prev_y1 = [], [], None
    for i in col_indices:
        y0 = box_bounds(boxes[i])[1]
        y1 = box_bounds(boxes[i])[3]
        if prev_y1 is not None and (y0 - prev_y1) > max_gap and cur:
            groups.append(cur)
            cur = []
        cur.append(i)
        prev_y1 = y1
    if cur:
        groups.append(cur)
    return groups
