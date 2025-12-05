import os
import Vision
import Cocoa
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from Quartz import (
    CGColorSpaceCreateDeviceRGB,
    CGBitmapContextCreate,
    CGBitmapContextCreateImage,
    CGImageSourceCreateImageAtIndex,
    CGImageSourceCreateWithURL,
)


def load_image_as_cgimage(path):
    url = Cocoa.NSURL.fileURLWithPath_(path)
    # Prefer CoreGraphics to decode directly to CGImage to avoid NSImage rasterization quirks
    source = CGImageSourceCreateWithURL(url, None)
    if source is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    cgimage = CGImageSourceCreateImageAtIndex(source, 0, None)
    if cgimage is not None:
        return cgimage

    # Fallback: rasterize via bitmap context if ImageIO could not decode
    nsimage = Cocoa.NSImage.alloc().initWithContentsOfURL_(url)
    if nsimage is None:
        raise RuntimeError(f"Could not decode image via NSImage: {path}")

    w = int(nsimage.size().width)
    h = int(nsimage.size().height)
    if w == 0 or h == 0:
        raise RuntimeError("NSImage has zero size, cannot rasterize.")

    colorspace = CGColorSpaceCreateDeviceRGB()
    ctx = CGBitmapContextCreate(
        None, w, h, 8, w * 4, colorspace, 2  # kCGImageAlphaPremultipliedLast
    )

    bounds = Cocoa.NSMakeRect(0, 0, w, h)
    nsimage.drawInRect_(bounds)

    return CGBitmapContextCreateImage(ctx)


def _rect_to_tuple(rect):
    """Normalize various CGRect/NSRect bridge shapes to (x, y, w, h)."""
    try:
        origin = rect.origin
        size = rect.size
        return float(origin.x), float(origin.y), float(size.width), float(size.height)
    except AttributeError:
        pass

    try:
        (x, y), (w, h) = rect
        return float(x), float(y), float(w), float(h)
    except Exception as exc:  # pragma: no cover - only hits on API change
        raise TypeError(f"Unrecognized rect type: {rect!r}") from exc


def _normalized_to_pixel_box(norm_box, width, height):
    """Convert Vision's normalized box (origin bottom-left) to pixel box (origin top-left)."""
    x, y, w, h = norm_box
    px0 = x * width
    py0 = (1.0 - y - h) * height
    px1 = px0 + w * width
    py1 = py0 + h * height
    return (px0, py0, px1, py1)


def _perform_request(handler, request):
    """Call Vision handler and surface any error details."""
    result = handler.performRequests_error_([request], None)
    if isinstance(result, tuple):
        success, error = result
    else:
        success, error = result, None

    if not success:
        raise RuntimeError(f"Vision request failed: {error}")


def recognize_text_with_boxes(path):
    """Run Vision OCR and return detections with normalized bounding boxes."""
    path = os.path.abspath(path)
    cgimage = load_image_as_cgimage(path)

    if cgimage is None:
        raise RuntimeError("Failed to produce CGImage")

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cgimage, None
    )

    _perform_request(handler, request)

    results = request.results() or []
    detections = []
    for obs in results:
        bbox = _rect_to_tuple(obs.boundingBox())
        candidates = obs.topCandidates_(1)
        text = candidates[0].string() if candidates else ""
        detections.append(
            {
                "text": text,
                "bbox": bbox,  # normalized (x, y, w, h), origin bottom-left
            }
        )

    return detections


def ocr_image(path):
    detections = recognize_text_with_boxes(path)
    if not detections:
        print("No text detected")
        return ""

    return "\n".join(det["text"] for det in detections if det["text"])


def _measure_text(draw, text, font):
    """Cross-version text measurement for Pillow (textsize removed in newer versions)."""
    if hasattr(draw, "textbbox"):
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        return x1 - x0, y1 - y0
    # Fallback for older Pillow
    return draw.textsize(text, font=font)


def save_annotated_image(path, output_path=None, box_color="red"):
    """Create an annotated PNG showing OCR bounding boxes."""
    detections = recognize_text_with_boxes(path)
    image_path = Path(path)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    stroke_width = max(2, int(min(image.size) * 0.003))

    for det in detections:
        rect = _normalized_to_pixel_box(det["bbox"], image.width, image.height)
        draw.rectangle(rect, outline=box_color, width=stroke_width)
        if det["text"]:
            text = det["text"]
            text_w, text_h = _measure_text(draw, text, font)
            margin = 2
            x0, y0, _, _ = rect
            bg = (
                x0,
                max(0, y0 - text_h - margin * 2),
                x0 + text_w + margin * 2,
                max(0, y0 - text_h - margin * 2) + text_h + margin * 2,
            )
            draw.rectangle(bg, fill=box_color)
            draw.text((bg[0] + margin, bg[1] + margin), text, fill="white", font=font)

    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_annotated.png")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


# =============================================================================
# Contour Detection (UI Layout Extraction)
# =============================================================================


def detect_contours(
    path,
    contrast_adjustment=1.0,
    detect_dark_on_light=True,
    max_image_dimension=2048,
):
    """
    Detect contours in an image using VNDetectContoursRequest.

    This traces UI regions, borders, cards, panels, icons, dividers - shapes
    that rectangle detection misses. Contours perfectly trace UI boundaries.

    Args:
        path: Path to the input image
        contrast_adjustment: Contrast boost (0.0-3.0, default 1.0)
        detect_dark_on_light: True for dark UI on light bg, False for inverse
        max_image_dimension: Max dimension for processing (default 2048)

    Returns:
        List of contour groups, each with:
        - 'path': list of normalized (x, y) points
        - 'bbox': bounding box (x, y, w, h) normalized
        - 'point_count': number of points in contour
        - 'child_contours': nested contours inside this one
    """
    path = os.path.abspath(path)
    cgimage = load_image_as_cgimage(path)

    if cgimage is None:
        raise RuntimeError("Failed to produce CGImage")

    request = Vision.VNDetectContoursRequest.alloc().init()

    # Configure contour detection parameters
    request.setContrastAdjustment_(contrast_adjustment)
    request.setDetectsDarkOnLight_(detect_dark_on_light)
    request.setMaximumImageDimension_(max_image_dimension)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cgimage, None
    )

    _perform_request(handler, request)

    results = request.results() or []
    if not results:
        return []

    def extract_contour_data(contour, depth=0):
        """Recursively extract contour and child contours."""
        # Get the path points
        point_count = contour.pointCount()
        points = []

        if point_count > 0:
            # Get normalized path
            path_obj = contour.normalizedPath()
            if path_obj:
                # Extract points from the CGPath
                points = _extract_cgpath_points(path_obj)

        # Calculate bounding box from points
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        else:
            bbox = (0, 0, 0, 0)

        # Get child contours
        child_count = contour.childContourCount()
        children = []
        for i in range(child_count):
            child = contour.childContourAtIndex_(i)
            if child:
                children.append(extract_contour_data(child, depth + 1))

        return {
            "path": points,
            "bbox": bbox,
            "point_count": point_count,
            "depth": depth,
            "child_contours": children,
        }

    # The result contains a top-level contour observation
    observation = results[0]
    top_level_contours = []

    # Get the top-level contour count
    contour_count = observation.contourCount()
    for i in range(contour_count):
        contour = observation.contourAtIndex_error_(i, None)
        if contour:
            top_level_contours.append(extract_contour_data(contour))

    return top_level_contours


def _extract_cgpath_points(cgpath):
    """Extract points from a CGPath object."""
    points = []

    # Use the path's bounding box and element enumeration
    try:
        # Get path elements via applier function
        # Since pyobjc doesn't easily expose CGPathApply, we'll use
        # a workaround: convert to bezier points via the path's methods
        element_count = cgpath.elementCount() if hasattr(cgpath, "elementCount") else 0

        if element_count > 0:
            for i in range(element_count):
                pt_array = cgpath.elementAtIndex_associatedPoints_(i, None)
                if pt_array:
                    elem_type, pts = pt_array
                    for pt in pts:
                        if hasattr(pt, "x"):
                            points.append((float(pt.x), float(pt.y)))
        else:
            # Fallback: try to get points from NSBezierPath representation
            bounds = cgpath.boundingBox() if hasattr(cgpath, "boundingBox") else None
            if bounds:
                x, y = bounds.origin.x, bounds.origin.y
                w, h = bounds.size.width, bounds.size.height
                # Return corner points as approximation
                points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    except Exception:
        pass

    return points


def flatten_contours(contours, min_area=0.0001):
    """
    Flatten nested contour hierarchy into a list of shapes.

    Args:
        contours: Nested contour data from detect_contours
        min_area: Minimum area (normalized) to include

    Returns:
        List of contours sorted by area (largest first)
    """
    flat = []

    def recurse(contour_list):
        for c in contour_list:
            area = c["bbox"][2] * c["bbox"][3]
            if area >= min_area:
                flat.append(c)
            if c["child_contours"]:
                recurse(c["child_contours"])

    recurse(contours)
    flat.sort(key=lambda c: c["bbox"][2] * c["bbox"][3], reverse=True)
    return flat


def save_contours_annotated(
    path,
    output_path=None,
    contrast_adjustment=1.0,
    detect_dark_on_light=True,
    max_image_dimension=2048,
    min_area=0.0005,
    max_contours=50,
    draw_paths=True,
    draw_boxes=True,
):
    """
    Detect contours and draw them on the image.

    Args:
        path: Path to the input image
        output_path: Path to save output (default: adds _contours suffix)
        contrast_adjustment: Contrast boost for detection
        detect_dark_on_light: Detection mode
        max_image_dimension: Max processing dimension
        min_area: Minimum contour area to draw (normalized)
        max_contours: Maximum number of contours to draw
        draw_paths: Draw the actual contour paths
        draw_boxes: Draw bounding boxes around contours

    Returns:
        Path to the saved annotated image
    """
    contours = detect_contours(
        path,
        contrast_adjustment=contrast_adjustment,
        detect_dark_on_light=detect_dark_on_light,
        max_image_dimension=max_image_dimension,
    )

    # Flatten and filter contours
    flat_contours = flatten_contours(contours, min_area=min_area)[:max_contours]

    image_path = Path(path)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    stroke_width = max(1, int(min(image.size) * 0.002))

    # Color palette
    colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEAA7",
        "#DDA0DD",
        "#98D8C8",
        "#F7DC6F",
        "#BB8FCE",
        "#85C1E9",
        "#F8B500",
        "#00CED1",
        "#FF69B4",
        "#32CD32",
        "#FF4500",
    ]

    for i, contour in enumerate(flat_contours):
        color = colors[i % len(colors)]

        # Draw bounding box
        if draw_boxes:
            x, y, w, h = contour["bbox"]
            pixel_box = _normalized_to_pixel_box(
                (x, y, w, h), image.width, image.height
            )
            draw.rectangle(pixel_box, outline=color, width=stroke_width)

            # Label
            label = f"#{i + 1}"
            text_w, text_h = _measure_text(draw, label, font)
            margin = 1
            x0, y0, _, _ = pixel_box
            bg = (
                x0,
                max(0, y0 - text_h - 2),
                x0 + text_w + 4,
                max(0, y0 - text_h - 2) + text_h + 2,
            )
            draw.rectangle(bg, fill=color)
            draw.text((bg[0] + 2, bg[1]), label, fill="black", font=font)

        # Draw contour path
        if draw_paths and contour["path"] and len(contour["path"]) > 1:
            # Convert normalized points to pixel coordinates
            pixel_points = []
            for nx, ny in contour["path"]:
                px = nx * image.width
                py = (1.0 - ny) * image.height  # Flip Y coordinate
                pixel_points.append((px, py))

            # Draw path as connected lines
            if len(pixel_points) >= 2:
                draw.line(pixel_points, fill=color, width=stroke_width)

    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_contours.png")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    return str(output_path)


# =============================================================================
# Feature Print & Tile Clustering (Icon/Element Detection)
# =============================================================================


def generate_feature_print(path):
    """
    Generate a feature print vector for the image using Vision framework.

    Returns a 768-dimensional feature vector that represents the image's
    visual content. Useful for image fingerprinting and similarity.

    Returns:
        dict with:
        - 'element_count': number of elements in feature vector (768)
        - 'element_type': data type identifier
        - 'data': numpy array of feature values (float32)
    """
    path = os.path.abspath(path)
    cgimage = load_image_as_cgimage(path)

    if cgimage is None:
        raise RuntimeError("Failed to produce CGImage")

    request = Vision.VNGenerateImageFeaturePrintRequest.alloc().init()
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cgimage, None
    )

    _perform_request(handler, request)

    results = request.results() or []
    if not results:
        raise RuntimeError("No feature print results returned")

    observation = results[0]

    element_count = None
    element_type = None

    if hasattr(observation, "elementCount"):
        element_count = observation.elementCount()
    if hasattr(observation, "elementType"):
        element_type = observation.elementType()

    # Extract the feature data
    feature_data = None
    if hasattr(observation, "data"):
        data = observation.data()
        if data:
            data_bytes = data.bytes()
            data_length = data.length()
            if data_length > 0 and element_count:
                arr = np.frombuffer(bytes(data_bytes[:data_length]), dtype=np.float32)
                feature_data = arr.copy()

    return {
        "element_count": element_count,
        "element_type": element_type,
        "data": feature_data,
    }


def generate_tile_featureprints(path, tile_size=64, stride=32, min_variance=0.01):
    """
    Generate featureprints for tiles across an image.

    This enables detection of repeated icons, thumbnails, and similar elements
    by clustering tiles with similar featureprints.

    Args:
        path: Path to the input image
        tile_size: Size of each tile in pixels (square)
        stride: Step size between tiles
        min_variance: Minimum pixel variance to include tile (filters blank areas)

    Returns:
        List of tiles, each with:
        - 'bbox': pixel coordinates (x, y, w, h)
        - 'bbox_norm': normalized coordinates
        - 'featureprint': numpy array of feature values
        - 'variance': pixel variance (complexity measure)
    """
    path = os.path.abspath(path)
    image = Image.open(path).convert("RGB")
    img_w, img_h = image.size

    tiles = []

    for y in range(0, img_h - tile_size + 1, stride):
        for x in range(0, img_w - tile_size + 1, stride):
            # Extract tile
            tile = image.crop((x, y, x + tile_size, y + tile_size))

            # Check variance to filter blank/uniform areas
            tile_array = np.array(tile).astype(np.float32) / 255.0
            variance = np.var(tile_array)

            if variance < min_variance:
                continue

            # Save tile temporarily to get featureprint
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tile.save(tmp.name)
                try:
                    fp = generate_feature_print(tmp.name)
                    if fp["data"] is not None:
                        tiles.append(
                            {
                                "bbox": (x, y, tile_size, tile_size),
                                "bbox_norm": (
                                    x / img_w,
                                    y / img_h,
                                    tile_size / img_w,
                                    tile_size / img_h,
                                ),
                                "featureprint": fp["data"],
                                "variance": float(variance),
                            }
                        )
                finally:
                    os.unlink(tmp.name)

    return tiles


def cluster_tiles(tiles, n_clusters=None, distance_threshold=0.5):
    """
    Cluster tiles by featureprint similarity using agglomerative clustering.

    Args:
        tiles: List of tiles from generate_tile_featureprints
        n_clusters: Number of clusters (None for auto via distance threshold)
        distance_threshold: Max distance for clustering (if n_clusters is None)

    Returns:
        List of clusters, each containing:
        - 'cluster_id': cluster index
        - 'tiles': list of tiles in this cluster
        - 'centroid': mean featureprint of cluster
        - 'size': number of tiles
    """
    if len(tiles) < 2:
        return [{"cluster_id": 0, "tiles": tiles, "centroid": None, "size": len(tiles)}]

    # Stack featureprints into matrix
    X = np.vstack([t["featureprint"] for t in tiles])

    # Normalize featureprints
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = X / norms

    # Simple agglomerative clustering using pairwise distances
    # Compute distance matrix
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster

    try:
        distances = pdist(X_norm, metric="cosine")
        Z = linkage(distances, method="average")

        if n_clusters is not None:
            labels = fcluster(Z, n_clusters, criterion="maxclust")
        else:
            labels = fcluster(Z, distance_threshold, criterion="distance")
    except ImportError:
        # Fallback: simple k-means-like clustering without scipy
        labels = _simple_cluster(X_norm, n_clusters or 5)

    # Group tiles by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(tiles[i])

    # Build cluster info
    result = []
    for cluster_id, cluster_tiles in clusters.items():
        centroid = np.mean([t["featureprint"] for t in cluster_tiles], axis=0)
        result.append(
            {
                "cluster_id": int(cluster_id),
                "tiles": cluster_tiles,
                "centroid": centroid,
                "size": len(cluster_tiles),
            }
        )

    # Sort by size (largest clusters first)
    result.sort(key=lambda c: c["size"], reverse=True)
    return result


def _simple_cluster(X, k):
    """Simple k-means clustering fallback without scipy."""
    n = len(X)
    if n <= k:
        return list(range(n))

    # Initialize centroids randomly
    indices = np.random.choice(n, k, replace=False)
    centroids = X[indices].copy()

    labels = np.zeros(n, dtype=int)

    for _ in range(20):  # Max iterations
        # Assign points to nearest centroid
        for i in range(n):
            distances = np.linalg.norm(centroids - X[i], axis=1)
            labels[i] = np.argmin(distances)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centroids[j] = X[mask].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels


def save_clustered_tiles_annotated(
    path,
    output_path=None,
    tile_size=64,
    stride=48,
    min_variance=0.01,
    n_clusters=None,
    distance_threshold=0.3,
    min_cluster_size=2,
):
    """
    Detect and visualize repeated visual elements via tile clustering.

    Tiles with similar featureprints are grouped and colored the same,
    revealing repeated icons, buttons, thumbnails, etc.

    Args:
        path: Path to input image
        output_path: Path to save output
        tile_size: Tile size in pixels
        stride: Step between tiles
        min_variance: Min variance to include tile
        n_clusters: Number of clusters (None for auto)
        distance_threshold: Clustering threshold
        min_cluster_size: Only show clusters with at least this many tiles

    Returns:
        Path to the saved annotated image
    """
    print(f"  Generating tile featureprints (tile={tile_size}, stride={stride})...")
    tiles = generate_tile_featureprints(
        path, tile_size=tile_size, stride=stride, min_variance=min_variance
    )
    print(f"  Found {len(tiles)} tiles with sufficient variance")

    if len(tiles) < 2:
        print("  Not enough tiles to cluster")
        return None

    print(f"  Clustering tiles...")
    clusters = cluster_tiles(
        tiles, n_clusters=n_clusters, distance_threshold=distance_threshold
    )
    print(f"  Found {len(clusters)} clusters")

    # Filter to significant clusters
    significant_clusters = [c for c in clusters if c["size"] >= min_cluster_size]
    print(f"  {len(significant_clusters)} clusters with {min_cluster_size}+ tiles")

    image_path = Path(path)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")  # RGBA for transparency
    font = ImageFont.load_default()

    # Color palette
    colors = [
        (255, 107, 107, 180),  # coral red
        (78, 205, 196, 180),  # teal
        (69, 183, 209, 180),  # sky blue
        (150, 206, 180, 180),  # sage green
        (255, 234, 167, 180),  # pale yellow
        (221, 160, 221, 180),  # plum
        (152, 216, 200, 180),  # mint
        (247, 220, 111, 180),  # gold
        (187, 143, 206, 180),  # lavender
        (133, 193, 233, 180),  # light blue
    ]

    for cluster_idx, cluster in enumerate(significant_clusters):
        color = colors[cluster_idx % len(colors)]
        solid_color = color[:3]

        for tile in cluster["tiles"]:
            x, y, w, h = tile["bbox"]

            # Draw semi-transparent fill
            draw.rectangle([x, y, x + w, y + h], fill=color, outline=solid_color)

        # Label the cluster
        if cluster["tiles"]:
            first_tile = cluster["tiles"][0]
            x, y, _, _ = first_tile["bbox"]
            label = f"C{cluster_idx + 1}({cluster['size']})"
            text_w, text_h = _measure_text(draw, label, font)
            draw.rectangle([x, y, x + text_w + 4, y + text_h + 2], fill=solid_color)
            draw.text((x + 2, y), label, fill="white", font=font)

    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_clusters.png")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    return str(output_path)


if __name__ == "__main__":
    sample_image = (
        Path(__file__).resolve().parent.parent
        / "test_images"
        / "20251021_123325_226_Spotify_Taylor_Swift_-_The_Fate_of_Ophelia.png"
    )

    print("=" * 60)
    print("Apple Vision Framework Demo")
    print("=" * 60)

    # OCR Demo
    print("\n[1] Text Recognition (OCR)")
    print("-" * 40)
    detections = recognize_text_with_boxes(sample_image)
    if detections:
        print(f"Detected {len(detections)} text regions:")
        for det in detections[:5]:
            text_preview = (
                det["text"][:50] + "..." if len(det["text"]) > 50 else det["text"]
            )
            print(f"  - '{text_preview}' @ {det['bbox']}")
        if len(detections) > 5:
            print(f"  ... and {len(detections) - 5} more")
    else:
        print("No text detected")

    annotated_path = save_annotated_image(sample_image)
    print(f"OCR annotated image: {annotated_path}")

    # Contour Detection Demo
    print("\n[2] Contour Detection (UI Layout)")
    print("-" * 40)
    try:
        contours = detect_contours(
            sample_image,
            contrast_adjustment=1.5,
            detect_dark_on_light=False,  # Spotify is dark mode
        )
        flat = flatten_contours(contours, min_area=0.001)
        print(f"Detected {len(flat)} contours (filtered by min area)")
        for i, c in enumerate(flat[:5]):
            x, y, w, h = c["bbox"]
            print(
                f"  #{i + 1}: {w:.3f}x{h:.3f} @ ({x:.3f}, {y:.3f}) pts={c['point_count']}"
            )
        if len(flat) > 5:
            print(f"  ... and {len(flat) - 5} more")

        contour_path = save_contours_annotated(
            sample_image,
            contrast_adjustment=1.5,
            detect_dark_on_light=False,
            min_area=0.001,
            max_contours=30,
        )
        print(f"Contour annotated image: {contour_path}")
    except Exception as e:
        import traceback

        print(f"Error detecting contours: {e}")
        traceback.print_exc()

    # Feature Print Demo
    print("\n[3] Feature Print (Image Fingerprint)")
    print("-" * 40)
    try:
        feature_info = generate_feature_print(sample_image)
        print(f"Feature vector dimensions: {feature_info['element_count']}")
        print(f"Element type: {feature_info['element_type']}")
        if feature_info.get("data") is not None:
            print(f"Feature data shape: {feature_info['data'].shape}")
            print(f"Sample values: {feature_info['data'][:5]}...")
    except Exception as e:
        print(f"Error generating feature print: {e}")

    # Tile Clustering Demo
    print("\n[4] Tile Clustering (Repeated Element Detection)")
    print("-" * 40)
    try:
        cluster_path = save_clustered_tiles_annotated(
            sample_image,
            tile_size=48,
            stride=32,
            min_variance=0.005,
            distance_threshold=0.25,
            min_cluster_size=2,
        )
        if cluster_path:
            print(f"Cluster annotated image: {cluster_path}")
    except Exception as e:
        import traceback

        print(f"Error clustering tiles: {e}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
