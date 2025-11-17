"""
Compare V1 (dilation-based) vs V2 (flood-fill) image detection methods.

Usage:
    python compare_detection_methods.py path/to/image.png
"""

import sys
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# Import both versions
from image_detection import detect_images as detect_v1
from image_detection_v2 import detect_images as detect_v2


def load_image(image_path):
    """Load image and convert to RGB numpy array."""
    pil_img = Image.open(image_path).convert('RGB')
    return np.array(pil_img)


def draw_boxes_comparison(image_rgb, boxes_v1, boxes_v2):
    """
    Create side-by-side comparison with boxes drawn.
    
    V1 boxes in red, V2 boxes in green.
    """
    height, width = image_rgb.shape[:2]
    
    # Create comparison image (side by side)
    comparison = np.zeros((height, width * 2 + 20, 3), dtype=np.uint8)
    comparison[:, :width] = image_rgb
    comparison[:, width+20:] = image_rgb
    
    # Draw V1 boxes in red on left
    for x, y, w, h in boxes_v1:
        cv2.rectangle(comparison, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(comparison, "V1", (x + 5, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw V2 boxes in green on right
    offset = width + 20
    for x, y, w, h in boxes_v2:
        cv2.rectangle(comparison, (offset + x, y), (offset + x + w, y + h), (0, 255, 0), 2)
        cv2.putText(comparison, "V2", (offset + x + 5, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add labels
    cv2.putText(comparison, "V1: Dilation-Based", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    cv2.putText(comparison, "V2: Flood-Fill", (offset + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Add counts
    cv2.putText(comparison, f"Found: {len(boxes_v1)}", (10, height - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(comparison, f"Found: {len(boxes_v2)}", (offset + 10, height - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return comparison


def analyze_differences(boxes_v1, boxes_v2):
    """Analyze and report differences between detection methods."""
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"\nDetection Counts:")
    print(f"  V1 (Dilation):  {len(boxes_v1)} images")
    print(f"  V2 (Flood-Fill): {len(boxes_v2)} images")
    
    if len(boxes_v1) == 0 and len(boxes_v2) == 0:
        print("\n⚠ Neither method found any images")
        return
    
    # Compare box sizes
    if boxes_v1:
        v1_areas = [w * h for x, y, w, h in boxes_v1]
        print(f"\nV1 Box Areas:")
        print(f"  Total: {sum(v1_areas):,} px²")
        print(f"  Average: {np.mean(v1_areas):,.0f} px²")
        print(f"  Range: {min(v1_areas):,} - {max(v1_areas):,} px²")
    
    if boxes_v2:
        v2_areas = [w * h for x, y, w, h in boxes_v2]
        print(f"\nV2 Box Areas:")
        print(f"  Total: {sum(v2_areas):,} px²")
        print(f"  Average: {np.mean(v2_areas):,.0f} px²")
        print(f"  Range: {min(v2_areas):,} - {max(v2_areas):,} px²")
    
    # Check for matches (boxes in similar locations)
    def boxes_match(box1, box2, threshold=0.5):
        """Check if two boxes overlap significantly."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        iou = intersection / union if union > 0 else 0
        return iou > threshold
    
    # Find matches
    matched_v1 = set()
    matched_v2 = set()
    
    for i, b1 in enumerate(boxes_v1):
        for j, b2 in enumerate(boxes_v2):
            if boxes_match(b1, b2):
                matched_v1.add(i)
                matched_v2.add(j)
    
    num_matches = len(matched_v1)
    v1_only = len(boxes_v1) - num_matches
    v2_only = len(boxes_v2) - num_matches
    
    print(f"\nOverlap Analysis:")
    print(f"  Both detected: {num_matches} images")
    print(f"  V1 only: {v1_only} images")
    print(f"  V2 only: {v2_only} images")
    
    # Recommendations
    print(f"\n" + "-"*60)
    print("RECOMMENDATIONS:")
    print("-"*60)
    
    if len(boxes_v2) == 0 and len(boxes_v1) > 0:
        print("  → V2 found nothing. Try increasing color_tolerance (e.g., 40-50)")
    elif len(boxes_v1) > len(boxes_v2) * 2:
        print("  → V1 found many more boxes (possible false positives)")
        print("  → V2 is more conservative - likely more accurate")
    elif len(boxes_v2) > len(boxes_v1) * 2:
        print("  → V2 found many more boxes")
        print("  → Check if V1 missed real images or V2 has false positives")
    elif num_matches == max(len(boxes_v1), len(boxes_v2)):
        print("  → Both methods found similar results!")
    else:
        print("  → Results differ significantly - manual inspection recommended")
    
    if v1_areas and v2_areas:
        v1_avg = np.mean(v1_areas)
        v2_avg = np.mean(v2_areas)
        if v1_avg > v2_avg * 1.5:
            print("  → V1 boxes are much larger (likely over-growing)")
            print("  → V2 boundaries are tighter (more accurate)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_detection_methods.py <image_path>")
        print("\nExample:")
        print("  python compare_detection_methods.py test_images/20251021_123325_226_Spotify_Taylor_Swift_-_The_Fate_of_Ophelia.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"Loading image: {image_path}")
    image_rgb = load_image(image_path)
    print(f"Image size: {image_rgb.shape[1]}x{image_rgb.shape[0]}")
    
    # No OCR boxes for comparison (would need PaddleOCR)
    ocr_boxes = []
    
    print("\n" + "="*60)
    print("Running V1 (Dilation-Based)...")
    print("="*60)
    result_v1 = detect_v1(image_rgb, ocr_boxes)
    boxes_v1 = result_v1['image_boxes']
    
    print("\n" + "="*60)
    print("Running V2 (Flood-Fill)...")
    print("="*60)
    result_v2 = detect_v2(image_rgb, ocr_boxes)
    boxes_v2 = result_v2['image_boxes']
    
    # Analyze
    analyze_differences(boxes_v1, boxes_v2)
    
    # Create comparison visualization
    comparison = draw_boxes_comparison(image_rgb, boxes_v1, boxes_v2)
    
    # Save comparison
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_comparison.png"
    Image.fromarray(comparison).save(output_path)
    print(f"\n✓ Comparison saved to: {output_path}")
    
    # Optionally save intermediate steps
    output_dir = Path(image_path).parent / f"{Path(image_path).stem}_comparison_steps"
    output_dir.mkdir(exist_ok=True)
    
    # Save V1 steps
    v1_steps = result_v1['intermediate_steps']
    for key in ['edges_sparse', 'edges_connected', 'grown_boxes']:
        if key in v1_steps:
            data = v1_steps[key]
            if isinstance(data, np.ndarray):
                Image.fromarray(data).save(output_dir / f"v1_{key}.png")
    
    # Save V2 steps
    v2_steps = result_v2['intermediate_steps']
    for key in ['edges_sparse', 'filled_regions', 'grown_boxes']:
        if key in v2_steps:
            data = v2_steps[key]
            if isinstance(data, np.ndarray):
                Image.fromarray(data).save(output_dir / f"v2_{key}.png")
    
    print(f"✓ Intermediate steps saved to: {output_dir}/")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()


