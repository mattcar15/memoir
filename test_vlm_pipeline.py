#!/usr/bin/env python3
"""
Test script to verify the VLM-based pipeline works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from memoir.processing.pipeline.main import ImagePipeline
from PIL import Image


def main():
    print("=" * 70)
    print("Testing VLM-based Pipeline")
    print("=" * 70)
    print()

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = ImagePipeline(
        similarity_threshold=0.7,
        memory_window_minutes=5,
        phash_threshold=10,
        verbose=True,
    )
    print("✅ Pipeline initialized")
    print()

    # Check if there are test images
    test_image_dir = Path(__file__).parent / "testing" / "marked_snapshots"
    if not test_image_dir.exists():
        print(f"❌ Test image directory not found: {test_image_dir}")
        return 1

    # Get first test image
    test_images = list(test_image_dir.glob("*.png"))
    if not test_images:
        print(f"❌ No test images found in {test_image_dir}")
        return 1

    test_image_path = test_images[0]
    print(f"Using test image: {test_image_path.name}")
    print()

    # Load and process image
    try:
        image = Image.open(test_image_path)
        print(f"✅ Loaded image: {image.size[0]}x{image.size[1]}")
        print()

        # Process image
        result = pipeline.process_image(image, save_images=False)

        if result:
            print()
            print("=" * 70)
            print("✅ Processing successful!")
            print("=" * 70)
            print(f"Memory ID: {result['memory_id']}")
            print(f"Processing Method: {result['processing_method']}")
            print(f"Summary: {result['summary'][:100]}...")
            print(f"Total Time: {result['total_processing_time_seconds']:.2f}s")
            if result.get('vlm_stats'):
                print(f"VLM Stats: {result['vlm_stats']}")
            print("=" * 70)
            return 0
        else:
            print()
            print("❌ Processing failed")
            return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


