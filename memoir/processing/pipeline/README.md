# Image Processing Pipeline

This module implements an intelligent screenshot processing pipeline with memory consolidation.

## Overview

The pipeline processes screenshots through several stages:

1. **Duplicate Detection**: Uses perceptual hashing (pHash) to detect duplicate images within a 5-minute window
2. **Image Downscaling**: Scales images for efficient processing
3. **Text Extraction**: Uses PaddleOCR to extract text from screenshots
4. **Processing Decision**: 
   - If text >= 100 characters: Use OCR result directly
   - If text < 100 characters: Use LLM (Ollama) for image analysis
5. **Memory Consolidation**: Uses cosine similarity (default 0.7) to merge related screenshots into memories

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- `paddleocr` - For text extraction
- `paddlepaddle` - Backend for PaddleOCR
- `imagehash` - For perceptual hashing

## Usage

### Benchmark Mode

Process existing screenshots and get detailed statistics:

```bash
# Basic usage - process all screenshots in logs/screenshots/
python -m memoir benchmark

# Process only first 5 images
python -m memoir benchmark --max-images 5

# Skip LLM processing (faster, OCR only)
python -m memoir benchmark --skip-llm

# Custom similarity threshold
python -m memoir benchmark --similarity-threshold 0.8

# Don't save processed images
python -m memoir benchmark --no-save

# Full example with all options
python -m memoir benchmark \
  --screenshots-dir logs/screenshots \
  --max-images 10 \
  --skip-llm \
  --output-dir logs/benchmark_output \
  --similarity-threshold 0.75 \
  --embedding-model embeddinggemma \
  --model gemma3:4b
```

### Programmatic Usage

```python
from memoir.pipeline import ImagePipeline
from PIL import Image

# Initialize pipeline
pipeline = ImagePipeline(
    similarity_threshold=0.7,  # Cosine similarity threshold
    memory_window_minutes=5,   # Time window for active memories
    phash_threshold=10,        # pHash difference threshold
    ocr_text_threshold=100,    # Min chars for OCR-only processing
    enable_llm=True,          # Use LLM for image-heavy content
    embedding_model="embeddinggemma",
    ollama_model="gemma3:4b",
)

# Process an image
image = Image.open("screenshot.png")
result = pipeline.process_image(
    image,
    output_dir=Path("output"),
    save_images=True,
)

# Get statistics
stats = pipeline.get_statistics()
print(f"Memories created: {stats['memories_created']}")
print(f"Images consolidated: {stats['images_consolidated']}")

# Print formatted statistics
pipeline.print_statistics()
```

## Pipeline Statistics

The benchmark command provides detailed statistics:

- **Total images processed**: Number of images loaded
- **Duplicates skipped**: Images filtered by pHash
- **Images with sufficient OCR**: Screenshots where OCR text was >= 100 chars
- **Images needing LLM**: Screenshots processed with Ollama
- **Memories created**: Number of distinct memory sessions
- **Images consolidated**: Screenshots merged into existing memories
- **Images per memory distribution**: Breakdown of how many images per memory

## Memory Consolidation

Memories are consolidated based on:

1. **Time Window**: Only memories active in the last 5 minutes are candidates
2. **Semantic Similarity**: Cosine similarity between embeddings must exceed threshold (default 0.7)
3. **Image Storage**: 
   - First image in a memory: Saved at original resolution
   - Subsequent images: Saved at downscaled resolution (balanced tier)

## Configuration

Key parameters can be customized:

- `similarity_threshold`: 0.0-1.0, default 0.7 (higher = stricter matching)
- `memory_window_minutes`: Default 5 minutes
- `phash_threshold`: Default 10 (lower = stricter duplicate detection)
- `ocr_text_threshold`: Default 100 characters

## Example Output

```
🔬 BENCHMARK MODE
================================================================================
Screenshots directory: logs/screenshots
Output directory: logs/benchmark_output
Max images: 5
LLM processing: Enabled
Similarity threshold: 0.7
Save images: True
================================================================================

📁 Found 11 images
🔢 Processing first 5 images

🔧 Initializing PaddleOCR...
✅ PaddleOCR initialized

================================================================================
Processing 1/5: 2025-10-10_02-19-05.png
================================================================================

📸 Processing image #1...
🔍 Checking for duplicates (pHash)...
✅ Image is unique
🔽 Downscaling image for processing...
✅ Downscaled to 1024x768
📝 Extracting text with OCR...
✅ Extracted 523 chars from 15 lines
✅ Using OCR text (>= 100 chars)
📝 Summary: User is viewing/working with: good temperature for ollama gemma3:4b...
🔄 Creating embedding...
✅ Created embedding (768 dimensions) in 0.123s
🔍 Searching for matching memory...
🆕 Creating new memory
💾 Saving original resolution image...
✅ Saved: logs/benchmark_output/original_2025-10-10_14-23-45-123456.png
✅ Processing complete in 2.45s

[... more processing ...]

🏁 BENCHMARK COMPLETE
================================================================================
Total time: 12.34s
Average time per image: 2.47s

📊 PIPELINE STATISTICS
================================================================================
Total images processed: 5
Duplicates skipped: 0
Images with sufficient OCR: 4
Images needing LLM: 1

Memories created: 3
Images consolidated into existing memories: 2
Total memories: 3
Active memories: 3

Images per memory distribution:
  1 images: 1 memories
  2 images: 2 memories
================================================================================
```

## Implementation Details

### pHash Duplicate Detection

The pipeline uses perceptual hashing to detect near-duplicate images. Images with a pHash difference <= 10 within a 5-minute window are considered duplicates and skipped.

### OCR Processing

PaddleOCR is used with:
- Angle classification enabled
- English language
- CPU processing (GPU optional)

### LLM Fallback

When OCR extracts < 100 characters, the image is processed with Ollama using the configured model (default: gemma3:4b).

### Memory Consolidation

Embeddings are created using the configured embedding model (default: embeddinggemma). Cosine similarity is calculated between the new embedding and all active memory embeddings. If similarity >= threshold, the image is added to the existing memory.

## Troubleshooting

### PaddleOCR Installation Issues

If you encounter issues with PaddleOCR:

```bash
# Try installing with specific versions
pip install paddlepaddle==2.5.0
pip install paddleocr==2.7.0
```

### Memory Issues

If processing large images causes memory issues:
- Use `--skip-llm` to disable LLM processing
- Process fewer images at a time with `--max-images`
- Images are automatically downscaled for processing

### Performance

To speed up benchmarking:
- Use `--skip-llm` flag
- Use `--no-save` to skip saving images
- Process fewer images with `--max-images`




