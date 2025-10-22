# Getting Started with the Image Processing Pipeline

## Step 1: Install Dependencies

Install the new dependencies for OCR and image hashing:

```bash
cd /Users/mattcarroll/code/memoir
source venv/bin/activate  # If using venv
pip install -r requirements.txt
```

This will install:
- PaddleOCR (text extraction)
- PaddlePaddle (OCR backend)
- imagehash (duplicate detection)

## Step 2: Test the Pipeline

Run a quick test on your existing screenshots (OCR only, fast):

```bash
python -m memoir benchmark --skip-llm --max-images 3
```

This will:
- Process the first 3 images from `logs/screenshots/`
- Skip LLM processing (faster)
- Show you how the pipeline works

## Step 3: Try Full Processing

Run with LLM enabled on a few images:

```bash
python -m memoir benchmark --max-images 3
```

This includes:
- OCR text extraction
- LLM processing for image-heavy screenshots
- Memory consolidation

## Step 4: Process All Screenshots

Once you're comfortable, process all your screenshots:

```bash
python -m memoir benchmark --skip-llm
```

Or with LLM:

```bash
python -m memoir benchmark
```

## What to Expect

### First Run Output

```
🔬 BENCHMARK MODE
================================================================================
Screenshots directory: logs/screenshots
Output directory: logs/benchmark_output
Max images: 3
LLM processing: Disabled
Similarity threshold: 0.7
Save images: True
================================================================================

📁 Found 11 images
🔢 Processing first 3 images

🔧 Initializing PaddleOCR...
✅ PaddleOCR initialized

================================================================================
Processing 1/3: 2025-10-10_02-19-05.png
================================================================================

📸 Processing image #1...
🔍 Checking for duplicates (pHash)...
✅ Image is unique
🔽 Downscaling image for processing...
✅ Downscaled to 1024x768
📝 Extracting text with OCR...
✅ Extracted 523 chars from 15 lines
✅ Using OCR text (>= 100 chars)
📝 Summary: User is viewing/working with: good temperature for ollama...
🔄 Creating embedding...
✅ Created embedding (768 dimensions) in 0.123s
🔍 Searching for matching memory...
🆕 Creating new memory
💾 Saving original resolution image...
✅ Saved: logs/benchmark_output/original_2025-10-10_14-23-45.png
✅ Processing complete in 2.45s

[... more processing ...]

📊 PIPELINE STATISTICS
================================================================================
Total images processed: 3
Duplicates skipped: 0
Images with sufficient OCR: 2
Images needing LLM: 1

Memories created: 2
Images consolidated into existing memories: 1
Total memories: 2
Active memories: 2

Images per memory distribution:
  1 images: 1 memories
  2 images: 1 memories
================================================================================
```

## Understanding the Results

### Key Metrics

1. **Duplicates skipped**: Near-identical images filtered out
2. **OCR vs LLM**: How many images had enough text vs needed vision AI
3. **Memories created**: Distinct activity sessions detected
4. **Images consolidated**: Screenshots grouped into existing memories
5. **Images per memory**: Shows how screenshots clustered together

### What This Tells You

- High OCR count = Text-heavy screenshots (good for fast processing)
- High LLM count = Image-heavy content (slower but more accurate)
- Few memories with many images = Good consolidation
- Many memories with 1 image = Diverse activities

## Tuning the Pipeline

### For Stricter Memory Separation

```bash
python -m memoir benchmark --similarity-threshold 0.8 --max-images 5
```

Higher threshold = more distinct memories (less consolidation)

### For Looser Memory Grouping

```bash
python -m memoir benchmark --similarity-threshold 0.6 --max-images 5
```

Lower threshold = fewer memories (more consolidation)

## Common Use Cases

### Quick Testing (Fast)
```bash
python -m memoir benchmark --skip-llm --max-images 5 --no-save
```

### Production Analysis (Accurate)
```bash
python -m memoir benchmark
```

### Custom Directory
```bash
python -m memoir benchmark --screenshots-dir /path/to/images
```

### Adjust Memory Matching
```bash
python -m memoir benchmark --similarity-threshold 0.75
```

## Troubleshooting

### "No module named 'paddleocr'"

Install dependencies:
```bash
pip install paddleocr paddlepaddle imagehash
```

### "No images found"

Check that screenshots exist:
```bash
ls logs/screenshots/
```

Or specify a different directory:
```bash
python -m memoir benchmark --screenshots-dir /path/to/screenshots
```

### Very Slow Processing

Use `--skip-llm` to disable LLM:
```bash
python -m memoir benchmark --skip-llm
```

### PaddleOCR Initialization Failed

Make sure you have enough disk space (PaddleOCR downloads models on first run):
```bash
df -h
```

## Next Steps

1. Review the statistics to understand your screenshot patterns
2. Adjust similarity threshold based on your needs
3. Consider integrating the pipeline into the main capture loop
4. See `PIPELINE_USAGE.md` for detailed documentation
5. See `memoir/pipeline/README.md` for technical details

## Support

- Check `PIPELINE_USAGE.md` for detailed usage
- Check `memoir/pipeline/README.md` for API documentation
- Check `memoir/pipeline/IMPLEMENTATION_SUMMARY.md` for technical details
- Review `memoir/pipeline/PROCESSING.md` for the original design doc

## Quick Reference

```bash
# Fast test (3 images, OCR only)
python -m memoir benchmark --skip-llm --max-images 3

# Full test (5 images, OCR + LLM)
python -m memoir benchmark --max-images 5

# Process everything (OCR only)
python -m memoir benchmark --skip-llm

# Process everything (full pipeline)
python -m memoir benchmark

# Custom similarity
python -m memoir benchmark --similarity-threshold 0.8

# Don't save images
python -m memoir benchmark --no-save --skip-llm
```




