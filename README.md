# Screenshot Memory System

A Python application that captures screenshots at regular intervals, processes them through Ollama's vision-capable LLM using LangChain, and logs activity summaries.

## Features

- 📸 Automated screenshot capture at configurable intervals
- 🤖 AI-powered activity summarization using Ollama (gemma3:4b)
- 📊 Three resolution tiers: efficient (800x600), balanced (1024x768), detailed (1920x1080) - **detailed recommended by default**
- 💾 Optional screenshot image saving
- 📝 Individual JSON log files with timestamps
- ⚙️ Fully configurable via command-line arguments
- 📈 **Performance metrics tracking**: processing times, image sizes, token counts
- 🔬 **Comparison mode**: test all three resolution tiers side-by-side
- 🔥 **Model warmup**: keeps Ollama model loaded in memory for faster processing

## Prerequisites

1. **Python 3.8+** with virtual environment activated
2. **Ollama** installed and running locally
   - Install from: https://ollama.ai
   - Pull the model: `ollama pull gemma3:4b`

## Installation

1. Activate your virtual environment (if not already activated):
   ```bash
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run with default settings (detailed resolution, 1-minute intervals):
```bash
python main.py
```

**Note:** The system defaults to "detailed" resolution (1920x1080) because processing time is dominated by model inference, not image size. Higher resolution provides better quality with no meaningful performance penalty.

### Command-Line Options

```bash
python main.py [OPTIONS]
```

**Options:**

- `--resolution-tier {efficient,balanced,detailed}`
  - Choose screenshot resolution tier (default: detailed)
  - **efficient**: 800x600 - smaller files
  - **balanced**: 1024x768 - medium files
  - **detailed**: 1920x1080 - best quality (recommended, no performance penalty)

- `--save-images`
  - Save screenshot images alongside logs
  - Images saved to `logs/screenshots/`

- `--interval MINUTES`
  - Minutes between captures (default: 1)

- `--model MODEL_NAME`
  - Ollama model to use (default: gemma3:4b)

- `--run-once`
  - Run a single capture and exit (useful for testing)

- `--compare-tiers`
  - Compare all three resolution tiers with performance metrics
  - **Must be used with `--run-once`**
  - Generates a detailed comparison report

- `--no-warmup`
  - Skip model warmup at startup (not recommended)
  - By default, the system warms up the model on startup for faster processing

### Examples

**Captures every 5 minutes with image saving:**
```bash
python main.py --interval 5 --save-images
```

**Quick captures every 30 seconds:**
```bash
python main.py --interval 0.5
```

**Test with a single capture:**
```bash
python main.py --run-once
```

**Use a different Ollama model:**
```bash
python main.py --model llava:7b
```

**Compare all resolution tiers:**
```bash
python main.py --run-once --compare-tiers
```

**Compare tiers and save all screenshots:**
```bash
python main.py --run-once --compare-tiers --save-images
```

## Output Structure

```
memoir/
├── logs/
│   ├── screenshots/           # (if --save-images enabled)
│   │   └── 2024-10-10_14-30-00.png
│   └── 2024-10-10_14-30-00.json  # Individual log files
├── main.py
├── requirements.txt
└── README.md
```

### Log File Format

Each log entry is saved as an individual JSON file with comprehensive statistics:

```json
{
  "timestamp": "2024-10-10T14:30:00.123456",
  "summary": "The user is currently working in a code editor...",
  "resolution_tier": "balanced",
  "screenshot_path": "logs/screenshots/2024-10-10_14-30-00.png",
  "stats": {
    "original_resolution": "2560x1440",
    "scaled_resolution": "1024x576",
    "capture_time_seconds": 0.123,
    "resize_time_seconds": 0.045,
    "image_size_bytes": 245678,
    "image_size_kb": 239.92,
    "ollama_processing_time_seconds": 3.456,
    "prompt_text": "Summarize the user's current activity",
    "prompt_length_chars": 35,
    "image_base64_size_bytes": 327570,
    "image_base64_size_kb": 319.89,
    "response_length_chars": 156,
    "response_word_count": 28,
    "prompt_eval_token_count": 1245,
    "response_token_count": 42,
    "screenshot_save_time_seconds": 0.067,
    "total_processing_time_seconds": 3.691
  }
}
```

## Tips

- **First Run**: Test with `--run-once` to verify Ollama connection
- **Benchmarking**: Use `--run-once --compare-tiers` to see which resolution tier works best for your setup
- **Performance Monitoring**: Check the `stats` field in log files to track processing times
- **Model Status**: Run `ollama ps` in terminal to verify the model is loaded and stays "hot"
- **Warmup**: The system automatically warms up the model at startup (takes ~1-2s) for faster subsequent processing
- **Keep Alive**: Model stays loaded for 1 hour after last use, avoiding cold starts
- **Privacy**: Logs contain AI summaries of your screen - store securely
- **Resolution**: Always use `detailed` (default) - processing time is dominated by model inference, not image size
- **Disk Space**: Enable `--save-images` only if you need visual records (images can be large)
- **Graceful Shutdown**: Press `Ctrl+C` to stop cleanly

## Performance Metrics

The system tracks detailed performance metrics for each capture:

- **Capture metrics**: Original/scaled resolutions, image sizes, capture/resize times
- **Processing metrics**: Ollama processing time, base64 encoding size
- **Response metrics**: Character/word counts, token counts (when available)
- **Total time**: End-to-end processing time

Use `--compare-tiers` with `--run-once` to get a side-by-side comparison of all three resolution tiers. Note: processing time is dominated by model inference, so all tiers have similar performance - detailed (1920x1080) is recommended for best quality.

### Model Warmup & Keep-Alive

The system automatically warms up the Ollama model at startup and configures it to stay loaded in memory:

- **Warmup**: Sends a minimal prompt at startup to pre-load the model (~1-2 seconds)
- **Keep-Alive**: Configured to keep model loaded for 1 hour after last use
- **Benefits**: Subsequent captures are much faster (no cold start delays)
- **Verification**: Run `ollama ps` to see loaded models and their status

This optimization can reduce processing time by 50-80% after the initial warmup, especially noticeable with larger models.

## Troubleshooting

**"Error processing with Ollama"**
- Ensure Ollama is running: `ollama serve`
- Verify model is available: `ollama list`
- Pull the model if needed: `ollama pull gemma3:4b`
- Check if model is loaded: `ollama ps` (should show model with "loaded" status)

**"Error capturing screenshot"**
- On macOS, you may need to grant Screen Recording permissions
- Go to: System Preferences → Security & Privacy → Privacy → Screen Recording

**Import errors**
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

## License

MIT


# memoir
