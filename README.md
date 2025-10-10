# Screenshot Memory System

A Python application that captures screenshots at regular intervals, processes them through Ollama's vision-capable LLM using LangChain, and logs activity summaries. Now with **semantic search** powered by vector embeddings!

## Features

### Core Features
- 📸 Automated screenshot capture at configurable intervals
- 🤖 AI-powered activity summarization using Ollama (gemma3:4b)
- 📊 Three resolution tiers: efficient (800x600), balanced (1024x768), detailed (1920x1080) - **detailed recommended by default**
- 💾 Optional screenshot image saving
- 📝 Individual JSON log files with timestamps
- ⚙️ Fully configurable via command-line arguments
- 📈 **Performance metrics tracking**: processing times, image sizes, token counts
- 🔬 **Comparison mode**: test all three resolution tiers side-by-side
- 🔥 **Model warmup**: keeps Ollama model loaded in memory for faster processing

### NEW Features 🎉
- 🧠 **Vector embeddings**: Convert summaries to embeddings using [embeddinggemma](https://ollama.com/library/embeddinggemma)
- 🔍 **Semantic search**: Find memories by meaning, not just keywords
- 🗄️ **Vector database**: ChromaDB for efficient similarity search
- ♻️ **Live reload**: Auto-restart on code changes during development

## Prerequisites

1. **Python 3.8+** with virtual environment activated
2. **Ollama** installed and running locally
   - Install from: https://ollama.ai
   - Pull the models: 
     - `ollama pull gemma3:4b` (for vision/summarization)
     - `ollama pull embeddinggemma` (for vector embeddings - optional)

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

**Capture screenshots (with vectorization by default):**
```bash
python main.py
# Vectorization is ON by default - your memories are automatically searchable!
```

**Capture with image saving:**
```bash
python main.py --save-images
```

**Disable vectorization (if you want logs only):**
```bash
python main.py --disable-vectorization
```

**Search your memories:**
```bash
python main.py search "what was I doing last night?"
python main.py search "coding projects" --results 10
```

**Note:** The system defaults to "detailed" resolution (1920x1080) because processing time is dominated by model inference, not image size. Higher resolution provides better quality with no meaningful performance penalty.

### Commands

#### Capture Mode (default)
Captures screenshots and processes them with AI. **Vectorization is enabled by default** for automatic semantic search.

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

- `--disable-vectorization` 🆕
  - **Disable** vector embeddings (vectorization is **ON by default**)
  - Use this if you only want JSON logs without semantic search
  - Search functionality will not be available if disabled

- `--embedding-model MODEL_NAME` 🆕
  - Embedding model to use (default: embeddinggemma)
  - Used for creating searchable vector embeddings

- `--live-reload` 🆕
  - Enable live reload on code changes (for development)
  - Automatically restarts when `.py` files in `memoir/` change

#### Search Mode 🆕
Search your memories using semantic similarity. Requires that you've captured memories with vectorization enabled (which is the default).

```bash
python main.py search "QUERY" [OPTIONS]
```

**Arguments:**
- `QUERY` - Your search query text

**Options:**
- `--results N` - Number of results to return (default: 5)
- `--embedding-model MODEL_NAME` - Embedding model (default: embeddinggemma)

**Note:** Search works automatically with any memories captured without `--disable-vectorization`.

### Examples

**Basic capture every 5 minutes with image saving:**
```bash
python main.py --interval 5 --save-images
```

**Capture without vectorization (logs only):**
```bash
python main.py --disable-vectorization
```

**Quick captures every 30 seconds:**
```bash
python main.py --interval 0.5
```

**Test with a single capture:**
```bash
python main.py --run-once
```

**Development mode with live reload:**
```bash
python main.py --live-reload
```

**Use a different Ollama model:**
```bash
python main.py --model llava:7b
```

**Compare all resolution tiers:**
```bash
python main.py --run-once --compare-tiers
```

**Search your memories:**
```bash
# Basic search
python main.py search "working on Python code"

# Get more results
python main.py search "what was I doing yesterday?" --results 10

# Search with custom embedding model
python main.py search "coding projects" --embedding-model embeddinggemma
```

## Output Structure

```
memoir/
├── logs/
│   ├── screenshots/           # (if --save-images enabled)
│   │   └── 2024-10-10_14-30-00.png
│   └── 2024-10-10_14-30-00.json  # Individual log files
├── vector_db/                 # ChromaDB persistence (if --enable-vectorization)
│   └── chroma.sqlite3         # Vector embeddings database
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
  "memory_id": "2024-10-10_14-30-00",
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

### General
- **First Run**: Test with `capture --run-once` to verify Ollama connection
- **Benchmarking**: Use `capture --run-once --compare-tiers` to see which resolution tier works best for your setup
- **Performance Monitoring**: Check the `stats` field in log files to track processing times
- **Model Status**: Run `ollama ps` in terminal to verify the model is loaded and stays "hot"
- **Warmup**: The system automatically warms up the model at startup (takes ~1-2s) for faster subsequent processing
- **Keep Alive**: Model stays loaded for 1 hour after last use, avoiding cold starts
- **Privacy**: Logs contain AI summaries of your screen - store securely
- **Resolution**: Always use `detailed` (default) - processing time is dominated by model inference, not image size
- **Disk Space**: Enable `--save-images` only if you need visual records (images can be large)
- **Graceful Shutdown**: Press `Ctrl+C` to stop cleanly

### Vector Search 🆕
- **Enabled by default**: All memories are automatically vectorized and searchable (disable with `--disable-vectorization`)
- **Search anytime**: Even while capturing, you can open another terminal and run `python main.py search "query"`
- **Natural queries**: Use natural language like "when was I coding?" or "meetings with clients"
- **Vector DB persistence**: Your embeddings are stored in `vector_db/` and persist across runs
- **Embedding model**: [embeddinggemma](https://ollama.com/library/embeddinggemma) is a 300M parameter model (622MB download)
- **First time setup**: Run `ollama pull embeddinggemma` before first use

### Development 🆕
- **Live reload**: Use `--live-reload` during development to auto-restart on code changes
- **Works with all features**: Combine with any other flags for rapid iteration

## Performance Metrics

The system tracks detailed performance metrics for each capture:

- **Capture metrics**: Original/scaled resolutions, image sizes, capture/resize times
- **Processing metrics**: Ollama processing time, base64 encoding size
- **Response metrics**: Character/word counts, token counts (when available)
- **Embedding metrics**: Embedding creation time, vector dimensions
- **Total time**: End-to-end processing time

Use `capture --compare-tiers --run-once` to get a side-by-side comparison of all three resolution tiers. Note: processing time is dominated by model inference, so all tiers have similar performance - detailed (1920x1080) is recommended for best quality.

### Embedding Performance
- **embeddinggemma** creates 300-dimensional vectors
- Embedding creation takes ~0.1-0.5 seconds per summary
- Vector search is extremely fast (<0.1s for most queries)
- ChromaDB stores embeddings efficiently on disk

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

**"Error creating embedding"**
- Make sure you've pulled the embedding model: `ollama pull embeddinggemma`
- Check Ollama is running: `ollama ps`
- Verify the model name matches: `ollama list | grep embedding`

**"No memories in vector store yet"**
- You need to run capture mode first to build your memory database
- Vectorization is enabled by default (unless you used `--disable-vectorization`)
- Example: `python main.py --run-once`

**"Error capturing screenshot"**
- On macOS, you may need to grant Screen Recording permissions
- Go to: System Preferences → Security & Privacy → Privacy → Screen Recording

**Import errors**
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

**ChromaDB errors**
- If you see database corruption errors, you can reset: delete the `vector_db/` directory
- Your JSON logs are safe - just re-run with `--enable-vectorization` to rebuild

## Architecture

### Components

- **`capture.py`**: Screenshot capture and image processing
- **`processor.py`**: Ollama LLM integration for summarization
- **`embeddings.py`**: Vector embedding creation using embeddinggemma 🆕
- **`vector_store.py`**: ChromaDB integration for semantic search 🆕
- **`storage.py`**: File storage and vector database integration
- **`live_reload.py`**: Development hot-reload functionality 🆕
- **`cli.py`**: Command-line interface and orchestration
- **`config.py`**: Configuration and constants

### Data Flow

1. **Capture**: Screenshot → Resize → Base64 encode
2. **Summarize**: Image + Prompt → Ollama (gemma3:4b) → Summary text
3. **Embed** (if enabled): Summary → Ollama (embeddinggemma) → Vector embedding
4. **Store**: 
   - JSON log file (always)
   - Screenshot PNG (if `--save-images`)
   - Vector embedding in ChromaDB (if `--enable-vectorization`)
5. **Search**: Query text → Vector embedding → ChromaDB similarity search → Results

### Vector Search Details

- **Embedding model**: [embeddinggemma](https://ollama.com/library/embeddinggemma) (300M params, 622MB)
- **Vector dimensions**: 300
- **Database**: ChromaDB (SQLite backend)
- **Similarity metric**: Cosine distance
- **Search speed**: Sub-100ms for most queries
- **Persistence**: All embeddings stored in `vector_db/`

## License

MIT


# memoir
