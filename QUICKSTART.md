# Quick Start Guide

## Setup (First Time Only)

### 1. Install Dependencies
```bash
# Activate your virtual environment
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 2. Install Ollama Models
```bash
# Required: Vision model for summarization
ollama pull gemma3:4b

# Optional: Embedding model for semantic search
ollama pull embeddinggemma
```

## Basic Usage

### Capture Screenshots (Vectorization ON by default)
```bash
# Run with defaults (captures every minute, vectorization enabled)
python main.py

# Run once to test
python main.py --run-once

# Capture with image saving
python main.py --save-images
```

### Capture WITHOUT Vectorization
```bash
# Disable vectorization if you only want JSON logs
python main.py --disable-vectorization

# You won't be able to search these memories
```

### Search Your Memories
```bash
# Basic search (works automatically - vectorization is on by default)
python main.py search "what was I working on?"

# Get more results
python main.py search "coding projects" --results 10
```

## Development Mode

### Live Reload
```bash
# Auto-restart when code changes (vectorization is on by default)
python main.py --live-reload

# With other options
python main.py --live-reload --save-images
```

## Common Workflows

### Daily Memory Capture
```bash
# Start capturing (vectorization is on by default)
python main.py --interval 5

# Leave running in the background
# Later, search your memories:
python main.py search "meetings with clients"
```

### Testing & Development
```bash
# Single test capture (vectorization on by default)
python main.py --run-once

# Test without vectorization
python main.py --run-once --disable-vectorization

# Compare resolution tiers
python main.py --run-once --compare-tiers
```

### Save Screenshots
```bash
# Capture with screenshot images saved (vectorization on by default)
python main.py --save-images --interval 10

# Without vectorization
python main.py --save-images --disable-vectorization
```

## Troubleshooting Quick Fixes

### Ollama Not Running
```bash
# Start Ollama service
ollama serve

# Verify it's working
ollama ps
```

### Missing Models
```bash
# Check installed models
ollama list

# Install missing models
ollama pull gemma3:4b
ollama pull embeddinggemma
```

### No Search Results
```bash
# First, build the database with capture mode (vectorization is on by default)
python main.py --run-once

# Now search should work
python main.py search "test query"
```

### Reset Vector Database
```bash
# If you want to start fresh
rm -rf vector_db/

# Rebuild by running capture (vectorization is on by default)
python main.py --run-once
```

## Tips

- **Vectorization is ON by default** - all memories are automatically searchable!
- **Use `--disable-vectorization`** only if you don't want search functionality
- **Leave it running** in the background for continuous memory capture
- **Use natural language** for searches - the AI understands context
- **Vector DB persists** - your memories are saved across sessions
- **JSON logs are separate** - you have both structured logs and searchable memories

## Next Steps

1. Read the full [README.md](README.md) for all options
2. Check your logs in `logs/` directory
3. Explore your vector database in `vector_db/`
4. Customize intervals and models to your needs

Happy memory capturing! 🎉

