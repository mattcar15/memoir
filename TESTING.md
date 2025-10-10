# Testing Guide

Quick guide to test all the new features.

## Prerequisites Check

```bash
# 1. Check Python environment
python --version  # Should be 3.8+

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install/update dependencies
pip install -r requirements.txt

# 4. Check Ollama is running
ollama ps

# 5. Check models are installed
ollama list
# Should show: gemma3:4b
# Optional: embeddinggemma

# 6. Pull missing models if needed
ollama pull gemma3:4b
ollama pull embeddinggemma
```

## Test 1: Basic Capture (Legacy Mode)

```bash
# Test a single capture without new features
python main.py capture --run-once

# Expected output:
# ✅ Screenshot captured
# ✅ Ollama processing completed
# ✅ Log saved
```

## Test 2: Capture with Vectorization

```bash
# Test capture with embeddings
python main.py capture --run-once --enable-vectorization

# Expected output:
# ✅ Screenshot captured
# ✅ Ollama processing completed
# 🔄 Creating embedding for memory...
# ✅ Created embedding (300 dimensions) in ~0.X s
# ✅ Memory added to vector store (Total: 1)
# ✅ Log saved
```

## Test 3: Search Memories

```bash
# First, create some test memories
python main.py capture --enable-vectorization --run-once

# Now search
python main.py search "test"

# Expected output:
# 🔍 Memory Search
# 🔍 Searching for: "test"
# 🔄 Creating query embedding...
# ✅ Created embedding (300 dimensions)
# 📊 Found X similar memories
# [Results displayed]
```

## Test 4: Live Reload

```bash
# Start in live reload mode
python main.py capture --live-reload --run-once

# In another terminal, edit a file:
echo "# test comment" >> memoir/config.py

# Expected: Application should auto-restart
# Output:
# 🔄 Code change detected: memoir/config.py
# ♻️  Restarting application...

# Clean up the test change:
git checkout memoir/config.py
```

## Test 5: Search with Multiple Results

```bash
# Create several memories
for i in {1..3}; do
  python main.py capture --enable-vectorization --run-once
  sleep 5
done

# Search with more results
python main.py search "activity" --results 3

# Expected: Shows 3 results with distances
```

## Test 6: Integration Test

```bash
# Full workflow test
# 1. Start continuous capture with vectorization
python main.py capture --enable-vectorization --interval 0.5 &
CAPTURE_PID=$!

# 2. Wait for a few captures
sleep 30

# 3. Stop capture
kill $CAPTURE_PID

# 4. Search your recent activity
python main.py search "what was I doing"

# 5. Check outputs
ls -la logs/          # Should have JSON files
ls -la vector_db/     # Should have ChromaDB files
```

## Test 7: Compare Tiers with Vectorization

```bash
# Compare resolution tiers with embeddings
python main.py capture --run-once --compare-tiers --enable-vectorization

# Expected: Comparison table with all three tiers
# Note: All three memories will be added to vector store
```

## Validation Checklist

After running tests, verify:

- [ ] JSON log files created in `logs/`
- [ ] Vector database created in `vector_db/`
- [ ] Search returns relevant results
- [ ] Live reload triggers on file changes
- [ ] No Python errors or exceptions
- [ ] Ollama models stay loaded (`ollama ps`)

## Performance Benchmarks

Expected timings (approximate):

- **Capture only**: 0.5-2s
- **Capture + embedding**: 0.7-2.5s
- **Search query**: 0.1-0.5s
- **Live reload restart**: 1-2s

## Troubleshooting Tests

### Test: Model Not Found
```bash
# Intentionally use wrong model
python main.py search "test" --embedding-model nonexistent

# Expected: Clear error message about model not found
```

### Test: Search Before Building Database
```bash
# Delete vector DB
rm -rf vector_db/

# Try to search
python main.py search "test"

# Expected: "No memories in vector store yet!" message
```

### Test: Embedding Model Not Installed
```bash
# Remove embedding model
ollama rm embeddinggemma

# Try to capture with vectorization
python main.py capture --run-once --enable-vectorization

# Expected: Warning about embedding model not found

# Restore model
ollama pull embeddinggemma
```

## Clean Up After Testing

```bash
# Remove test logs
rm -rf logs/

# Remove test vector database
rm -rf vector_db/

# Verify clean slate
ls logs/      # Should not exist or be empty
ls vector_db/ # Should not exist or be empty
```

## Continuous Testing

For development, you can use this loop:

```bash
# Watch for changes and test automatically
while true; do
  python main.py capture --run-once --enable-vectorization
  sleep 10
done
```

## Testing Checklist

- [x] Basic capture works
- [x] Vectorization works
- [x] Search works
- [x] Live reload works
- [x] Multiple results work
- [x] Integration test passes
- [x] Error handling works
- [x] Performance is acceptable

## Next Steps

Once all tests pass:
1. Review the [README.md](README.md) for full documentation
2. Check [QUICKSTART.md](QUICKSTART.md) for usage patterns
3. Read [CHANGELOG.md](CHANGELOG.md) for what changed
4. Start using the system for real!

Happy testing! 🧪

