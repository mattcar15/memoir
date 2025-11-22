# Batch Size Analysis Tool

This tool analyzes how different batch sizes (number of images per inference call) impact system performance metrics like temperature, power consumption, latency, and memory usage.

## Purpose

The goal is to build predictive models that can estimate:
- How much temperature will spike for a given batch size
- How long inference will take
- How much power will be consumed
- How much GPU memory will be needed

This helps optimize the number of images to process in parallel for your specific hardware.

## Features

- **Automatic cooldown**: Waits for GPU temperature to drop below a threshold before each test
- **Multiple repeats**: Tests each batch size multiple times for statistical accuracy
- **Comprehensive metrics**: Tracks temperature, power, CPU/GPU usage, and memory
- **Predictive models**: Fits polynomial equations to predict metrics for untested batch sizes
- **Beautiful graphs**: Generates 9 plots showing relationships between batch size and metrics
- **JSON export**: Saves all raw data for further analysis

## Usage

### Basic Usage

Test batch sizes from 1 to 15 images:

```bash
python batch_size_analysis.py \
  --image-dir ../test_images \
  --prompt "Explain the key pieces of the image that explain what the user is doing."
```

### Advanced Options

```bash
python batch_size_analysis.py \
  --image-dir ../test_images \
  --prompt "Describe what you see in detail." \
  --min-batch-size 1 \
  --max-batch-size 20 \
  --batch-step 2 \
  --repeats 5 \
  --cooldown-temp 55.0 \
  --output-dir my_analysis_results
```

### All Options

- `--model`: Model to use (default: Qwen3-VL-2B-Instruct-MLX-8bit)
- `--image-dir`: Directory containing test images (required)
- `--prompt`: Prompt to use for inference
- `--min-batch-size`: Minimum batch size to test (default: 1)
- `--max-batch-size`: Maximum batch size to test (default: 15)
- `--batch-step`: Increment between batch sizes (default: 1)
  - Example: step=2 tests 1, 3, 5, 7, etc.
- `--repeats`: Times to repeat each batch size (default: 3)
- `--cooldown-temp`: GPU temp threshold in °C (default: 60.0)
- `--max-new-tokens`: Max output tokens (default: 64)
- `--max-image-side`: Max image dimension (default: 1024)
- `--output-dir`: Where to save results (default: batch_analysis_results)

## Output

The tool generates:

### 1. Console Output
- Real-time progress for each test
- Summary table with key metrics
- Predictive equations for each metric
- Example predictions for untested batch sizes

### 2. Graph (`batch_size_analysis.png`)
A 3×3 grid showing:
1. **Inference Latency**: How batch size affects processing time
2. **GPU Temperature Spike**: Temperature increase during inference
3. **GPU Peak Temperature**: Maximum temperature reached
4. **Average Total Power**: Mean power consumption
5. **Peak GPU Power**: Maximum GPU power draw
6. **CPU vs GPU Power**: Power distribution comparison
7. **Average GPU Usage**: GPU utilization percentage
8. **MLX GPU Memory**: Peak GPU memory usage
9. **Cooldown Time**: Time spent waiting for GPU to cool

### 3. JSON Results (`batch_analysis_results.json`)
Complete data for all tests including:
- Latency statistics (mean, std, min, max)
- Temperature metrics (before, peak, delta)
- Power metrics (mean, peak for CPU/GPU/total)
- Usage metrics (CPU, GPU, RAM)
- MLX GPU memory peaks
- Timing information

## Example Output

```
============================================================
Batch Size 5 Summary:
  Latency: 2.145s ± 0.032s
  GPU Temp: 54.2°C → 68.3°C (Δ14.1°C)
  GPU Power: avg=12.34W, peak=18.56W
  Total cooldown time: 45.3s
============================================================

PREDICTIVE MODELS (Polynomial Fits)
====================================================================

Latency (seconds):
  y = 0.8234 + 0.1456*x^1 + 0.0023*x^2
  R² = 0.9876

GPU Temp Spike (°C):
  y = 5.234 + 1.234*x^1 + 0.0456*x^2
  R² = 0.9654

GPU Power (W):
  y = 2.345 + 0.789*x^1 + 0.0123*x^2
  R² = 0.9432
```

## Interpreting Results

### Temperature Management
- **GPU Temp Delta**: Shows thermal impact of batch size
- Higher batch sizes = more heat = longer cooldown needed
- Use this to set appropriate cooldown thresholds in production

### Performance Optimization
- **Latency**: Find sweet spot where latency per image is minimized
- Smaller batches have overhead, larger batches may slow down
- Look for the batch size with best throughput (images/second)

### Resource Planning
- **GPU Memory**: Ensure your batch size fits in available memory
- **Power**: Higher batch sizes may trigger thermal throttling
- Balance performance vs heat/power consumption

## Tips

1. **Ensure enough images**: You need at least `max_batch_size` images in your test directory

2. **Be patient**: With cooldown waits, this can take a while
   - Example: 15 batch sizes × 3 repeats × 1min cooldown = ~45 minutes

3. **Interrupt safely**: Press Ctrl+C to stop - partial results will be saved

4. **Use consistent conditions**: 
   - Close other apps to reduce noise
   - Let the system reach idle state before starting
   - Use the same image types/sizes for fair comparison

5. **Adjust cooldown temp**: Lower values (e.g., 55°C) give more consistent results but take longer

## Integration

Use the fitted equations in your code to predict resource usage:

```python
# From the output equations
def predict_latency(batch_size):
    return 0.8234 + 0.1456 * batch_size + 0.0023 * batch_size**2

def predict_gpu_temp_spike(batch_size):
    return 5.234 + 1.234 * batch_size + 0.0456 * batch_size**2

# Decide if batch is safe
batch_size = 10
current_gpu_temp = 55.0
predicted_spike = predict_gpu_temp_spike(batch_size)

if current_gpu_temp + predicted_spike < 85.0:  # Safe threshold
    print(f"Safe to process {batch_size} images")
else:
    print(f"Batch too large, reduce size or wait for cooldown")
```

## Requirements

- Apple Silicon Mac (uses macmon for temperature monitoring)
- MLX and mlx-vlm installed
- matplotlib and numpy for graphing
- compute_stats_v2.py in the same directory

## Troubleshooting

**"macmon binary not found"**: Make sure you've built the macmon project in the correct location

**"No images found"**: Check your --image-dir path and ensure it contains supported image formats (jpg, png, etc.)

**"Model failed to load"**: Verify the model name/path and ensure you have enough disk space

**Graphs not showing**: Install matplotlib: `pip install matplotlib`



