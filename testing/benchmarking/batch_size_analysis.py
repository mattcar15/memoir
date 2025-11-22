#!/usr/bin/env python3
"""
Batch Size Analysis Script

Tests how different batch sizes (1-15 images) impact system metrics like temperature,
power consumption, and performance. Waits for GPU temperature to cool down between
tests to ensure fair comparison.

Produces graphs and fits equations to model the relationship between batch size
and system resource usage.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import psutil
from PIL import Image, ImageOps

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Import the system monitoring functions
sys.path.insert(0, str(Path(__file__).parent))
from compute_stats_v2 import (
    get_system_metrics,
    start_macmon,
    stop_macmon,
    apple_silicon,
)


@dataclass
class RepeatResult:
    """Results from a single repeat within a batch size test"""

    repeat_num: int
    latency: float
    cooldown_time: float
    cpu_temp_before: float
    gpu_temp_before: float
    cpu_temp_peak: float
    gpu_temp_peak: float
    cpu_power_mean: float
    gpu_power_mean: float
    total_power_mean: float
    mlx_gpu_peak_mb: float


@dataclass
class BatchResult:
    """Results from testing a single batch size"""

    batch_size: int
    num_repeats: int
    latency_mean: float
    latency_std: float
    latency_min: float
    latency_max: float
    # Temperature metrics
    cpu_temp_before: float
    gpu_temp_before: float
    cpu_temp_peak: float
    gpu_temp_peak: float
    cpu_temp_delta: float
    gpu_temp_delta: float
    # Power metrics
    cpu_power_mean: float
    gpu_power_mean: float
    total_power_mean: float
    cpu_power_peak: float
    gpu_power_peak: float
    total_power_peak: float
    # Usage metrics
    cpu_usage_mean: float
    gpu_usage_mean: float
    ram_usage_mean_gb: float
    ram_usage_mean_percent: float
    # MLX GPU memory
    mlx_gpu_peak_mb: float
    # Timing
    cooldown_time: float
    total_time: float
    # Per-repeat data for detailed plotting
    repeat_results: List[RepeatResult] = None


def list_images(folder: str) -> List[str]:
    """List all image files in a folder"""
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)
    )


def resize_image_if_needed(img: Image.Image, max_side: Optional[int]) -> Image.Image:
    """Resize image if it exceeds max_side"""
    if not max_side:
        return img
    width, height = img.size
    longest = max(width, height)
    if longest <= max_side:
        return img

    scale = max_side / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def wait_for_cooldown(
    target_gpu_temp: float, check_interval: float = 1.0, max_wait: float = 300.0
) -> float:
    """
    Wait for GPU temperature to drop below target.
    Returns the time spent waiting.
    """
    if not apple_silicon():
        print("Not on Apple Silicon, skipping cooldown wait")
        return 0.0

    start_time = time.time()
    print(f"\nWaiting for GPU temp to drop below {target_gpu_temp}°C...")

    consecutive_failures = 0
    max_failures = 3

    while True:
        # Timeout check
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            print(f"\n⚠ Timeout after {elapsed:.1f}s, proceeding anyway")
            return elapsed

        metrics = get_system_metrics(fresh=True)  # Get current temp, not buffered data
        if metrics and metrics["gpu_temp"] is not None:
            consecutive_failures = 0
            gpu_temp = metrics["gpu_temp"]
            print(f"  Current GPU temp: {gpu_temp:.1f}°C", end="\r")

            if gpu_temp < target_gpu_temp:
                elapsed = time.time() - start_time
                print(f"\n✓ GPU cooled to {gpu_temp:.1f}°C (waited {elapsed:.1f}s)")
                return elapsed
        else:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print(
                    f"\n⚠ Failed to get metrics {max_failures} times, proceeding anyway"
                )
                return time.time() - start_time

        time.sleep(check_interval)


def run_inference_with_monitoring(
    model,
    processor,
    config,
    images: List[str],
    prompt: str,
    max_new_tokens: int,
    max_image_side: Optional[int],
) -> tuple[float, List[Dict], float]:
    """
    Run inference on images sequentially (mlx-vlm doesn't support multi-image batching).
    Process each image separately but measure cumulative metrics.

    Returns:
        (total_latency, metrics_history, mlx_gpu_peak_mb)
    """

    # Load images EXACTLY like model_testing.py does
    def load_image(path: str) -> Image.Image:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return resize_image_if_needed(img, max_image_side)

    # Get baseline metrics before starting
    baseline_metrics = get_system_metrics(fresh=True)  # Get fresh data, not buffered

    # Reset MLX peak memory tracking
    mx.reset_peak_memory()

    total_latency = 0.0
    start_time = time.perf_counter()
    all_samples = []

    # Process each image sequentially (mlx-vlm bug prevents true batching)
    for img_path in images:
        img = load_image(img_path)

        formatted_prompt = apply_chat_template(
            processor,
            config,
            prompt,
            num_images=1,  # Always 1 since we process sequentially
        )

        # Run inference on single image
        start = time.perf_counter()
        output = generate(
            model,
            processor,
            formatted_prompt,
            [img],
            max_tokens=max_new_tokens,
            verbose=False,
        )
        end = time.perf_counter()

        total_latency += end - start

        # Sample multiple times to catch peak activity (GPU might still be hot)
        # Use fresh=True on first sample to get current data, not buffered data from during inference
        samples_for_this_image = []
        for i in range(4):  # Sample 4 times over 1 second
            sample = get_system_metrics(fresh=(i == 0))  # Only flush on first sample
            if sample:
                sample["timestamp"] = time.perf_counter()
                samples_for_this_image.append(sample)
            time.sleep(0.25)  # 250ms between samples

        # Keep the hottest sample from this image
        if samples_for_this_image:
            hottest = max(samples_for_this_image, key=lambda s: s.get("gpu_temp", 0))
            all_samples.append(hottest)

    end_time = time.perf_counter()

    # Get MLX GPU peak memory
    mlx_gpu_peak = mx.get_peak_memory() / (1024 * 1024)  # Convert to MB

    # Find the sample with peak GPU activity (highest temp is good proxy)
    peak_sample = None
    if all_samples:
        peak_sample = max(all_samples, key=lambda s: (s.get("gpu_temp") or 0))
        peak_sample["timestamp"] = end_time
        peak_sample["phase"] = "after"

    # Package metrics for return
    metrics_history = []
    if baseline_metrics:
        baseline_metrics["timestamp"] = start_time
        baseline_metrics["phase"] = "before"
        metrics_history.append(baseline_metrics)
    if peak_sample:
        metrics_history.append(peak_sample)

    return total_latency, metrics_history, mlx_gpu_peak


def test_batch_size(
    model,
    processor,
    config,
    all_images: List[str],
    batch_size: int,
    prompt: str,
    max_new_tokens: int,
    max_image_side: Optional[int],
    num_repeats: int,
    target_cooldown_temp: float,
) -> BatchResult:
    """
    Test a specific batch size multiple times and collect metrics.
    """
    print(f"\n{'='*70}")
    print(f"Testing batch size: {batch_size} images")
    print(f"{'='*70}")

    # Select images for this batch
    if batch_size > len(all_images):
        print(f"Warning: batch_size {batch_size} > available images {len(all_images)}")
        batch_size = len(all_images)

    images = all_images[:batch_size]

    latencies = []
    all_metrics = []
    mlx_gpu_peaks = []
    cooldown_times = []
    repeat_results = []  # Store per-repeat data for detailed plotting

    test_start_time = time.time()

    for repeat in range(num_repeats):
        print(f"\n--- Repeat {repeat + 1}/{num_repeats} ---")

        # Wait for cooldown before each repeat (except first)
        if repeat > 0:
            cooldown_time = wait_for_cooldown(target_cooldown_temp)
            cooldown_times.append(cooldown_time)
        else:
            cooldown_times.append(0.0)

        # Run inference with monitoring
        latency, metrics_history, mlx_peak = run_inference_with_monitoring(
            model,
            processor,
            config,
            images,
            prompt,
            max_new_tokens,
            max_image_side,
        )

        latencies.append(latency)
        all_metrics.extend(metrics_history)
        mlx_gpu_peaks.append(mlx_peak)

        # Extract per-repeat metrics for detailed plotting
        before_metrics = [m for m in metrics_history if m.get("phase") == "before"]
        after_metrics = [m for m in metrics_history if m.get("phase") == "after"]

        repeat_result = RepeatResult(
            repeat_num=repeat + 1,
            latency=latency,
            cooldown_time=cooldown_times[-1],
            cpu_temp_before=(
                before_metrics[0]["cpu_temp"]
                if before_metrics and before_metrics[0].get("cpu_temp") is not None
                else 0.0
            ),
            gpu_temp_before=(
                before_metrics[0]["gpu_temp"]
                if before_metrics and before_metrics[0].get("gpu_temp") is not None
                else 0.0
            ),
            cpu_temp_peak=(
                after_metrics[0]["cpu_temp"]
                if after_metrics and after_metrics[0].get("cpu_temp") is not None
                else 0.0
            ),
            gpu_temp_peak=(
                after_metrics[0]["gpu_temp"]
                if after_metrics and after_metrics[0].get("gpu_temp") is not None
                else 0.0
            ),
            cpu_power_mean=(
                after_metrics[0]["cpu_power"]
                if after_metrics and after_metrics[0].get("cpu_power") is not None
                else 0.0
            ),
            gpu_power_mean=(
                after_metrics[0]["gpu_power"]
                if after_metrics and after_metrics[0].get("gpu_power") is not None
                else 0.0
            ),
            total_power_mean=(
                after_metrics[0]["total_power"]
                if after_metrics and after_metrics[0].get("total_power") is not None
                else 0.0
            ),
            mlx_gpu_peak_mb=mlx_peak,
        )
        repeat_results.append(repeat_result)

        print(f"Latency: {latency:.3f}s, MLX GPU Peak: {mlx_peak:.1f}MB")

    test_end_time = time.time()
    total_time = test_end_time - test_start_time

    # Calculate statistics
    import statistics

    latency_mean = statistics.mean(latencies)
    latency_std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    latency_min = min(latencies)
    latency_max = max(latencies)

    # Temperature analysis - separate before/after phases
    before_metrics = [m for m in all_metrics if m.get("phase") == "before"]
    after_metrics = [m for m in all_metrics if m.get("phase") == "after"]

    # Get before temps (should be similar for all runs after cooldown)
    cpu_temps_before = [
        m["cpu_temp"] for m in before_metrics if m.get("cpu_temp") is not None
    ]
    gpu_temps_before = [
        m["gpu_temp"] for m in before_metrics if m.get("gpu_temp") is not None
    ]

    # Get after temps (peak temps after inference)
    cpu_temps_after = [
        m["cpu_temp"] for m in after_metrics if m.get("cpu_temp") is not None
    ]
    gpu_temps_after = [
        m["gpu_temp"] for m in after_metrics if m.get("gpu_temp") is not None
    ]

    cpu_temp_before = statistics.mean(cpu_temps_before) if cpu_temps_before else 0.0
    gpu_temp_before = statistics.mean(gpu_temps_before) if gpu_temps_before else 0.0
    cpu_temp_peak = statistics.mean(cpu_temps_after) if cpu_temps_after else 0.0
    gpu_temp_peak = statistics.mean(gpu_temps_after) if gpu_temps_after else 0.0
    cpu_temp_delta = cpu_temp_peak - cpu_temp_before
    gpu_temp_delta = gpu_temp_peak - gpu_temp_before

    # Power analysis - use after metrics which show peak usage
    cpu_power = [
        m["cpu_power"] for m in after_metrics if m.get("cpu_power") is not None
    ]
    gpu_power = [
        m["gpu_power"] for m in after_metrics if m.get("gpu_power") is not None
    ]
    total_power = [
        m["total_power"] for m in after_metrics if m.get("total_power") is not None
    ]

    cpu_power_mean = statistics.mean(cpu_power) if cpu_power else 0.0
    gpu_power_mean = statistics.mean(gpu_power) if gpu_power else 0.0
    total_power_mean = statistics.mean(total_power) if total_power else 0.0
    cpu_power_peak = max(cpu_power) if cpu_power else 0.0
    gpu_power_peak = max(gpu_power) if gpu_power else 0.0
    total_power_peak = max(total_power) if total_power else 0.0

    # Usage analysis
    cpu_usage = [
        m["cpu_usage"] for m in after_metrics if m.get("cpu_usage") is not None
    ]
    gpu_usage = [
        m["gpu_usage"] for m in after_metrics if m.get("gpu_usage") is not None
    ]
    ram_usage_gb = [
        m["ram_used_gb"] for m in after_metrics if m.get("ram_used_gb") is not None
    ]
    ram_usage_pct = [
        m["ram_usage_percent"]
        for m in after_metrics
        if m.get("ram_usage_percent") is not None
    ]

    cpu_usage_mean = statistics.mean(cpu_usage) if cpu_usage else 0.0
    gpu_usage_mean = statistics.mean(gpu_usage) if gpu_usage else 0.0
    ram_usage_mean_gb = statistics.mean(ram_usage_gb) if ram_usage_gb else 0.0
    ram_usage_mean_percent = statistics.mean(ram_usage_pct) if ram_usage_pct else 0.0

    mlx_gpu_peak_mb = statistics.mean(mlx_gpu_peaks) if mlx_gpu_peaks else 0.0

    result = BatchResult(
        batch_size=batch_size,
        num_repeats=num_repeats,
        latency_mean=latency_mean,
        latency_std=latency_std,
        latency_min=latency_min,
        latency_max=latency_max,
        cpu_temp_before=cpu_temp_before,
        gpu_temp_before=gpu_temp_before,
        cpu_temp_peak=cpu_temp_peak,
        gpu_temp_peak=gpu_temp_peak,
        cpu_temp_delta=cpu_temp_delta,
        gpu_temp_delta=gpu_temp_delta,
        cpu_power_mean=cpu_power_mean,
        gpu_power_mean=gpu_power_mean,
        total_power_mean=total_power_mean,
        cpu_power_peak=cpu_power_peak,
        gpu_power_peak=gpu_power_peak,
        total_power_peak=total_power_peak,
        cpu_usage_mean=cpu_usage_mean,
        gpu_usage_mean=gpu_usage_mean,
        ram_usage_mean_gb=ram_usage_mean_gb,
        ram_usage_mean_percent=ram_usage_mean_percent,
        mlx_gpu_peak_mb=mlx_gpu_peak_mb,
        cooldown_time=sum(cooldown_times),
        total_time=total_time,
        repeat_results=repeat_results,
    )

    # Print summary
    print(f"\n{'='*70}")
    print(f"Batch Size {batch_size} Summary:")
    print(f"  Latency: {latency_mean:.3f}s ± {latency_std:.3f}s")
    print(
        f"  GPU Temp: {gpu_temp_before:.1f}°C → {gpu_temp_peak:.1f}°C (Δ{gpu_temp_delta:.1f}°C)"
    )
    print(f"  GPU Power: avg={gpu_power_mean:.2f}W, peak={gpu_power_peak:.2f}W")
    print(f"  Total cooldown time: {sum(cooldown_times):.1f}s")
    print(f"{'='*70}")

    return result


def plot_results(results: List[BatchResult], output_dir: Path):
    """
    Create comprehensive plots showing how batch size affects metrics.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(
            "Error: matplotlib and numpy are required for plotting. Install with: pip install matplotlib numpy"
        )
        return

    batch_sizes = [r.batch_size for r in results]

    # Create 3x3 grid of plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("Batch Size Impact on System Metrics", fontsize=16, fontweight="bold")

    # 1. Latency
    ax = axes[0, 0]
    ax.plot(
        batch_sizes,
        [r.latency_mean for r in results],
        "o-",
        color="#FF6B6B",
        linewidth=2,
        markersize=8,
    )
    ax.fill_between(
        batch_sizes,
        [r.latency_mean - r.latency_std for r in results],
        [r.latency_mean + r.latency_std for r in results],
        alpha=0.3,
        color="#FF6B6B",
    )
    ax.set_xlabel("Batch Size (# images)")
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Inference Latency")
    ax.grid(True, alpha=0.3)

    # 2. GPU Temperature Delta
    ax = axes[0, 1]
    ax.plot(
        batch_sizes,
        [r.gpu_temp_delta for r in results],
        "o-",
        color="#4ECDC4",
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Batch Size (# images)")
    ax.set_ylabel("Temperature Increase (°C)")
    ax.set_title("GPU Temperature Spike")
    ax.grid(True, alpha=0.3)

    # 3. GPU Peak Temperature
    ax = axes[0, 2]
    ax.plot(
        batch_sizes,
        [r.gpu_temp_peak for r in results],
        "o-",
        color="#4ECDC4",
        linewidth=2,
        markersize=8,
    )
    ax.axhline(y=60, color="orange", linestyle="--", label="60°C threshold")
    ax.set_xlabel("Batch Size (# images)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("GPU Peak Temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Total Power Mean
    ax = axes[1, 0]
    ax.plot(
        batch_sizes,
        [r.total_power_mean for r in results],
        "o-",
        color="#95E1D3",
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Batch Size (# images)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Average Total Power")
    ax.grid(True, alpha=0.3)

    # 5. GPU Power Peak
    ax = axes[1, 1]
    ax.plot(
        batch_sizes,
        [r.gpu_power_peak for r in results],
        "o-",
        color="#4ECDC4",
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Batch Size (# images)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Peak GPU Power")
    ax.grid(True, alpha=0.3)

    # 6. CPU vs GPU Power
    ax = axes[1, 2]
    ax.plot(
        batch_sizes,
        [r.cpu_power_mean for r in results],
        "o-",
        label="CPU",
        color="#FF6B6B",
        linewidth=2,
        markersize=8,
    )
    ax.plot(
        batch_sizes,
        [r.gpu_power_mean for r in results],
        "o-",
        label="GPU",
        color="#4ECDC4",
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Batch Size (# images)")
    ax.set_ylabel("Power (W)")
    ax.set_title("CPU vs GPU Average Power")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7. GPU Usage
    ax = axes[2, 0]
    ax.plot(
        batch_sizes,
        [r.gpu_usage_mean for r in results],
        "o-",
        color="#4ECDC4",
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Batch Size (# images)")
    ax.set_ylabel("Usage (%)")
    ax.set_title("Average GPU Usage")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # 8. MLX GPU Memory
    ax = axes[2, 1]
    ax.plot(
        batch_sizes,
        [r.mlx_gpu_peak_mb for r in results],
        "o-",
        color="#A78BFA",
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Batch Size (# images)")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("MLX GPU Peak Memory")
    ax.grid(True, alpha=0.3)

    # 9. Cooldown Time
    ax = axes[2, 2]
    ax.plot(
        batch_sizes,
        [r.cooldown_time for r in results],
        "o-",
        color="#FFA07A",
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Batch Size (# images)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Total Cooldown Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / "batch_size_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Plots saved to: {output_path}")

    plt.show()


def plot_individual_batch_results(results: List[BatchResult], output_dir: Path):
    """
    Create separate detailed plots for each batch size showing metrics across repeats.
    Each batch size gets its own figure showing cooldown times, latencies, and temperatures.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(
            "Error: matplotlib and numpy are required for plotting. Install with: pip install matplotlib numpy"
        )
        return

    for result in results:
        if not result.repeat_results:
            print(
                f"Warning: No per-repeat data for batch size {result.batch_size}, skipping individual plot"
            )
            continue

        batch_size = result.batch_size
        repeat_nums = [r.repeat_num for r in result.repeat_results]

        # Create figure with 2x3 grid for this batch size
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(
            f"Batch Size {batch_size} - Detailed Metrics Across {result.num_repeats} Repeats",
            fontsize=14,
            fontweight="bold",
        )

        # 1. Latency per repeat
        ax = axes[0, 0]
        latencies = [r.latency for r in result.repeat_results]
        ax.plot(
            repeat_nums, latencies, "o-", color="#FF6B6B", linewidth=2, markersize=10
        )
        ax.axhline(
            y=result.latency_mean,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {result.latency_mean:.3f}s",
        )
        ax.set_xlabel("Repeat Number", fontsize=11)
        ax.set_ylabel("Latency (seconds)", fontsize=11)
        ax.set_title("Inference Latency per Repeat", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(repeat_nums)

        # 2. Cooldown time per repeat
        ax = axes[0, 1]
        cooldown_times = [r.cooldown_time for r in result.repeat_results]
        colors = ["#95E1D3" if t == 0 else "#FFA07A" for t in cooldown_times]
        bars = ax.bar(
            repeat_nums, cooldown_times, color=colors, alpha=0.8, edgecolor="black"
        )
        ax.set_xlabel("Repeat Number", fontsize=11)
        ax.set_ylabel("Cooldown Time (seconds)", fontsize=11)
        ax.set_title("Cooldown Time After Each Repeat", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(repeat_nums)
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, cooldown_times)):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}s",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # 3. GPU Temperature progression
        ax = axes[0, 2]
        gpu_temps_before = [r.gpu_temp_before for r in result.repeat_results]
        gpu_temps_peak = [r.gpu_temp_peak for r in result.repeat_results]
        x_pos = np.arange(len(repeat_nums))
        width = 0.35
        ax.bar(
            x_pos - width / 2,
            gpu_temps_before,
            width,
            label="Before",
            color="#4ECDC4",
            alpha=0.8,
        )
        ax.bar(
            x_pos + width / 2,
            gpu_temps_peak,
            width,
            label="Peak",
            color="#FF6B6B",
            alpha=0.8,
        )
        ax.axhline(
            y=60, color="orange", linestyle="--", alpha=0.7, label="60°C threshold"
        )
        ax.set_xlabel("Repeat Number", fontsize=11)
        ax.set_ylabel("GPU Temperature (°C)", fontsize=11)
        ax.set_title("GPU Temperature: Before vs Peak", fontsize=12, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(repeat_nums)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # 4. GPU Temperature Delta per repeat
        ax = axes[1, 0]
        gpu_temp_deltas = [
            r.gpu_temp_peak - r.gpu_temp_before for r in result.repeat_results
        ]
        ax.plot(
            repeat_nums,
            gpu_temp_deltas,
            "o-",
            color="#4ECDC4",
            linewidth=2,
            markersize=10,
        )
        ax.axhline(
            y=result.gpu_temp_delta,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {result.gpu_temp_delta:.1f}°C",
        )
        ax.set_xlabel("Repeat Number", fontsize=11)
        ax.set_ylabel("Temperature Increase (°C)", fontsize=11)
        ax.set_title("GPU Temperature Spike per Repeat", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(repeat_nums)

        # 5. Power consumption per repeat
        ax = axes[1, 1]
        gpu_power = [r.gpu_power_mean for r in result.repeat_results]
        cpu_power = [r.cpu_power_mean for r in result.repeat_results]
        total_power = [r.total_power_mean for r in result.repeat_results]
        ax.plot(
            repeat_nums,
            total_power,
            "o-",
            label="Total",
            color="#95E1D3",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            repeat_nums,
            gpu_power,
            "s-",
            label="GPU",
            color="#4ECDC4",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            repeat_nums,
            cpu_power,
            "^-",
            label="CPU",
            color="#FF6B6B",
            linewidth=2,
            markersize=8,
        )
        ax.set_xlabel("Repeat Number", fontsize=11)
        ax.set_ylabel("Power (W)", fontsize=11)
        ax.set_title("Power Consumption per Repeat", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(repeat_nums)

        # 6. MLX GPU Memory per repeat
        ax = axes[1, 2]
        mlx_memory = [r.mlx_gpu_peak_mb for r in result.repeat_results]
        ax.plot(
            repeat_nums, mlx_memory, "o-", color="#A78BFA", linewidth=2, markersize=10
        )
        ax.axhline(
            y=result.mlx_gpu_peak_mb,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {result.mlx_gpu_peak_mb:.1f}MB",
        )
        ax.set_xlabel("Repeat Number", fontsize=11)
        ax.set_ylabel("Memory (MB)", fontsize=11)
        ax.set_title("MLX GPU Peak Memory per Repeat", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(repeat_nums)

        plt.tight_layout()

        # Save figure
        output_path = output_dir / f"batch_size_{batch_size}_detailed.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Saved detailed plot: {output_path.name}")

        plt.close(fig)  # Close to free memory


def fit_models(results: List[BatchResult]):
    """
    Fit polynomial models to predict metrics based on batch size.
    """
    try:
        import numpy as np
        from numpy.polynomial import Polynomial
    except ImportError:
        print(
            "Error: numpy is required for model fitting. Install with: pip install numpy"
        )
        return

    batch_sizes = np.array([r.batch_size for r in results])

    print(f"\n{'='*70}")
    print("PREDICTIVE MODELS (Polynomial Fits)")
    print(f"{'='*70}\n")

    # Fit models for key metrics
    metrics_to_fit = [
        ("latency_mean", "Latency (seconds)", 2),
        ("gpu_temp_delta", "GPU Temp Spike (°C)", 2),
        ("gpu_power_mean", "GPU Power (W)", 2),
        ("total_power_mean", "Total Power (W)", 2),
        ("mlx_gpu_peak_mb", "MLX GPU Memory (MB)", 2),
    ]

    models = {}

    for attr, label, degree in metrics_to_fit:
        values = np.array([getattr(r, attr) for r in results])

        # Fit polynomial
        poly = Polynomial.fit(batch_sizes, values, degree)
        models[attr] = poly

        # Calculate R²
        fitted_values = poly(batch_sizes)
        ss_res = np.sum((values - fitted_values) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Print equation
        coeffs = poly.convert().coef
        equation = f"y = {coeffs[0]:.4f}"
        for i, c in enumerate(coeffs[1:], 1):
            equation += f" + {c:.4f}*x^{i}"

        print(f"{label}:")
        print(f"  {equation}")
        print(f"  R² = {r_squared:.4f}")
        print()

    # Example predictions
    print(f"\n{'='*70}")
    print("EXAMPLE PREDICTIONS")
    print(f"{'='*70}\n")

    test_batch_sizes = [5, 10, 20]
    for test_size in test_batch_sizes:
        print(f"Predicted metrics for batch size {test_size}:")
        for attr, label, _ in metrics_to_fit:
            if attr in models:
                predicted = models[attr](test_size)
                print(f"  {label}: {predicted:.4f}")
        print()

    return models


def main():
    parser = argparse.ArgumentParser(
        description="Analyze how different batch sizes impact system metrics"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lmstudio-community/Qwen3-VL-2B-Instruct-MLX-8bit",
        help="HF model repo or local path",
    )
    parser.add_argument(
        "--image-dir", type=str, required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the key pieces of the image that explain what the user is doing.",
        help="Prompt to use for all images",
    )
    parser.add_argument(
        "--min-batch-size", type=int, default=1, help="Minimum batch size to test"
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=15, help="Maximum batch size to test"
    )
    parser.add_argument(
        "--batch-step",
        type=int,
        default=1,
        help="Step size between batch sizes (default: 1)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of times to repeat each batch size test",
    )
    parser.add_argument(
        "--cooldown-temp",
        type=float,
        default=60.0,
        help="Target GPU temperature (°C) to wait for before each test",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64, help="Max tokens to generate per call"
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1024,
        help="Max image dimension in pixels",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="batch_analysis_results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Check if on Apple Silicon
    if not apple_silicon():
        print("Error: This script requires Apple Silicon")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load images
    images = list_images(args.image_dir)
    if not images:
        print(f"Error: No images found in {args.image_dir}")
        return 1

    print(f"Found {len(images)} images")

    if args.max_batch_size > len(images):
        print(
            f"Warning: max_batch_size ({args.max_batch_size}) > available images ({len(images)})"
        )
        print(f"Capping at {len(images)} images")
        args.max_batch_size = len(images)

    # Start macmon for system monitoring
    print("Starting system monitoring...")
    start_macmon()

    # Load model
    print(f"\nLoading model {args.model}...")
    model, processor = load(args.model)
    try:
        config = model.config
    except AttributeError:
        config = load_config(args.model)
    print("✓ Model loaded")
    print("Sleeping for 10 seconds to let the model warm up so stats are accurate...")
    time.sleep(10)

    # Wait for initial cooldown
    print(f"\nWaiting for initial cooldown to {args.cooldown_temp}°C...")
    wait_for_cooldown(args.cooldown_temp)

    # Test different batch sizes
    batch_sizes = range(args.min_batch_size, args.max_batch_size + 1, args.batch_step)
    results = []

    try:
        for batch_size in batch_sizes:
            result = test_batch_size(
                model,
                processor,
                config,
                images,
                batch_size,
                args.prompt,
                args.max_new_tokens,
                args.max_image_side,
                args.repeats,
                args.cooldown_temp,
            )
            results.append(result)

            # Save intermediate results
            results_file = output_dir / "batch_analysis_results.json"
            with open(results_file, "w") as f:
                # Convert results to dict, including nested repeat_results
                results_data = []
                for r in results:
                    r_dict = asdict(r)
                    # Convert repeat_results from list of dicts to list (asdict should handle this)
                    results_data.append(r_dict)
                json.dump(results_data, f, indent=2)

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user (Ctrl+C)")
        if not results:
            print("No data collected, exiting.")
            stop_macmon()
            return 1

    # Save final results
    results_file = output_dir / "batch_analysis_results.json"
    with open(results_file, "w") as f:
        # Convert results to dict, including nested repeat_results
        results_data = []
        for r in results:
            r_dict = asdict(r)
            results_data.append(r_dict)
        json.dump(results_data, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")

    # Print summary table
    print(f"\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    print(
        f"{'Batch':>6} | {'Latency':>10} | {'GPU Temp Δ':>11} | {'GPU Peak':>9} | {'GPU Power':>10} | {'MLX GPU':>10}"
    )
    print(
        f"{'Size':>6} | {'(sec)':>10} | {'(°C)':>11} | {'(°C)':>9} | {'(W)':>10} | {'(MB)':>10}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r.batch_size:>6} | {r.latency_mean:>10.3f} | {r.gpu_temp_delta:>11.1f} | "
            f"{r.gpu_temp_peak:>9.1f} | {r.gpu_power_mean:>10.2f} | {r.mlx_gpu_peak_mb:>10.1f}"
        )
    print("=" * 100)

    # Fit predictive models
    fit_models(results)

    # Create summary plots (all batch sizes on one figure)
    print("\nGenerating summary plots...")
    plot_results(results, output_dir)

    # Create individual plots for each batch size
    print("\nGenerating individual batch size plots...")
    plot_individual_batch_results(results, output_dir)

    # Cleanup
    stop_macmon()

    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
