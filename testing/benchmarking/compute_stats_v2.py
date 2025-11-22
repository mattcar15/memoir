import platform
from pathlib import Path
import subprocess
import json
import atexit
import argparse
import time
from typing import List, Dict

MACMON = (
    Path(__file__).parent
    / "macmon"
    / "target"
    / "aarch64-apple-darwin"
    / "release"
    / "macmon"
)

# Global process handle
_macmon_proc = None


def apple_silicon():
    return platform.machine() in ("arm64", "aarch64")


def start_macmon():
    """Start macmon process if not already running."""
    global _macmon_proc

    if _macmon_proc is not None and _macmon_proc.poll() is None:
        return  # Already running

    if not apple_silicon():
        return

    # Start macmon in pipe mode with no sample limit (-s 0 means infinite)
    # Update interval of 250ms for faster sampling during inference
    _macmon_proc = subprocess.Popen(
        [str(MACMON), "pipe", "-i", "250"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
    )

    # Register cleanup on exit
    atexit.register(stop_macmon)


def stop_macmon():
    """Stop the macmon process."""
    global _macmon_proc
    if _macmon_proc is not None:
        _macmon_proc.terminate()
        _macmon_proc.wait(timeout=2)
        _macmon_proc = None


def flush_stale_metrics():
    """
    Drain stale data from the macmon pipe buffer.
    Call this before get_system_metrics() if there's been a delay since
    the last read (e.g., during inference) to ensure fresh data.
    """
    import select

    if not apple_silicon() or _macmon_proc is None:
        return

    # Non-blocking drain of all available data
    while True:
        ready, _, _ = select.select([_macmon_proc.stdout], [], [], 0)
        if ready:
            line = _macmon_proc.stdout.readline()
            if not line:
                break
        else:
            break


def get_system_metrics(fresh=False):
    """
    Get current system metrics.

    Args:
        fresh: If True, drain stale buffered data and return only the most recent metrics.
               Use this after periods where you weren't reading (e.g., during inference).

    Returns:
        dict: Dictionary containing:
            - cpu_temp: CPU temperature in Celsius
            - gpu_temp: GPU temperature in Celsius
            - cpu_power: CPU power in Watts
            - gpu_power: GPU power in Watts
            - total_power: Total system power in Watts
            - cpu_usage: CPU usage percentage (combined E-cores and P-cores)
            - gpu_usage: GPU usage percentage
            - ram_used_gb: RAM used in GB
            - ram_total_gb: Total RAM in GB
            - ram_usage_percent: RAM usage percentage
    """
    import select

    if not apple_silicon():
        return None

    # Ensure macmon is running
    start_macmon()

    if _macmon_proc is None:
        return None

    # Read the next line of output
    try:
        if fresh:
            # Drain buffer and keep only the most recent data
            line = None
            while True:
                ready, _, _ = select.select([_macmon_proc.stdout], [], [], 0)
                if ready:
                    new_line = _macmon_proc.stdout.readline()
                    if new_line:
                        line = new_line
                    else:
                        break
                else:
                    break

            # If buffer was empty, do a blocking read for fresh data
            if line is None:
                line = _macmon_proc.stdout.readline()
        else:
            line = _macmon_proc.stdout.readline()

        if not line:
            # Process ended, restart it
            stop_macmon()
            start_macmon()
            line = _macmon_proc.stdout.readline()

        line = line.strip()
        if line:
            data = json.loads(line)

            # Extract temperatures
            temps = data.get("temp", {})

            # Extract power metrics
            cpu_power = data.get("cpu_power")
            gpu_power = data.get("gpu_power")
            total_power = data.get("all_power")

            # Extract usage percentages (values are 0-1, convert to 0-100)
            # ecpu_usage and pcpu_usage are [frequency_mhz, usage_fraction]
            ecpu_usage = data.get("ecpu_usage", [0, 0])
            pcpu_usage = data.get("pcpu_usage", [0, 0])
            gpu_usage_data = data.get("gpu_usage", [0, 0])

            # Average E-core and P-core usage for overall CPU usage (convert to percentage)
            cpu_usage = (
                ((ecpu_usage[1] + pcpu_usage[1]) / 2) * 100
                if len(ecpu_usage) > 1 and len(pcpu_usage) > 1
                else None
            )
            gpu_usage = gpu_usage_data[1] * 100 if len(gpu_usage_data) > 1 else None

            # Extract memory info
            memory = data.get("memory", {})
            ram_used = memory.get("ram_usage")  # bytes
            ram_total = memory.get("ram_total")  # bytes

            return {
                "cpu_temp": temps.get("cpu_temp_avg"),
                "gpu_temp": temps.get("gpu_temp_avg"),
                "cpu_power": cpu_power,
                "gpu_power": gpu_power,
                "total_power": total_power,
                "cpu_usage": cpu_usage,
                "gpu_usage": gpu_usage,
                "ram_used_gb": ram_used / (1024**3) if ram_used else None,
                "ram_total_gb": ram_total / (1024**3) if ram_total else None,
                "ram_usage_percent": (
                    (ram_used / ram_total * 100) if ram_used and ram_total else None
                ),
            }
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing macmon output: {e}")

    return None


def collect_metrics(duration: int, interval: int = 1) -> List[Dict]:
    """
    Collect system metrics over a period of time.

    Args:
        duration: Total time to collect metrics in seconds
        interval: Interval between samples in seconds (default: 1)

    Returns:
        List of metric dictionaries
    """
    num_samples = duration // interval
    metrics_history = []

    print(
        f"Collecting {num_samples} samples over {duration} seconds (interval: {interval}s)..."
    )

    for i in range(num_samples):
        metrics = get_system_metrics()
        if metrics:
            metrics["timestamp"] = time.time()
            metrics_history.append(metrics)
            print(
                f"Sample {i+1}/{num_samples}: "
                f"CPU Temp={metrics['cpu_temp']:.1f}°C, "
                f"GPU Temp={metrics['gpu_temp']:.1f}°C, "
                f"CPU Usage={metrics['cpu_usage']:.1f}%, "
                f"GPU Usage={metrics['gpu_usage']:.1f}%, "
                f"CPU Power={metrics['cpu_power']:.2f}W, "
                f"GPU Power={metrics['gpu_power']:.2f}W, "
                f"RAM={metrics['ram_used_gb']:.1f}/{metrics['ram_total_gb']:.1f}GB ({metrics['ram_usage_percent']:.1f}%)"
            )
        else:
            print(f"Sample {i+1}/{num_samples}: No data available")

    return metrics_history


def plot_metrics(metrics_history: List[Dict]):
    """
    Plot collected metrics over time.

    Args:
        metrics_history: List of metric dictionaries with timestamps
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Error: matplotlib is required for plotting. Install with: pip install matplotlib"
        )
        return

    if not metrics_history:
        print("No data to plot")
        return

    # Extract data
    timestamps = [
        (m["timestamp"] - metrics_history[0]["timestamp"]) for m in metrics_history
    ]
    cpu_temps = [m["cpu_temp"] for m in metrics_history]
    gpu_temps = [m["gpu_temp"] for m in metrics_history]
    cpu_power = [m["cpu_power"] for m in metrics_history]
    gpu_power = [m["gpu_power"] for m in metrics_history]
    total_power = [m["total_power"] for m in metrics_history]
    cpu_usage = [m["cpu_usage"] for m in metrics_history]
    gpu_usage = [m["gpu_usage"] for m in metrics_history]
    ram_used = [m["ram_used_gb"] for m in metrics_history]
    ram_total = [m["ram_total_gb"] for m in metrics_history]
    ram_usage_percent = [m["ram_usage_percent"] for m in metrics_history]

    # Create subplots - 3x2 grid for 6 charts
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("System Metrics Over Time", fontsize=16, fontweight="bold")

    # Temperature plot
    ax1.plot(timestamps, cpu_temps, label="CPU", color="#FF6B6B", linewidth=2)
    ax1.plot(timestamps, gpu_temps, label="GPU", color="#4ECDC4", linewidth=2)
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Temperature")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Power plot
    ax2.plot(timestamps, cpu_power, label="CPU", color="#FF6B6B", linewidth=2)
    ax2.plot(timestamps, gpu_power, label="GPU", color="#4ECDC4", linewidth=2)
    ax2.plot(
        timestamps,
        total_power,
        label="Total",
        color="#95E1D3",
        linewidth=2,
        linestyle="--",
    )
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Power (W)")
    ax2.set_title("Power Consumption")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # CPU Usage plot
    ax3.plot(timestamps, cpu_usage, label="CPU", color="#FF6B6B", linewidth=2)
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Usage (%)")
    ax3.set_title("CPU Usage")
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # GPU Usage plot
    ax4.plot(timestamps, gpu_usage, label="GPU", color="#4ECDC4", linewidth=2)
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Usage (%)")
    ax4.set_title("GPU Usage")
    ax4.set_ylim(0, 100)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # RAM Usage (GB) plot
    ax5.plot(timestamps, ram_used, label="Used", color="#A78BFA", linewidth=2)
    ax5.plot(
        timestamps,
        ram_total,
        label="Total",
        color="#D8B4FE",
        linewidth=2,
        linestyle="--",
        alpha=0.7,
    )
    ax5.fill_between(timestamps, ram_used, alpha=0.3, color="#A78BFA")
    ax5.set_xlabel("Time (seconds)")
    ax5.set_ylabel("Memory (GB)")
    ax5.set_title("RAM Usage")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # RAM Usage (%) plot
    ax6.plot(timestamps, ram_usage_percent, label="RAM", color="#A78BFA", linewidth=2)
    ax6.fill_between(timestamps, ram_usage_percent, alpha=0.3, color="#A78BFA")
    ax6.set_xlabel("Time (seconds)")
    ax6.set_ylabel("Usage (%)")
    ax6.set_title("RAM Usage Percentage")
    ax6.set_ylim(0, 100)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor Apple Silicon system metrics (temperature, power, usage)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration to collect metrics in seconds (default: 60)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Interval between samples in seconds (default: 1)",
    )
    parser.add_argument(
        "--graph", action="store_true", help="Display graphs of collected metrics"
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Suppress console output during collection",
    )

    args = parser.parse_args()

    if not apple_silicon():
        print("Error: This tool only works on Apple Silicon Macs")
        exit(1)

    # Collect metrics
    metrics_history = collect_metrics(args.duration, args.interval)

    # Print summary statistics
    if metrics_history:
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)

        avg_cpu_temp = sum(m["cpu_temp"] for m in metrics_history) / len(
            metrics_history
        )
        avg_gpu_temp = sum(m["gpu_temp"] for m in metrics_history) / len(
            metrics_history
        )
        avg_cpu_power = sum(m["cpu_power"] for m in metrics_history) / len(
            metrics_history
        )
        avg_gpu_power = sum(m["gpu_power"] for m in metrics_history) / len(
            metrics_history
        )
        avg_total_power = sum(m["total_power"] for m in metrics_history) / len(
            metrics_history
        )
        avg_cpu_usage = sum(m["cpu_usage"] for m in metrics_history) / len(
            metrics_history
        )
        avg_gpu_usage = sum(m["gpu_usage"] for m in metrics_history) / len(
            metrics_history
        )
        avg_ram_used = sum(m["ram_used_gb"] for m in metrics_history) / len(
            metrics_history
        )
        avg_ram_percent = sum(m["ram_usage_percent"] for m in metrics_history) / len(
            metrics_history
        )

        max_cpu_temp = max(m["cpu_temp"] for m in metrics_history)
        max_gpu_temp = max(m["gpu_temp"] for m in metrics_history)
        max_cpu_power = max(m["cpu_power"] for m in metrics_history)
        max_gpu_power = max(m["gpu_power"] for m in metrics_history)
        max_total_power = max(m["total_power"] for m in metrics_history)
        max_ram_used = max(m["ram_used_gb"] for m in metrics_history)
        max_ram_percent = max(m["ram_usage_percent"] for m in metrics_history)

        print(
            f"Temperature: CPU avg={avg_cpu_temp:.1f}°C (max={max_cpu_temp:.1f}°C), "
            f"GPU avg={avg_gpu_temp:.1f}°C (max={max_gpu_temp:.1f}°C)"
        )
        print(
            f"Power: CPU avg={avg_cpu_power:.2f}W (max={max_cpu_power:.2f}W), "
            f"GPU avg={avg_gpu_power:.2f}W (max={max_gpu_power:.2f}W)"
        )
        print(f"Total Power: avg={avg_total_power:.2f}W (max={max_total_power:.2f}W)")
        print(f"Usage: CPU avg={avg_cpu_usage:.1f}%, GPU avg={avg_gpu_usage:.1f}%")
        print(
            f"RAM: avg={avg_ram_used:.1f}GB ({avg_ram_percent:.1f}%), "
            f"max={max_ram_used:.1f}GB ({max_ram_percent:.1f}%)"
        )
        print("=" * 60)

    # Plot if requested
    if args.graph:
        plot_metrics(metrics_history)

    # Clean up
    stop_macmon()
