import argparse
import csv
import json
import os
import threading
import time
from statistics import fmean
from typing import Dict, List, Optional

import psutil
from PIL import Image, ImageOps

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


def list_images(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)
    )


def resize_image_if_needed(img: Image.Image, max_side: Optional[int]) -> Image.Image:
    if not max_side:
        return img
    width, height = img.size
    longest = max(width, height)
    if longest <= max_side:
        return img

    scale = max_side / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


MB = 1024 * 1024


def mem_stats_mb(proc: psutil.Process) -> Dict[str, float]:
    info = proc.memory_info()
    stats = {
        "rss_mb": info.rss / MB,
        "vms_mb": info.vms / MB,
    }

    for attr in ("shared", "text", "data"):
        value = getattr(info, attr, None)
        if value is not None:
            stats[f"{attr}_mb"] = value / MB

    return {k: round(v, 2) for k, v in stats.items()}


def log_mem(label: str, proc: psutil.Process) -> Dict[str, float]:
    stats = mem_stats_mb(proc)
    print(f"{label}: {stats}")
    return stats


class ResourceSampler:
    def __init__(self, proc: psutil.Process, sample_interval: float):
        self.proc = proc
        self.sample_interval = max(sample_interval, 0.05)
        self.samples: List[Dict[str, float]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        cpu_count = psutil.cpu_count(logical=True) or 1
        self._cpu_normalizer = float(cpu_count)

    def start(self):
        self.proc.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()

    def _run(self):
        while not self._stop_event.wait(self.sample_interval):
            self.samples.append(self._collect_sample())

    def _collect_sample(self) -> Dict[str, float]:
        raw_cpu = self.proc.cpu_percent(interval=None)
        cpu_pct = raw_cpu / self._cpu_normalizer
        return {
            "timestamp": time.perf_counter(),
            "cpu_pct": cpu_pct,
            "rss": self.proc.memory_info().rss,
            "mem_pct": self.proc.memory_percent(),
            "gpu_active": mx.get_active_memory(),
            "gpu_cache": mx.get_cache_memory(),
        }

    def summary(self) -> Dict[str, Optional[float]]:
        if not self.samples:
            return {}

        def _stat(key: str, fn) -> Optional[float]:
            vals = [s[key] for s in self.samples if s.get(key) is not None]
            return fn(vals) if vals else None

        return {
            "sample_count": len(self.samples),
            "cpu_pct_avg": _stat("cpu_pct", fmean),
            "cpu_pct_max": _stat("cpu_pct", max),
            "rss_avg_bytes": _stat("rss", fmean),
            "rss_max_bytes": _stat("rss", max),
            "mem_pct_avg": _stat("mem_pct", fmean),
            "mem_pct_max": _stat("mem_pct", max),
            "gpu_active_avg_bytes": _stat("gpu_active", fmean),
            "gpu_active_max_bytes": _stat("gpu_active", max),
            "gpu_cache_avg_bytes": _stat("gpu_cache", fmean),
            "gpu_cache_max_bytes": _stat("gpu_cache", max),
        }


def run_call(
    model,
    processor,
    config,
    images: List[str],
    prompt: str,
    max_new_tokens: int,
    proc: psutil.Process,
    csv_writer,
    call_idx: int,
    max_image_side: Optional[int],
    output_writer,
    sample_interval: float,
):
    """
    Run a single generate() call on one or more images, time it,
    and log basic load metrics.
    """

    def load_image(path: str) -> Image.Image:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return resize_image_if_needed(img, max_image_side)

    # You can also pass PIL.Image objects instead of paths:
    imgs = [load_image(p) for p in images]
    # imgs = images  # for MLX-VLM, list of paths is fine

    formatted_prompt = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=len(imgs),
    )

    # psutil: CPU% since last call; first call will be 0-ish
    cpu_before = proc.cpu_percent(interval=None)
    mem_before = proc.memory_info().rss
    mem_pct_before = proc.memory_percent()
    gpu_active_before = mx.get_active_memory()
    gpu_cache_before = mx.get_cache_memory()
    mx.reset_peak_memory()
    sampler = ResourceSampler(proc, sample_interval)
    sampler.start()

    start = time.perf_counter()
    try:
        _output = generate(
            model,
            processor,
            formatted_prompt,
            imgs,
            max_tokens=max_new_tokens,
            verbose=False,
        )
    finally:
        sampler.stop()
    end = time.perf_counter()

    cpu_after = proc.cpu_percent(interval=None)
    mem_after = proc.memory_info().rss
    mem_pct_after = proc.memory_percent()
    gpu_active_after = mx.get_active_memory()
    gpu_cache_after = mx.get_cache_memory()
    gpu_peak = mx.get_peak_memory()
    sampler_stats = sampler.summary()

    latency = end - start

    csv_writer.writerow(
        {
            "call_index": call_idx,
            "num_images": len(imgs),
            "start_time": start,
            "end_time": end,
            "latency_sec": latency,
            "cpu_before_pct": cpu_before,
            "cpu_after_pct": cpu_after,
            "rss_before_bytes": mem_before,
            "rss_after_bytes": mem_after,
            "mem_before_pct": mem_pct_before,
            "mem_after_pct": mem_pct_after,
            "gpu_active_before_bytes": gpu_active_before,
            "gpu_active_after_bytes": gpu_active_after,
            "gpu_cache_before_bytes": gpu_cache_before,
            "gpu_cache_after_bytes": gpu_cache_after,
            "gpu_peak_bytes": gpu_peak,
            "resource_sample_count": (
                sampler_stats.get("sample_count") if sampler_stats else None
            ),
            "cpu_pct_avg": sampler_stats.get("cpu_pct_avg") if sampler_stats else None,
            "cpu_pct_max": sampler_stats.get("cpu_pct_max") if sampler_stats else None,
            "rss_avg_bytes": (
                sampler_stats.get("rss_avg_bytes") if sampler_stats else None
            ),
            "rss_max_bytes": (
                sampler_stats.get("rss_max_bytes") if sampler_stats else None
            ),
            "mem_pct_avg": sampler_stats.get("mem_pct_avg") if sampler_stats else None,
            "mem_pct_max": sampler_stats.get("mem_pct_max") if sampler_stats else None,
            "gpu_active_avg_bytes": (
                sampler_stats.get("gpu_active_avg_bytes") if sampler_stats else None
            ),
            "gpu_active_max_bytes": (
                sampler_stats.get("gpu_active_max_bytes") if sampler_stats else None
            ),
            "gpu_cache_avg_bytes": (
                sampler_stats.get("gpu_cache_avg_bytes") if sampler_stats else None
            ),
            "gpu_cache_max_bytes": (
                sampler_stats.get("gpu_cache_max_bytes") if sampler_stats else None
            ),
        }
    )

    if output_writer is not None:
        record = {
            "call_index": call_idx,
            "images": images,
            "prompt": prompt,
            "output": _output.text,
        }
        output_writer.write(json.dumps(record))
        output_writer.write("\n")

    return latency


def benchmark_sequential(
    model,
    processor,
    config,
    images: List[str],
    prompt: str,
    max_new_tokens: int,
    sleep_between: float,
    csv_writer,
    max_image_side: Optional[int],
    output_writer,
    sample_interval: float,
    proc: psutil.Process,
    call_idx_offset: int = 0,
):
    latencies = []
    for i, img in enumerate(images):
        log_mem(f"\n--- Call {i + call_idx_offset} BEFORE ---", proc)
        latency = run_call(
            model,
            processor,
            config,
            [img],
            prompt,
            max_new_tokens,
            proc,
            csv_writer,
            call_idx=i + call_idx_offset,
            max_image_side=max_image_side,
            output_writer=output_writer,
            sample_interval=sample_interval,
        )
        latencies.append(latency)
        log_mem(f"--- Call {i + call_idx_offset} AFTER ---", proc)
        if sleep_between > 0:
            time.sleep(sleep_between)
    return latencies


def benchmark_batch(
    model,
    processor,
    config,
    images: List[str],
    prompt: str,
    max_new_tokens: int,
    batch_size: int,
    sleep_between: float,
    csv_writer,
    max_image_side: Optional[int],
    output_writer,
    sample_interval: float,
    proc: psutil.Process,
    call_idx_offset: int = 0,
):
    latencies = []
    call_idx = call_idx_offset
    n = len(images)

    for start_idx in range(0, n, batch_size):
        batch = images[start_idx : start_idx + batch_size]
        log_mem(f"\n--- Batch {call_idx} BEFORE ---", proc)
        latency = run_call(
            model,
            processor,
            config,
            batch,
            prompt,
            max_new_tokens,
            proc,
            csv_writer,
            call_idx=call_idx,
            max_image_side=max_image_side,
            output_writer=output_writer,
            sample_interval=sample_interval,
        )
        latencies.append(latency)
        log_mem(f"--- Batch {call_idx} AFTER ---", proc)
        call_idx += 1
        if sleep_between > 0:
            time.sleep(sleep_between)
    return latencies


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-VL-2B MLX-8bit on Mac"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lmstudio-community/Qwen3-VL-2B-Instruct-MLX-8bit",
        help="HF model repo or local path",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=False,
        help="Directory containing images (not required for --idle-mode)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        help="Prompt to use for all images (not required for --idle-mode)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sequential", "batch"],
        default="sequential",
        help="sequential: one image per call; batch: many per call",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for batch mode",
    )
    parser.add_argument(
        "--sleep-between",
        type=float,
        default=0.0,
        help="Seconds to sleep between calls (0 = pile it on)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max tokens to generate per call",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1024,
        help="Largest allowed image dimension in pixels (images are downscaled if larger)",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="benchmark_results.csv",
        help="Where to write per-call metrics",
    )
    parser.add_argument(
        "--outputs-file",
        type=str,
        default="benchmark_outputs.jsonl",
        help="Where to write raw model outputs (JSONL)",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.25,
        help="Sampling interval (seconds) for CPU/GPU/RAM usage during each call",
    )
    parser.add_argument(
        "--stress-test",
        action="store_true",
        help="Run in stress test mode (repeat indefinitely until interrupted)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the benchmark (default: 1, ignored if --stress-test is set)",
    )
    parser.add_argument(
        "--iteration-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between iterations in stress test mode",
    )
    parser.add_argument(
        "--idle-mode",
        action="store_true",
        help="Load the model and hold it in memory without running inference (for testing idle resource usage)",
    )
    parser.add_argument(
        "--idle-duration",
        type=float,
        default=None,
        help="How long to keep model idle in seconds (default: indefinite, use Ctrl+C to stop)",
    )
    parser.add_argument(
        "--idle-sample-interval",
        type=float,
        default=5.0,
        help="Interval in seconds for sampling resources in idle mode (default: 5.0)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.idle_mode:
        if not args.image_dir:
            parser.error("--image-dir is required unless using --idle-mode")
        if not args.prompt:
            parser.error("--prompt is required unless using --idle-mode")
        images = list_images(args.image_dir)
        if not images:
            raise SystemExit(f"No images found in {args.image_dir}")
        print(f"Found {len(images)} images.")
    else:
        images = []

    proc = psutil.Process(os.getpid())

    print(f"Loading model {args.model}…")
    log_mem("Before load", proc)
    model, processor = load(args.model)
    # Qwen example in mlx-vlm uses model.config; some examples use load_config()
    try:
        config = model.config
    except AttributeError:
        config = load_config(args.model)

    log_mem("After load", proc)

    # Check if we're in idle mode
    if args.idle_mode:
        print("\n" + "=" * 60)
        print("IDLE MODE: Model loaded and held in memory")
        print("=" * 60)
        if args.idle_duration:
            print(
                f"Will hold for {args.idle_duration} seconds (or press Ctrl+C to stop)"
            )
        else:
            print("Will hold indefinitely (press Ctrl+C to stop)")
        print(f"Sampling resources every {args.idle_sample_interval} seconds")
        print("=" * 60 + "\n")

        # Create a sampler for idle monitoring
        idle_sampler = ResourceSampler(proc, args.idle_sample_interval)
        idle_sampler.start()

        start_time = time.perf_counter()
        try:
            if args.idle_duration:
                # Sleep for specified duration
                time.sleep(args.idle_duration)
            else:
                # Wait indefinitely
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nIdle mode interrupted by user (Ctrl+C)")
        finally:
            idle_sampler.stop()
            end_time = time.perf_counter()

        # Report idle statistics
        idle_duration = end_time - start_time
        stats = idle_sampler.summary()

        print("\n" + "=" * 60)
        print("=== IDLE MODE SUMMARY ===")
        print("=" * 60)
        print(f"Duration:           {idle_duration:.2f} seconds")
        print(f"Samples collected:  {stats.get('sample_count', 0)}")

        print(f"\nCPU Usage:")
        print(f"  Average:          {stats.get('cpu_pct_avg', 0):.2f}%")
        print(f"  Peak:             {stats.get('cpu_pct_max', 0):.2f}%")

        print(f"\nMemory (RSS):")
        rss_avg_mb = (stats.get("rss_avg_bytes", 0) or 0) / MB
        rss_max_mb = (stats.get("rss_max_bytes", 0) or 0) / MB
        print(f"  Average:          {rss_avg_mb:.2f} MB")
        print(f"  Peak:             {rss_max_mb:.2f} MB")

        print(f"\nMemory %:")
        print(f"  Average:          {stats.get('mem_pct_avg', 0):.2f}%")
        print(f"  Peak:             {stats.get('mem_pct_max', 0):.2f}%")

        print(f"\nGPU Active Memory:")
        gpu_active_avg_mb = (stats.get("gpu_active_avg_bytes", 0) or 0) / MB
        gpu_active_max_mb = (stats.get("gpu_active_max_bytes", 0) or 0) / MB
        print(f"  Average:          {gpu_active_avg_mb:.2f} MB")
        print(f"  Peak:             {gpu_active_max_mb:.2f} MB")

        print(f"\nGPU Cache Memory:")
        gpu_cache_avg_mb = (stats.get("gpu_cache_avg_bytes", 0) or 0) / MB
        gpu_cache_max_mb = (stats.get("gpu_cache_max_bytes", 0) or 0) / MB
        print(f"  Average:          {gpu_cache_avg_mb:.2f} MB")
        print(f"  Peak:             {gpu_cache_max_mb:.2f} MB")

        # Write idle stats to CSV if desired
        with open(args.out_csv, "w", newline="") as f:
            fieldnames = [
                "mode",
                "duration_sec",
                "sample_count",
                "cpu_pct_avg",
                "cpu_pct_max",
                "rss_avg_bytes",
                "rss_max_bytes",
                "mem_pct_avg",
                "mem_pct_max",
                "gpu_active_avg_bytes",
                "gpu_active_max_bytes",
                "gpu_cache_avg_bytes",
                "gpu_cache_max_bytes",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"mode": "idle", "duration_sec": idle_duration, **stats})

        print(f"\nIdle stats written to: {args.out_csv}")
        print("=" * 60)
        return

    # Determine if we're in stress test mode
    if args.stress_test:
        print("Model loaded. Starting STRESS TEST mode (press Ctrl+C to stop)…")
        iterations = float("inf")
    else:
        iterations = args.repeat
        if iterations > 1:
            print(f"Model loaded. Starting benchmark with {iterations} iterations…")
        else:
            print("Model loaded. Starting benchmark…")

    t0 = time.perf_counter()
    all_latencies = []
    iteration_count = 0
    call_idx_offset = 0

    try:
        with open(args.out_csv, "w", newline="") as f, open(
            args.outputs_file, "w"
        ) as outputs_file:
            fieldnames = [
                "iteration",
                "call_index",
                "num_images",
                "start_time",
                "end_time",
                "latency_sec",
                "cpu_before_pct",
                "cpu_after_pct",
                "rss_before_bytes",
                "rss_after_bytes",
                "mem_before_pct",
                "mem_after_pct",
                "gpu_active_before_bytes",
                "gpu_active_after_bytes",
                "gpu_cache_before_bytes",
                "gpu_cache_after_bytes",
                "gpu_peak_bytes",
                "resource_sample_count",
                "cpu_pct_avg",
                "cpu_pct_max",
                "rss_avg_bytes",
                "rss_max_bytes",
                "mem_pct_avg",
                "mem_pct_max",
                "gpu_active_avg_bytes",
                "gpu_active_max_bytes",
                "gpu_cache_avg_bytes",
                "gpu_cache_max_bytes",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Wrap the writer to inject iteration number
            class IterationWriter:
                def __init__(self, writer, iteration):
                    self.writer = writer
                    self.iteration = iteration

                def writerow(self, row_dict):
                    row_dict["iteration"] = self.iteration
                    self.writer.writerow(row_dict)

            while iteration_count < iterations:
                iteration_count += 1
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration_count}")
                print(f"{'='*60}")

                iteration_writer = IterationWriter(writer, iteration_count)

                if args.mode == "sequential":
                    latencies = benchmark_sequential(
                        model,
                        processor,
                        config,
                        images,
                        args.prompt,
                        args.max_new_tokens,
                        args.sleep_between,
                        iteration_writer,
                        args.max_image_side,
                        outputs_file,
                        args.sample_interval,
                        proc,
                        call_idx_offset=call_idx_offset,
                    )
                else:
                    latencies = benchmark_batch(
                        model,
                        processor,
                        config,
                        images,
                        args.prompt,
                        args.max_new_tokens,
                        args.batch_size,
                        args.sleep_between,
                        iteration_writer,
                        args.max_image_side,
                        outputs_file,
                        args.sample_interval,
                        proc,
                        call_idx_offset=call_idx_offset,
                    )

                all_latencies.extend(latencies)
                call_idx_offset += len(latencies)

                # Print iteration summary
                iter_mean = sum(latencies) / len(latencies)
                print(f"\nIteration {iteration_count} complete:")
                print(f"  Calls: {len(latencies)}")
                print(f"  Mean latency: {iter_mean:.3f}s")
                print(f"  Min: {min(latencies):.3f}s, Max: {max(latencies):.3f}s")

                # Sleep between iterations if requested
                if args.iteration_delay > 0 and iteration_count < iterations:
                    print(
                        f"Sleeping for {args.iteration_delay}s before next iteration…"
                    )
                    time.sleep(args.iteration_delay)

    except KeyboardInterrupt:
        print("\n\nStress test interrupted by user (Ctrl+C)")
        if not all_latencies:
            print("No data collected, exiting.")
            return

    t1 = time.perf_counter()

    total_calls = len(all_latencies)
    total_images = len(images) * iteration_count
    total_time = t1 - t0

    latencies_sorted = sorted(all_latencies)
    mean_lat = sum(all_latencies) / len(all_latencies)
    median_lat = latencies_sorted[len(latencies_sorted) // 2]

    print(f"\n{'='*60}")
    print(f"=== Benchmark summary ===")
    print(f"{'='*60}")
    print(f"Mode:            {args.mode}")
    print(f"Model:           {args.model}")
    print(f"Iterations:      {iteration_count}")
    print(f"Images per iter: {len(images)}")
    print(f"Images total:    {total_images}")
    print(f"Calls total:     {total_calls}")
    print(f"Total wall time: {total_time:.3f} s")
    print(
        f"Latency (s):     min={min(all_latencies):.3f}  "
        f"max={max(all_latencies):.3f}  "
        f"mean={mean_lat:.3f}  "
        f"median={median_lat:.3f}"
    )
    print(f"Throughput:      {total_images / total_time:.3f} images/sec")
    print(f"CSV written to:  {args.out_csv}")


if __name__ == "__main__":
    main()
