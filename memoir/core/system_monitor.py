"""
System monitoring for adaptive processing.

This module monitors battery, CPU/GPU temperatures, power, and usage to determine
when it's appropriate to run ML inference.

Uses macmon for real-time temperature and power monitoring on Apple Silicon.
"""

import subprocess
import re
import time
import platform
import json
import atexit
import select
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


# Path to macmon binary (relative to this file)
MACMON_PATH = (
    Path(__file__).parent.parent.parent
    / "testing"
    / "benchmarking"
    / "macmon"
    / "target"
    / "aarch64-apple-darwin"
    / "release"
    / "macmon"
)

# Global macmon process handle
_macmon_proc = None


def apple_silicon():
    """Check if running on Apple Silicon."""
    return platform.machine() in ("arm64", "aarch64")


def start_macmon():
    """Start macmon process if not already running."""
    global _macmon_proc

    if _macmon_proc is not None and _macmon_proc.poll() is None:
        return  # Already running

    if not apple_silicon():
        return

    if not MACMON_PATH.exists():
        print(f"Warning: macmon not found at {MACMON_PATH}")
        return

    # Start macmon in pipe mode with 250ms sampling interval
    _macmon_proc = subprocess.Popen(
        [str(MACMON_PATH), "pipe", "-i", "250"],
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
        try:
            _macmon_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _macmon_proc.kill()
        _macmon_proc = None


def get_macmon_metrics(fresh=False) -> Optional[Dict]:
    """
    Get current system metrics from macmon.

    Args:
        fresh: If True, drain stale buffered data and return only the most recent metrics

        Returns:
        Dictionary with temperature, power, and usage data, or None if unavailable
    """
    if not apple_silicon():
        return None

    # Ensure macmon is running
    start_macmon()

    if _macmon_proc is None:
        return None

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
            if _macmon_proc:
                line = _macmon_proc.stdout.readline()
            else:
                return None

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
            ecpu_usage = data.get("ecpu_usage", [0, 0])
            pcpu_usage = data.get("pcpu_usage", [0, 0])
            gpu_usage_data = data.get("gpu_usage", [0, 0])

            # Average E-core and P-core usage for overall CPU usage
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
    except (json.JSONDecodeError, AttributeError, IndexError) as e:
        # Silently handle errors - this runs frequently
        pass

    return None


def read_battery() -> tuple[int, bool]:
    """
    Read battery percentage and charging state using pmset.

    Returns:
        Tuple of (battery_percentage, is_charging)
    """
    try:
        out = subprocess.check_output(
            ["pmset", "-g", "batt"],
            text=True,
            timeout=2,
        )

        # Parse percentage
        pct_match = re.search(r"(\d+)%", out)
        pct = int(pct_match.group(1)) if pct_match else 100

        # Check if charging (AC Power or actively charging)
        charging = ("AC Power" in out) or (
            "charging" in out.lower() and "discharging" not in out.lower()
        )

        return pct, charging

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # Assume fully charged and on AC if we can't read battery
        return 100, True


class SystemMonitor:
    """Monitor system resources and provide processing recommendations."""

    def __init__(
        self,
        cache_seconds: int = 30,
        verbose: bool = False,
        # Temperature thresholds (Celsius)
        gpu_temp_max: float = 80.0,
        gpu_temp_target: float = 60.0,
        cpu_temp_max: float = 85.0,
        cpu_temp_target: float = 65.0,
        # Battery thresholds
        battery_min_pct: int = 20,
        battery_comfortable_pct: int = 50,
    ):
        """
        Initialize the system monitor.

        Args:
            cache_seconds: Cache monitoring results for this many seconds
            verbose: If True, print debug messages
            gpu_temp_max: Max safe GPU temperature (°C)
            gpu_temp_target: Target GPU temperature for comfortable operation (°C)
            cpu_temp_max: Max safe CPU temperature (°C)
            cpu_temp_target: Target CPU temperature for comfortable operation (°C)
            battery_min_pct: Minimum battery percentage before pausing
            battery_comfortable_pct: Battery percentage for comfortable operation
        """
        self.cache_seconds = cache_seconds
        self.verbose = verbose

        # Temperature thresholds
        self.gpu_temp_max = gpu_temp_max
        self.gpu_temp_target = gpu_temp_target
        self.cpu_temp_max = cpu_temp_max
        self.cpu_temp_target = cpu_temp_target

        # Battery thresholds
        self.battery_min_pct = battery_min_pct
        self.battery_comfortable_pct = battery_comfortable_pct

        # Cache
        self._cached_data: Optional[Dict] = None
        self._cache_time: Optional[datetime] = None

        # Start macmon if on Apple Silicon
        if apple_silicon():
            start_macmon()

    def _log(self, message: str):
        """Log message if verbose is enabled."""
        if self.verbose:
            print(f"[SystemMonitor] {message}")

    def _compute_score(
        self,
        macmon_data: Optional[Dict],
        battery_pct: int,
        charging: bool,
    ) -> float:
        """
        Compute a processing score based on system state.

        Higher score = safer to run inference.

        Args:
            macmon_data: Temperature, power, and usage data from macmon
            battery_pct: Battery percentage (0-100)
            charging: Whether device is charging

        Returns:
            Score between -1.0 and 1.0
        """
        # Start with neutral score
        score = 0.0

        # Battery factor (most important when not charging)
        if charging:
            score += 0.4  # Big boost when plugged in
        else:
            if battery_pct < self.battery_min_pct:
                score -= 0.6  # Critical battery
            elif battery_pct < self.battery_comfortable_pct:
                score -= 0.2  # Low battery
            else:
                # Scale from 0 to 0.2 based on battery level
                battery_norm = (battery_pct - self.battery_comfortable_pct) / (
                    100 - self.battery_comfortable_pct
                )
                score += battery_norm * 0.2

        # If we don't have macmon data, be conservative
        if not macmon_data:
            self._log("No macmon data available, being conservative")
            return score - 0.3

        # GPU temperature factor (very important)
        gpu_temp = macmon_data.get("gpu_temp")
        if gpu_temp is not None:
            if gpu_temp > self.gpu_temp_max:
                score -= 0.7  # Too hot!
            elif gpu_temp > self.gpu_temp_target:
                # Linear penalty as temp rises above target
                temp_factor = (gpu_temp - self.gpu_temp_target) / (
                    self.gpu_temp_max - self.gpu_temp_target
                )
                score -= temp_factor * 0.5
            else:
                # Bonus for cool GPU
                temp_bonus = (self.gpu_temp_target - gpu_temp) / self.gpu_temp_target
                score += temp_bonus * 0.2

        # CPU temperature factor
        cpu_temp = macmon_data.get("cpu_temp")
        if cpu_temp is not None:
            if cpu_temp > self.cpu_temp_max:
                score -= 0.4  # Too hot!
            elif cpu_temp > self.cpu_temp_target:
                temp_factor = (cpu_temp - self.cpu_temp_target) / (
                    self.cpu_temp_max - self.cpu_temp_target
                )
                score -= temp_factor * 0.3

        # GPU usage factor (high usage means something else is using it)
        gpu_usage = macmon_data.get("gpu_usage")
        if gpu_usage is not None and gpu_usage > 50:
            usage_factor = (gpu_usage - 50) / 50
            score -= usage_factor * 0.2

        # Power consumption factor (less important but still relevant)
        gpu_power = macmon_data.get("gpu_power")
        if gpu_power is not None and gpu_power > 10.0:  # 10W threshold
            power_factor = min((gpu_power - 10.0) / 20.0, 1.0)
            score -= power_factor * 0.15

        # Clamp to [-1.0, 1.0]
        return max(min(score, 1.0), -1.0)

    def _recommend_action(self, score: float) -> str:
        """
        Convert score to recommended action.

        Args:
            score: Processing score

        Returns:
            "RUN", "SLOW", or "PAUSE"
        """
        if score > 0.3:
            return "RUN"
        elif score > -0.1:
            return "SLOW"
        else:
            return "PAUSE"

    def get_system_state(self, use_cache: bool = True) -> Optional[Dict]:
        """
        Get current system state with temperature, power, battery, and usage info.

        Args:
            use_cache: If True, use cached data if available and fresh

        Returns:
            Dictionary with system state or None if failed
        """
        # Check cache
        if use_cache and self._cached_data and self._cache_time:
            age = (datetime.now() - self._cache_time).total_seconds()
            if age < self.cache_seconds:
                return self._cached_data

        # Read fresh data
        macmon_data = get_macmon_metrics(fresh=True)
        battery_pct, charging = read_battery()

        # Compute score and recommendation
        score = self._compute_score(macmon_data, battery_pct, charging)
        action = self._recommend_action(score)

        # Compile state
        state = {
            "battery_pct": battery_pct,
            "charging": charging,
            "score": round(score, 3),
            "recommendation": action,
            "timestamp": datetime.now().isoformat(),
        }

        # Add macmon data if available
        if macmon_data:
            state.update(
                {
                    "cpu_temp": macmon_data.get("cpu_temp"),
                    "gpu_temp": macmon_data.get("gpu_temp"),
                    "cpu_power_w": macmon_data.get("cpu_power"),
                    "gpu_power_w": macmon_data.get("gpu_power"),
                    "total_power_w": macmon_data.get("total_power"),
                    "cpu_usage": macmon_data.get("cpu_usage"),
                    "gpu_usage": macmon_data.get("gpu_usage"),
                    "ram_used_gb": macmon_data.get("ram_used_gb"),
                    "ram_total_gb": macmon_data.get("ram_total_gb"),
                    "ram_usage_percent": macmon_data.get("ram_usage_percent"),
                }
            )
        else:
            # Add None values for missing data
            state.update(
                {
                    "cpu_temp": None,
                    "gpu_temp": None,
                    "cpu_power_w": None,
                    "gpu_power_w": None,
                    "total_power_w": None,
                    "cpu_usage": None,
                    "gpu_usage": None,
                    "ram_used_gb": None,
                    "ram_total_gb": None,
                    "ram_usage_percent": None,
                }
            )

        # Update cache
        self._cached_data = state
        self._cache_time = datetime.now()

        return state

    def get_recommendation(self, use_cache: bool = True) -> str:
        """
        Get processing recommendation.

        Args:
            use_cache: If True, use cached data if available

        Returns:
            "RUN", "SLOW", or "PAUSE"
        """
        state = self.get_system_state(use_cache=use_cache)
        if state is None:
            # If we can't read system state, be conservative
            return "SLOW"
        return state["recommendation"]

    def is_on_battery(self, use_cache: bool = True) -> bool:
        """
        Check if system is running on battery power.

        Args:
            use_cache: If True, use cached data if available

        Returns:
            True if on battery, False if charging/plugged in
        """
        state = self.get_system_state(use_cache=use_cache)
        if state is None:
            # If we can't read state, assume on battery (be conservative)
            return True
        return not state["charging"]

    def wait_for_cooldown(
        self,
        target_gpu_temp: float = 60.0,
        check_interval: float = 2.0,
        max_wait: float = 300.0,
    ) -> float:
        """
        Wait for GPU temperature to drop below target.

        Args:
            target_gpu_temp: Target GPU temperature in Celsius
            check_interval: Seconds between checks
            max_wait: Maximum seconds to wait

        Returns:
            Time spent waiting in seconds
        """
        if not apple_silicon():
            self._log("Not on Apple Silicon, skipping cooldown wait")
            return 0.0

        start_time = time.time()
        self._log(f"Waiting for GPU temp to drop below {target_gpu_temp}°C...")

        consecutive_failures = 0
        max_failures = 5

        while True:
            # Timeout check
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                self._log(f"Timeout after {elapsed:.1f}s, proceeding anyway")
                return elapsed

            # Get fresh metrics
            macmon_data = get_macmon_metrics(fresh=True)
            if macmon_data and macmon_data.get("gpu_temp") is not None:
                consecutive_failures = 0
                gpu_temp = macmon_data["gpu_temp"]
                self._log(f"Current GPU temp: {gpu_temp:.1f}°C")

                if gpu_temp < target_gpu_temp:
                    elapsed = time.time() - start_time
                    self._log(f"GPU cooled to {gpu_temp:.1f}°C (waited {elapsed:.1f}s)")
                    return elapsed
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    self._log(
                        f"Failed to get metrics {max_failures} times, proceeding anyway"
                    )
                    return time.time() - start_time

            time.sleep(check_interval)

    def print_state(self):
        """Print current system state (for debugging)."""
        state = self.get_system_state(use_cache=False)
        if not state:
            print("Failed to read system state")
            return

        print("\n" + "=" * 70)
        print("System Monitor State")
        print("=" * 70)

        # Battery
        charging_str = "(charging)" if state["charging"] else "(on battery)"
        print(f"Battery:         {state['battery_pct']}% {charging_str}")

        # Temperatures
        if state.get("cpu_temp") is not None:
            print(f"CPU Temp:        {state['cpu_temp']:.1f}°C")
        if state.get("gpu_temp") is not None:
            print(f"GPU Temp:        {state['gpu_temp']:.1f}°C")

        # Power
        if state.get("cpu_power_w") is not None:
            print(f"CPU Power:       {state['cpu_power_w']:.3f} W")
        if state.get("gpu_power_w") is not None:
            print(f"GPU Power:       {state['gpu_power_w']:.3f} W")
        if state.get("total_power_w") is not None:
            print(f"Total Power:     {state['total_power_w']:.3f} W")

        # Usage
        if state.get("cpu_usage") is not None:
            print(f"CPU Usage:       {state['cpu_usage']:.1f}%")
        if state.get("gpu_usage") is not None:
            print(f"GPU Usage:       {state['gpu_usage']:.1f}%")

        # RAM
        if (
            state.get("ram_used_gb") is not None
            and state.get("ram_total_gb") is not None
        ):
            print(
                f"RAM:             {state['ram_used_gb']:.1f}/{state['ram_total_gb']:.1f} GB "
                f"({state['ram_usage_percent']:.1f}%)"
            )

        # Score and recommendation
        print(f"\nScore:           {state['score']:.3f}")
        print(f"Recommendation:  {state['recommendation']}")
        print("=" * 70 + "\n")


# Global singleton instance
_global_monitor: Optional[SystemMonitor] = None


def get_global_monitor(
    cache_seconds: int = 30,
    verbose: bool = False,
    **kwargs,
) -> SystemMonitor:
    """
    Get or create the global system monitor instance.

    Args:
        cache_seconds: Cache monitoring results for this many seconds
        verbose: If True, print debug messages
        **kwargs: Additional arguments passed to SystemMonitor constructor

    Returns:
        SystemMonitor instance
    """
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = SystemMonitor(
            cache_seconds=cache_seconds,
            verbose=verbose,
            **kwargs,
        )

    return _global_monitor


def get_processing_recommendation() -> str:
    """
    Convenience function to get processing recommendation.

    Returns:
        "RUN", "SLOW", or "PAUSE"
    """
    monitor = get_global_monitor()
    return monitor.get_recommendation()


def is_on_battery() -> bool:
    """
    Convenience function to check if on battery power.

    Returns:
        True if on battery, False if charging
    """
    monitor = get_global_monitor()
    return monitor.is_on_battery()
