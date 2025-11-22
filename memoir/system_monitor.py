"""
System monitoring for adaptive processing.

This module monitors battery, CPU/GPU power, and thermal state to determine
when it's appropriate to run ML inference.
"""

import subprocess
import re
import time
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta


class SystemMonitor:
    """Monitor system resources and provide processing recommendations."""
    
    def __init__(self, cache_seconds: int = 30, verbose: bool = False):
        """
        Initialize the system monitor.
        
        Args:
            cache_seconds: Cache monitoring results for this many seconds
            verbose: If True, print debug messages
        """
        self.cache_seconds = cache_seconds
        self.verbose = verbose
        
        # Cache
        self._cached_data: Optional[Dict] = None
        self._cache_time: Optional[datetime] = None
    
    def _log(self, message: str):
        """Log message if verbose is enabled."""
        if self.verbose:
            print(f"[SystemMonitor] {message}")
    
    def _read_powermetrics(self) -> Optional[Dict[str, float]]:
        """
        Read CPU/GPU/ANE power and thermal pressure using powermetrics.
        
        Requires sudo permissions.
        
        Returns:
            Dictionary with power and thermal data, or None if failed
        """
        try:
            out = subprocess.check_output(
                [
                    "sudo",
                    "powermetrics",
                    "--samplers",
                    "cpu_power,gpu_power,ane_power,thermal",
                    "--format",
                    "text",
                    "-n",
                    "1",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self._log(f"powermetrics failed: {e}")
            return None
        
        data = {}
        
        # Power regexes (values are in mW)
        cpu_match = re.search(r"CPU Power:\s+([\d\.]+)\s*mW", out)
        gpu_match = re.search(r"GPU Power:\s+([\d\.]+)\s*mW", out)
        ane_match = re.search(r"ANE Power:\s+([\d\.]+)\s*mW", out)
        
        # Convert mW to W
        data["cpu"] = float(cpu_match.group(1)) / 1000.0 if cpu_match else 0.0
        data["gpu"] = float(gpu_match.group(1)) / 1000.0 if gpu_match else 0.0
        data["ane"] = float(ane_match.group(1)) / 1000.0 if ane_match else 0.0
        data["soc_power"] = data["cpu"] + data["gpu"] + data["ane"]
        
        # Thermal pressure
        thermal_match = re.search(r"Current pressure level:\s+(\w+)", out)
        data["thermal_pressure"] = thermal_match.group(1) if thermal_match else "Unknown"
        
        return data
    
    def _read_battery(self) -> Tuple[int, bool]:
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
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self._log(f"pmset failed: {e}")
            # Assume fully charged and on AC if we can't read battery
            return 100, True
    
    def _compute_score(
        self, power_data: Dict[str, float], battery_pct: int, charging: bool
    ) -> float:
        """
        Compute a processing score based on system state.
        
        Higher score = safer to run inference.
        
        Args:
            power_data: Power and thermal data
            battery_pct: Battery percentage (0-100)
            charging: Whether device is charging
            
        Returns:
            Score between -1.0 and 1.0
        """
        # Normalize inputs
        battery_factor = battery_pct / 100.0
        gpu_power_factor = min(power_data["gpu"] / 0.012, 1.0)  # 12 mW = moderate load
        
        # Thermal pressure mapping
        thermal_map = {
            "Nominal": 0.0,
            "Moderate": 0.3,
            "Heavy": 0.7,
            "Trapping": 1.0,
        }
        thermal_factor = thermal_map.get(power_data["thermal_pressure"], 0.5)
        
        # Compute score
        score = (
            battery_factor * 1.0  # more battery = better
            - gpu_power_factor * 0.6  # heavy GPU load = worse
            - thermal_factor * 0.3  # high thermal pressure = worse
            + (0.5 if charging else -0.2)  # charging is much better
        )
        
        return max(min(score, 1.0), -1.0)
    
    def _recommend_action(self, score: float) -> str:
        """
        Convert score to recommended action.
        
        Args:
            score: Processing score
            
        Returns:
            "RUN", "SLOW", or "PAUSE"
        """
        if score > 0.4:
            return "RUN"
        elif score > 0.1:
            return "SLOW"
        else:
            return "PAUSE"
    
    def get_system_state(self, use_cache: bool = True) -> Optional[Dict]:
        """
        Get current system state with power, battery, and thermal info.
        
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
        power_data = self._read_powermetrics()
        if power_data is None:
            # If powermetrics fails, return a safe default
            self._log("Failed to read powermetrics, using safe defaults")
            power_data = {
                "cpu": 0.0,
                "gpu": 0.0,
                "ane": 0.0,
                "soc_power": 0.0,
                "thermal_pressure": "Unknown",
            }
        
        battery_pct, charging = self._read_battery()
        
        # Compute score and recommendation
        score = self._compute_score(power_data, battery_pct, charging)
        action = self._recommend_action(score)
        
        # Compile state
        state = {
            "battery_pct": battery_pct,
            "charging": charging,
            "cpu_power_w": power_data["cpu"],
            "gpu_power_w": power_data["gpu"],
            "ane_power_w": power_data["ane"],
            "soc_power_w": power_data["soc_power"],
            "thermal_pressure": power_data["thermal_pressure"],
            "score": round(score, 3),
            "recommendation": action,
            "timestamp": datetime.now().isoformat(),
        }
        
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
    
    def print_state(self):
        """Print current system state (for debugging)."""
        state = self.get_system_state(use_cache=False)
        if not state:
            print("Failed to read system state")
            return
        
        print("\n" + "=" * 60)
        print("System Monitor State")
        print("=" * 60)
        print(f"Battery:         {state['battery_pct']}% {'(charging)' if state['charging'] else '(on battery)'}")
        print(f"CPU Power:       {state['cpu_power_w']:.3f} W")
        print(f"GPU Power:       {state['gpu_power_w']:.3f} W")
        print(f"ANE Power:       {state['ane_power_w']:.3f} W")
        print(f"Total SoC:       {state['soc_power_w']:.3f} W")
        print(f"Thermal:         {state['thermal_pressure']}")
        print(f"\nScore:           {state['score']:.3f}")
        print(f"Recommendation:  {state['recommendation']}")
        print("=" * 60 + "\n")


# Global singleton instance
_global_monitor: Optional[SystemMonitor] = None


def get_global_monitor(cache_seconds: int = 30, verbose: bool = False) -> SystemMonitor:
    """
    Get or create the global system monitor instance.
    
    Args:
        cache_seconds: Cache monitoring results for this many seconds
        verbose: If True, print debug messages
        
    Returns:
        SystemMonitor instance
    """
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = SystemMonitor(
            cache_seconds=cache_seconds,
            verbose=verbose,
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




