import subprocess
import re
import time
import argparse

# ---------------------------------------------------------
# Helpers to read system data
# ---------------------------------------------------------


def read_powermetrics():
    """
    Reads CPU/GPU/ANE power + thermal pressure using powermetrics.
    CPU Power is the total power across all CPU cores/clusters.
    Returns a dictionary with cleaned numeric fields.
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
        )
    except subprocess.CalledProcessError as e:
        print("powermetrics failed:", e)
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

    # Temperature - thermal sampler shows pressure level, not exact temps
    # Extract thermal pressure instead
    thermal_match = re.search(r"Current pressure level:\s+(\w+)", out)
    data["thermal_pressure"] = thermal_match.group(1) if thermal_match else "Unknown"

    return data


def read_battery():
    """
    Reads battery percentage + charging state using pmset.
    """
    out = subprocess.check_output(["pmset", "-g", "batt"], text=True)
    pct = int(re.search(r"(\d+)%", out).group(1))
    # Check if drawing from AC Power (not Battery Power) or actively charging
    charging = ("AC Power" in out) or (
        "charging" in out.lower() and "discharging" not in out.lower()
    )
    return pct, charging


# ---------------------------------------------------------
# Decision logic
# ---------------------------------------------------------


def compute_score(data, battery_pct, charging):
    """
    Combines battery, thermal pressure, and GPU power into a single score.
    Higher score => safer to run inference.
    """

    # Normalize inputs
    battery_factor = battery_pct / 100.0
    gpu_power_factor = min(data["gpu"] / 0.012, 1.0)  # 12 mW = 0.012 W = moderate load

    # Thermal pressure mapping
    thermal_map = {"Nominal": 0.0, "Moderate": 0.3, "Heavy": 0.7, "Trapping": 1.0}
    thermal_factor = thermal_map.get(data["thermal_pressure"], 0.5)

    score = (
        battery_factor * 1.0  # more battery = better
        - gpu_power_factor * 0.6  # heavy GPU load = worse
        - thermal_factor * 0.3  # high thermal pressure = worse
        + (0.5 if charging else -0.2)  # charging is much better
    )

    return max(min(score, 1.0), -1.0)


def recommend_action(score):
    """
    Score → recommended inference behavior.
    """
    if score > 0.4:
        return "RUN"
    elif score > 0.1:
        return "SLOW"
    else:
        return "PAUSE"


# ---------------------------------------------------------
# Main monitoring logic
# ---------------------------------------------------------


def run_monitor():
    """Run a single monitoring check and print results."""
    # Read everything
    data = read_powermetrics()
    if data is None:
        print(
            "Failed to read powermetrics. Make sure to run with sudo or check permissions."
        )
        return False

    battery_pct, charging = read_battery()

    # Compute score
    score = compute_score(data, battery_pct, charging)
    action = recommend_action(score)

    # Print results
    print(
        "Battery:        ",
        battery_pct,
        "%",
        "(charging)" if charging else "(discharging)",
    )
    print("CPU Power (total):", data["cpu"], "W")
    print("GPU Power:       ", data["gpu"], "W")
    print("ANE Power:       ", data["ane"], "W")
    print("Total SoC:       ", data["soc_power"], "W")
    print()
    print("Thermal Pressure:", data["thermal_pressure"])
    print()
    print("Score:          ", round(score, 3))
    print("Recommendation: ", action)
    print()
    return True


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor system power and thermal status"
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=None,
        help="Run continuously at specified interval (seconds). If not set, runs once.",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=None,
        help="Number of times to run (default: infinite when using --interval)",
    )

    args = parser.parse_args()

    print("\n=== System Monitor ===\n")

    if args.interval is None:
        # Run once
        run_monitor()
    else:
        # Run continuously
        print(f"Monitoring every {args.interval}s (Ctrl+C to stop)\n")
        count = 0
        try:
            while True:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}]")

                if not run_monitor():
                    break

                count += 1
                if args.count is not None and count >= args.count:
                    print(f"Completed {count} checks.")
                    break

                print(f"Waiting {args.interval}s...\n")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n\nStopped after {count} checks.")
            exit(0)
