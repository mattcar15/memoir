#!/usr/bin/env python3
"""
Screenshot Memory System - Entry Point
Captures screenshots at regular intervals, processes them with Ollama via LangChain,
and logs the results with timestamps. Also provides API endpoints for retrieving snapshots.
"""

import argparse
import sys
from pathlib import Path

from memoir.cli import run
from memoir.runner import run_both


def main():
    """Main entry point with mode selection."""
    # If no arguments provided, default to both mode
    if len(sys.argv) == 1:
        sys.argv.append("both")

    # Use the CLI directly
    run()


if __name__ == "__main__":
    main()
