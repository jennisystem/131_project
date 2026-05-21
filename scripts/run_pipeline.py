"""Run the full basketball shot tracking pipeline.

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from basketball_tracker.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the basketball shot tracker pipeline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
