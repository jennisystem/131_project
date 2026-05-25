"""Run the tracker for all sample videos using their per-clip configs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from basketball_tracker.pipeline import run_pipeline


CONFIGS = [
    "configs/sample_shot1.yaml",
    "configs/sample_shot2.yaml",
    "configs/sample_shot3.yaml",
]


def main() -> None:
    for config_path in CONFIGS:
        print(f"\nRunning {config_path}")
        run_pipeline(config_path)


if __name__ == "__main__":
    main()
