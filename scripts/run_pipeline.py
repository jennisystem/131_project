"""Run the full basketball shot tracking pipeline.

Examples:
    # use defaults from config
    python scripts/run_pipeline.py --config configs/default.yaml

    # override the input video
    python scripts/run_pipeline.py --video data/raw/my_shot.mp4

    # override both video and output directory
    python scripts/run_pipeline.py --video data/raw/my_shot.mp4 \
                                   --output-dir outputs/my_shot_run1
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
    parser.add_argument("--video", default=None,
                        help="Path to input video (overrides config video.input_path).")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (overrides config video.output_dir). "
                             "If --video is given without --output-dir, defaults to outputs/<video_stem>/.")
    args = parser.parse_args()
    run_pipeline(args.config, video_path=args.video, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
