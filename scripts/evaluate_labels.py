"""Compare pipeline detections against manually labeled centers.

Usage:
    python scripts/evaluate_labels.py \\
        --detections outputs/sample_shot/detections.csv \\
        --labels data/labels/sample_labels.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from basketball_tracker.metrics import load_labels, evaluate_against_labels


def load_detections_csv(path: str) -> list[dict]:
    detections = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row["x"]) if row.get("x") else None
            y = float(row["y"]) if row.get("y") else None
            detections.append({
                "frame_idx": int(row["frame_idx"]),
                "x": x,
                "y": y,
                "status": row.get("status", ""),
            })
    return detections


def main():
    parser = argparse.ArgumentParser(description="Evaluate detections against manual labels.")
    parser.add_argument("--detections", required=True, help="Path to detections CSV.")
    parser.add_argument("--labels", required=True, help="Path to labels CSV.")
    args = parser.parse_args()

    detections = load_detections_csv(args.detections)
    labels = load_labels(args.labels)
    result = evaluate_against_labels(detections, labels)

    print("\nEvaluation Results")
    print("==================")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
