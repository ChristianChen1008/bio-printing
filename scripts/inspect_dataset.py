from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bioprint_data.config import load_dataset_config
from bioprint_data.fiftyone_io import create_or_load_dataset, dataset_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the configured FiftyOne dataset."
    )
    parser.add_argument(
        "--config",
        default="configs/dataset.yaml",
        help="Path to the dataset configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(Path(args.config))
    dataset = create_or_load_dataset(config)
    summary = dataset_summary(dataset, config)

    print(f"Dataset: {summary['name']}")
    print(f"Samples: {summary['sample_count']}")
    print("Fields:")
    for field_name in summary["numeric_fields"]:
        print(f"  - {field_name}: numeric")
    for field_name in summary["categorical_fields"]:
        print(f"  - {field_name}: categorical")
    print(f"  - {summary['target_column']}: target")


if __name__ == "__main__":
    main()
