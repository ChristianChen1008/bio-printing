from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bioprint_data.config import load_dataset_config
from bioprint_data.fiftyone_io import ensure_dataset_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the configured FiftyOne dataset for metadata import."
    )
    parser.add_argument(
        "--config",
        default="configs/dataset.yaml",
        help="Path to the dataset configuration file.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to a future CSV or Excel metadata table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(Path(args.config))
    ensure_dataset_dirs(config)

    if args.metadata is None:
        print("No metadata table provided. Dataset schema is ready for future import.")
        print(f"Expected image path column: {config.image_path_column}")
        print(f"Numeric fields: {', '.join(config.numeric_fields)}")
        print(f"Categorical fields: {', '.join(config.categorical_fields)}")
        return

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata table does not exist: {metadata_path}")

    raise NotImplementedError(
        "Metadata import will be implemented after a real CSV/Excel format is available."
    )


if __name__ == "__main__":
    main()
