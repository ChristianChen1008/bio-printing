from pathlib import Path

import pytest

from src.bioprint_data.config import ConfigError, load_dataset_config


def write_config(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_load_dataset_config_resolves_paths(tmp_path):
    config_path = tmp_path / "dataset.yaml"
    write_config(
        config_path,
        """
dataset:
  name: bioprint_test
  overwrite: false
paths:
  images_dir: data/raw/images
  metadata_dir: data/raw/metadata
  processed_dir: data/processed
columns:
  image_path: image_path
  target: label
fields:
  numeric:
    motion_speed:
      display_name: motion speed
      unit: mm_s
  categorical:
    print_path:
      display_name: print path
""",
    )

    config = load_dataset_config(config_path)

    assert config.dataset_name == "bioprint_test"
    assert config.overwrite is False
    assert config.project_root == tmp_path
    assert config.images_dir == tmp_path / "data" / "raw" / "images"
    assert config.metadata_dir == tmp_path / "data" / "raw" / "metadata"
    assert config.processed_dir == tmp_path / "data" / "processed"
    assert config.image_path_column == "image_path"
    assert config.target_column == "label"
    assert config.numeric_fields == ["motion_speed"]
    assert config.categorical_fields == ["print_path"]


def test_load_dataset_config_rejects_missing_sections(tmp_path):
    config_path = tmp_path / "dataset.yaml"
    write_config(
        config_path,
        """
dataset:
  name: broken
""",
    )

    with pytest.raises(ConfigError, match="Missing required config section"):
        load_dataset_config(config_path)
