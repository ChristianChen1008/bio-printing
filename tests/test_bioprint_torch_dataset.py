from pathlib import Path

import pytest

from bioprint_data.config import DatasetConfig
from bioprint_data.torch_dataset import FiftyOneBioprintDataset


class FakeSample:
    def __init__(self, filepath, values):
        self.filepath = filepath
        self._values = values

    def get_field(self, name):
        return self._values.get(name)


def make_config(tmp_path: Path) -> DatasetConfig:
    return DatasetConfig(
        config_path=tmp_path / "configs" / "dataset.yaml",
        project_root=tmp_path,
        dataset_name="bioprint_test",
        overwrite=False,
        images_dir=tmp_path / "data" / "raw" / "images",
        metadata_dir=tmp_path / "data" / "raw" / "metadata",
        processed_dir=tmp_path / "data" / "processed",
        image_path_column="image_path",
        target_column="label",
        numeric_fields=["motion_speed", "print_height"],
        categorical_fields=["print_path"],
        field_definitions={},
    )


def test_torch_dataset_returns_expected_sample_dict(tmp_path):
    sample = FakeSample(
        "image.png",
        {
            "motion_speed": 10.0,
            "print_height": 0.5,
            "print_path": "arc",
            "label": "success",
        },
    )
    dataset = FiftyOneBioprintDataset(
        [sample], make_config(tmp_path), image_loader=lambda path: f"loaded:{path}"
    )

    item = dataset[0]

    assert item["image"] == "loaded:image.png"
    assert item["numeric_features"] == [10.0, 0.5]
    assert item["categorical_features"] == {"print_path": "arc"}
    assert item["target"] == "success"
    assert item["filepath"] == "image.png"


def test_torch_dataset_raises_clear_error_for_missing_numeric_field(tmp_path):
    sample = FakeSample(
        "image.png",
        {
            "motion_speed": 10.0,
            "print_path": "arc",
            "label": "success",
        },
    )
    dataset = FiftyOneBioprintDataset(
        [sample], make_config(tmp_path), image_loader=lambda path: f"loaded:{path}"
    )

    with pytest.raises(ValueError, match="print_height.*image.png"):
        dataset[0]
