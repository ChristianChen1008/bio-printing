from pathlib import Path

from src.bioprint_data.config import DatasetConfig
from src.bioprint_data.fiftyone_io import ensure_dataset_dirs, ensure_dataset_schema


class FakeDataset:
    def __init__(self):
        self.fields = {}

    def add_sample_field(self, name, field_type):
        self.fields[name] = field_type


class FakeFiftyOne:
    class FloatField:
        pass

    class StringField:
        pass


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
        numeric_fields=["motion_speed", "curvature_radius"],
        categorical_fields=["print_path"],
        field_definitions={},
    )


def test_ensure_dataset_dirs_creates_expected_directories(tmp_path):
    config = make_config(tmp_path)

    ensure_dataset_dirs(config)

    assert config.images_dir.is_dir()
    assert config.metadata_dir.is_dir()
    assert config.processed_dir.is_dir()


def test_ensure_dataset_schema_adds_configured_fields(tmp_path, monkeypatch):
    import src.bioprint_data.fiftyone_io as fiftyone_io

    monkeypatch.setattr(fiftyone_io, "_fo", lambda: FakeFiftyOne)
    dataset = FakeDataset()
    config = make_config(tmp_path)

    ensure_dataset_schema(dataset, config)

    assert dataset.fields["motion_speed"].__name__ == "FloatField"
    assert dataset.fields["curvature_radius"].__name__ == "FloatField"
    assert dataset.fields["print_path"].__name__ == "StringField"
    assert dataset.fields["label"].__name__ == "StringField"
