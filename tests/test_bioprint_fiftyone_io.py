from pathlib import Path

from bioprint_data.config import DatasetConfig
from bioprint_data.fiftyone_io import (
    create_or_load_dataset,
    ensure_dataset_dirs,
    ensure_dataset_schema,
)


class FakeDataset:
    def __init__(self):
        self.fields = {}
        self.added_fields = []

    def add_sample_field(self, name, field_type):
        self.added_fields.append(name)
        self.fields[name] = field_type


class FakeFiftyOne:
    class FloatField:
        pass

    class StringField:
        pass

    def __init__(self, existing=False):
        self.existing = existing
        self.created = []
        self.loaded = []
        self.deleted = []
        self.datasets = {}

    def dataset_exists(self, name):
        return self.existing

    def Dataset(self, name, **kwargs):
        dataset = FakeDataset()
        dataset.name = name
        self.created.append((name, kwargs))
        self.datasets[name] = dataset
        self.existing = True
        return dataset

    def load_dataset(self, name):
        self.loaded.append(name)
        dataset = self.datasets.get(name) or FakeDataset()
        dataset.name = name
        self.datasets[name] = dataset
        return dataset

    def delete_dataset(self, name):
        self.deleted.append(name)
        self.datasets.pop(name, None)
        self.existing = False


def make_config(tmp_path: Path, *, overwrite: bool = False) -> DatasetConfig:
    return DatasetConfig(
        config_path=tmp_path / "configs" / "dataset.yaml",
        project_root=tmp_path,
        dataset_name="bioprint_test",
        overwrite=overwrite,
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
    import bioprint_data.fiftyone_io as fiftyone_io

    monkeypatch.setattr(fiftyone_io, "_fo", lambda: FakeFiftyOne)
    dataset = FakeDataset()
    config = make_config(tmp_path)

    ensure_dataset_schema(dataset, config)

    assert dataset.fields["motion_speed"].__name__ == "FloatField"
    assert dataset.fields["curvature_radius"].__name__ == "FloatField"
    assert dataset.fields["print_path"].__name__ == "StringField"
    assert dataset.fields["label"].__name__ == "StringField"


def test_create_or_load_dataset_creates_persistent_dataset(tmp_path, monkeypatch):
    import bioprint_data.fiftyone_io as fiftyone_io

    fake_fo = FakeFiftyOne(existing=False)
    monkeypatch.setattr(fiftyone_io, "_fo", lambda: fake_fo)
    config = make_config(tmp_path)

    dataset = create_or_load_dataset(config)

    assert dataset.name == "bioprint_test"
    assert fake_fo.created == [("bioprint_test", {"persistent": True})]
    assert fake_fo.loaded == []
    assert fake_fo.deleted == []


def test_create_or_load_dataset_loads_existing_without_overwrite(tmp_path, monkeypatch):
    import bioprint_data.fiftyone_io as fiftyone_io

    fake_fo = FakeFiftyOne(existing=True)
    existing_dataset = FakeDataset()
    existing_dataset.name = "bioprint_test"
    fake_fo.datasets["bioprint_test"] = existing_dataset
    monkeypatch.setattr(fiftyone_io, "_fo", lambda: fake_fo)
    config = make_config(tmp_path)

    dataset = create_or_load_dataset(config)

    assert dataset is existing_dataset
    assert fake_fo.loaded == ["bioprint_test"]
    assert fake_fo.deleted == []
    assert fake_fo.created == []


def test_create_or_load_dataset_deletes_and_recreates_persistent_with_overwrite(
    tmp_path, monkeypatch
):
    import bioprint_data.fiftyone_io as fiftyone_io

    fake_fo = FakeFiftyOne(existing=True)
    monkeypatch.setattr(fiftyone_io, "_fo", lambda: fake_fo)
    config = make_config(tmp_path, overwrite=True)

    dataset = create_or_load_dataset(config)

    assert dataset.name == "bioprint_test"
    assert fake_fo.deleted == ["bioprint_test"]
    assert fake_fo.created == [("bioprint_test", {"persistent": True})]
    assert fake_fo.loaded == []


def test_create_or_load_dataset_loads_when_concurrent_create_already_exists(
    tmp_path, monkeypatch
):
    import bioprint_data.fiftyone_io as fiftyone_io

    class ConcurrentCreateFiftyOne(FakeFiftyOne):
        def __init__(self):
            super().__init__(existing=False)
            self.exists_checks = 0
            self.loaded_dataset = FakeDataset()
            self.loaded_dataset.name = "bioprint_test"

        def dataset_exists(self, name):
            self.exists_checks += 1
            return self.exists_checks > 1

        def Dataset(self, name, **kwargs):
            self.created.append((name, kwargs))
            raise RuntimeError("already exists")

        def load_dataset(self, name):
            self.loaded.append(name)
            return self.loaded_dataset

    fake_fo = ConcurrentCreateFiftyOne()
    monkeypatch.setattr(fiftyone_io, "_fo", lambda: fake_fo)
    config = make_config(tmp_path)

    dataset = create_or_load_dataset(config)

    assert dataset is fake_fo.loaded_dataset
    assert fake_fo.created == [("bioprint_test", {"persistent": True})]
    assert fake_fo.loaded == ["bioprint_test"]
    assert fake_fo.loaded_dataset.added_fields == [
        "motion_speed",
        "curvature_radius",
        "print_path",
        "label",
    ]


def test_ensure_dataset_schema_skips_existing_fields_from_schema(tmp_path, monkeypatch):
    import bioprint_data.fiftyone_io as fiftyone_io

    class DatasetWithSchema(FakeDataset):
        def get_field_schema(self):
            return {"motion_speed": FakeFiftyOne.FloatField}

    monkeypatch.setattr(fiftyone_io, "_fo", lambda: FakeFiftyOne)
    dataset = DatasetWithSchema()
    config = make_config(tmp_path)

    ensure_dataset_schema(dataset, config)

    assert "motion_speed" not in dataset.added_fields
    assert dataset.added_fields == ["curvature_radius", "print_path", "label"]
