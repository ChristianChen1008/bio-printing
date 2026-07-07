from __future__ import annotations

from typing import Any

from .config import DatasetConfig


def _fo():
    import fiftyone as fo

    return fo


def ensure_dataset_dirs(config: DatasetConfig) -> None:
    config.images_dir.mkdir(parents=True, exist_ok=True)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)


def create_or_load_dataset(config: DatasetConfig) -> Any:
    fo = _fo()
    if fo.dataset_exists(config.dataset_name):
        if config.overwrite:
            fo.delete_dataset(config.dataset_name)
        else:
            dataset = fo.load_dataset(config.dataset_name)
            ensure_dataset_schema(dataset, config)
            return dataset

    try:
        dataset = fo.Dataset(config.dataset_name, persistent=True)
    except Exception:
        if not config.overwrite and fo.dataset_exists(config.dataset_name):
            dataset = fo.load_dataset(config.dataset_name)
            ensure_dataset_schema(dataset, config)
            return dataset
        raise
    ensure_dataset_schema(dataset, config)
    return dataset


def load_existing_dataset(config: DatasetConfig) -> Any:
    fo = _fo()
    if not fo.dataset_exists(config.dataset_name):
        raise ValueError(
            f"FiftyOne dataset does not exist: {config.dataset_name}. "
            "Run scripts/init_fiftyone_dataset.py first."
        )
    dataset = fo.load_dataset(config.dataset_name)
    ensure_dataset_schema(dataset, config)
    return dataset


def ensure_dataset_schema(dataset: Any, config: DatasetConfig) -> None:
    fo = _fo()
    existing_fields = set()
    if hasattr(dataset, "get_field_schema"):
        existing_fields = set(dataset.get_field_schema())

    for field_name in config.numeric_fields:
        if field_name not in existing_fields:
            dataset.add_sample_field(field_name, fo.FloatField)
    for field_name in config.categorical_fields:
        if field_name not in existing_fields:
            dataset.add_sample_field(field_name, fo.StringField)
    if config.target_column and config.target_column not in existing_fields:
        dataset.add_sample_field(config.target_column, fo.StringField)


def dataset_summary(dataset: Any, config: DatasetConfig) -> dict[str, Any]:
    return {
        "name": dataset.name,
        "sample_count": len(dataset),
        "numeric_fields": config.numeric_fields,
        "categorical_fields": config.categorical_fields,
        "target_column": config.target_column,
    }
