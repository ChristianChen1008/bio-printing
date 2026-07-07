from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


class ConfigError(ValueError):
    """Raised when the dataset configuration is invalid."""


@dataclass(frozen=True)
class DatasetConfig:
    config_path: Path
    project_root: Path
    dataset_name: str
    overwrite: bool
    images_dir: Path
    metadata_dir: Path
    processed_dir: Path
    image_path_column: str
    target_column: str
    numeric_fields: list[str]
    categorical_fields: list[str]
    field_definitions: dict[str, Any]


def load_dataset_config(config_path: str | Path) -> DatasetConfig:
    resolved_config_path = Path(config_path).resolve()

    try:
        config_text = resolved_config_path.read_text(encoding="utf-8")
        raw_config = (
            yaml.safe_load(config_text)
            if yaml is not None
            else _safe_load_basic_yaml(config_text)
        )
    except OSError as exc:
        raise ConfigError(f"Unable to read config file: {resolved_config_path}") from exc
    except _yaml_error_types() as exc:
        raise ConfigError(f"Invalid YAML in config file: {resolved_config_path}") from exc

    if not isinstance(raw_config, dict):
        raise ConfigError("Dataset config must be a mapping")

    dataset = _required_mapping(raw_config, "dataset")
    paths = _required_mapping(raw_config, "paths")
    columns = _required_mapping(raw_config, "columns")
    fields = _required_mapping(raw_config, "fields")

    dataset_name = _required_non_empty_string(dataset, "name", "dataset.name")
    image_path_column = _required_non_empty_string(
        columns, "image_path", "columns.image_path"
    )
    target_column = _required_non_empty_string(columns, "target", "columns.target")
    numeric_fields = _required_mapping(fields, "numeric")
    categorical_fields = _required_mapping(fields, "categorical")

    project_root = (
        resolved_config_path.parent.parent
        if resolved_config_path.parent.name == "configs"
        else resolved_config_path.parent
    )

    return DatasetConfig(
        config_path=resolved_config_path,
        project_root=project_root,
        dataset_name=dataset_name,
        overwrite=bool(dataset.get("overwrite", False)),
        images_dir=_resolve_path(project_root, paths.get("images_dir")),
        metadata_dir=_resolve_path(project_root, paths.get("metadata_dir")),
        processed_dir=_resolve_path(project_root, paths.get("processed_dir")),
        image_path_column=image_path_column,
        target_column=target_column,
        numeric_fields=list(numeric_fields),
        categorical_fields=list(categorical_fields),
        field_definitions=dict(fields),
    )


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if key not in config:
        raise ConfigError(f"Missing required config section: {key}")
    if not isinstance(value, dict):
        raise ConfigError(f"Config section must be a mapping: {key}")
    return value


def _required_non_empty_string(
    config: dict[str, Any], key: str, display_name: str
) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Config value must be a non-empty string: {display_name}")
    return value


def _resolve_path(project_root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("Config path values must be non-empty strings")

    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _yaml_error_types() -> tuple[type[Exception], ...]:
    if yaml is None:
        return (ValueError,)
    return (yaml.YAMLError,)


def _safe_load_basic_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, result)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Invalid YAML line: {raw_line}")

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        while indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if raw_value == "":
            value: dict[str, Any] = {}
            parent[key] = value
            stack.append((indent, value))
        else:
            parent[key] = _parse_basic_yaml_value(raw_value)

    return result


def _parse_basic_yaml_value(value: str) -> Any:
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    return value
