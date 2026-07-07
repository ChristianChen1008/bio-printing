# FiftyOne Dataset Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a configuration-driven FiftyOne dataset framework that can initialize an empty bioprinting experiment dataset and later import images plus experiment parameters.

**Architecture:** Keep configuration loading, FiftyOne operations, command-line scripts, and PyTorch access in separate focused files. The first working slice initializes directories and an empty FiftyOne dataset schema without requiring real images or metadata.

**Tech Stack:** Python, PyYAML, FiftyOne, pytest/unittest-compatible tests, optional PyTorch and Pillow for the later dataset wrapper

---

## File Structure

- Create: `configs/dataset.yaml`
  - Store dataset name, local paths, column names, numeric fields, categorical fields, and target field.
- Create: `data/raw/images/.gitkeep`
  - Preserve the future raw image directory.
- Create: `data/raw/metadata/.gitkeep`
  - Preserve the future metadata table directory.
- Create: `data/processed/.gitkeep`
  - Preserve the future processed-data directory.
- Create: `src/bioprint_data/__init__.py`
  - Export stable public helpers.
- Create: `src/bioprint_data/config.py`
  - Load and validate YAML configuration.
- Create: `src/bioprint_data/fiftyone_io.py`
  - Create/load FiftyOne datasets and apply schema.
- Create: `src/bioprint_data/torch_dataset.py`
  - Provide a small PyTorch Dataset wrapper around FiftyOne samples.
- Create: `scripts/init_fiftyone_dataset.py`
  - Initialize the empty dataset from config.
- Create: `scripts/import_to_fiftyone.py`
  - Validate future import inputs and exit cleanly when no metadata exists.
- Create: `scripts/inspect_dataset.py`
  - Print dataset summary and schema.
- Create: `tests/test_bioprint_config.py`
  - Test config loading and validation without FiftyOne.
- Create: `tests/test_bioprint_fiftyone_io.py`
  - Test FiftyOne schema code with a fake in-memory dataset object.
- Create: `tests/test_bioprint_torch_dataset.py`
  - Test PyTorch wrapper sample conversion without requiring torch import at module import time.

---

### Task 1: Add Dataset Config and Loader

**Files:**
- Create: `configs/dataset.yaml`
- Create: `src/bioprint_data/__init__.py`
- Create: `src/bioprint_data/config.py`
- Test: `tests/test_bioprint_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bioprint_config.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_bioprint_config.py -v
```

Expected: FAIL because `src.bioprint_data.config` does not exist.

- [ ] **Step 3: Add the default YAML config**

Create `configs/dataset.yaml`:

```yaml
dataset:
  name: bioprint_experiments
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
      unit: null
    extrusion_speed:
      display_name: extrusion speed
      unit: null
    material_concentration:
      display_name: material concentration
      unit: null
    print_height:
      display_name: print height
      unit: null
    curvature_radius:
      display_name: curvature radius
      unit: null
  categorical:
    print_path:
      display_name: print path
```

- [ ] **Step 4: Add package exports**

Create `src/bioprint_data/__init__.py`:

```python
from .config import DatasetConfig, ConfigError, load_dataset_config

__all__ = ["DatasetConfig", "ConfigError", "load_dataset_config"]
```

- [ ] **Step 5: Implement config loading**

Create `src/bioprint_data/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the dataset configuration is missing required structure."""


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
    target_column: str | None
    numeric_fields: list[str]
    categorical_fields: list[str]
    field_definitions: dict[str, Any]


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing required config section: {key}")
    return value


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def load_dataset_config(config_path: str | Path) -> DatasetConfig:
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    if not isinstance(data, dict):
        raise ConfigError("Config file must contain a YAML mapping")

    dataset = _require_mapping(data, "dataset")
    paths = _require_mapping(data, "paths")
    columns = _require_mapping(data, "columns")
    fields = _require_mapping(data, "fields")

    dataset_name = dataset.get("name")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ConfigError("dataset.name must be a non-empty string")

    image_path_column = columns.get("image_path")
    if not isinstance(image_path_column, str) or not image_path_column.strip():
        raise ConfigError("columns.image_path must be a non-empty string")

    numeric = fields.get("numeric", {})
    categorical = fields.get("categorical", {})
    if not isinstance(numeric, dict):
        raise ConfigError("fields.numeric must be a mapping")
    if not isinstance(categorical, dict):
        raise ConfigError("fields.categorical must be a mapping")

    project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent

    return DatasetConfig(
        config_path=config_path,
        project_root=project_root,
        dataset_name=dataset_name,
        overwrite=bool(dataset.get("overwrite", False)),
        images_dir=_resolve_path(project_root, str(paths.get("images_dir", ""))),
        metadata_dir=_resolve_path(project_root, str(paths.get("metadata_dir", ""))),
        processed_dir=_resolve_path(project_root, str(paths.get("processed_dir", ""))),
        image_path_column=image_path_column,
        target_column=columns.get("target"),
        numeric_fields=list(numeric.keys()),
        categorical_fields=list(categorical.keys()),
        field_definitions=fields,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_bioprint_config.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```powershell
git add configs/dataset.yaml src/bioprint_data/__init__.py src/bioprint_data/config.py tests/test_bioprint_config.py
git commit -m "feat: add bioprint dataset config loader"
```

---

### Task 2: Add FiftyOne Dataset Initialization Helpers

**Files:**
- Create: `src/bioprint_data/fiftyone_io.py`
- Test: `tests/test_bioprint_fiftyone_io.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bioprint_fiftyone_io.py`:

```python
from pathlib import Path

from src.bioprint_data.config import DatasetConfig
from src.bioprint_data.fiftyone_io import ensure_dataset_dirs, ensure_dataset_schema


class FakeDataset:
    def __init__(self):
        self.fields = {}

    def add_sample_field(self, name, field_type):
        self.fields[name] = field_type


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


def test_ensure_dataset_schema_adds_configured_fields(tmp_path):
    dataset = FakeDataset()
    config = make_config(tmp_path)

    ensure_dataset_schema(dataset, config)

    assert dataset.fields["motion_speed"].__name__ == "FloatField"
    assert dataset.fields["curvature_radius"].__name__ == "FloatField"
    assert dataset.fields["print_path"].__name__ == "StringField"
    assert dataset.fields["label"].__name__ == "StringField"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_bioprint_fiftyone_io.py -v
```

Expected: FAIL because `src.bioprint_data.fiftyone_io` does not exist.

- [ ] **Step 3: Implement FiftyOne helpers**

Create `src/bioprint_data/fiftyone_io.py`:

```python
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

    dataset = fo.Dataset(config.dataset_name)
    ensure_dataset_schema(dataset, config)
    return dataset


def ensure_dataset_schema(dataset: Any, config: DatasetConfig) -> None:
    fo = _fo()
    for field_name in config.numeric_fields:
        dataset.add_sample_field(field_name, fo.FloatField)
    for field_name in config.categorical_fields:
        dataset.add_sample_field(field_name, fo.StringField)
    if config.target_column:
        dataset.add_sample_field(config.target_column, fo.StringField)


def dataset_summary(dataset: Any, config: DatasetConfig) -> dict[str, Any]:
    return {
        "name": dataset.name,
        "sample_count": len(dataset),
        "numeric_fields": config.numeric_fields,
        "categorical_fields": config.categorical_fields,
        "target_column": config.target_column,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_bioprint_fiftyone_io.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/bioprint_data/fiftyone_io.py tests/test_bioprint_fiftyone_io.py
git commit -m "feat: add FiftyOne dataset helpers"
```

---

### Task 3: Add Initialization and Inspection Scripts

**Files:**
- Create: `scripts/init_fiftyone_dataset.py`
- Create: `scripts/inspect_dataset.py`
- Create: `data/raw/images/.gitkeep`
- Create: `data/raw/metadata/.gitkeep`
- Create: `data/processed/.gitkeep`

- [ ] **Step 1: Create preserved data directories**

Create empty files:

```text
data/raw/images/.gitkeep
data/raw/metadata/.gitkeep
data/processed/.gitkeep
```

- [ ] **Step 2: Add the init script**

Create `scripts/init_fiftyone_dataset.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from src.bioprint_data.config import load_dataset_config
from src.bioprint_data.fiftyone_io import create_or_load_dataset, dataset_summary, ensure_dataset_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the bioprint FiftyOne dataset.")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Path to dataset YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(Path(args.config))
    ensure_dataset_dirs(config)
    dataset = create_or_load_dataset(config)
    summary = dataset_summary(dataset, config)

    print(f"Dataset: {summary['name']}")
    print(f"Samples: {summary['sample_count']}")
    print(f"Numeric fields: {', '.join(summary['numeric_fields'])}")
    print(f"Categorical fields: {', '.join(summary['categorical_fields'])}")
    print(f"Target column: {summary['target_column']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add the inspect script**

Create `scripts/inspect_dataset.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from src.bioprint_data.config import load_dataset_config
from src.bioprint_data.fiftyone_io import create_or_load_dataset, dataset_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the configured FiftyOne dataset.")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Path to dataset YAML config.")
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
    if summary["target_column"]:
        print(f"  - {summary['target_column']}: target")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run syntax verification**

Run:

```powershell
python -m py_compile scripts/init_fiftyone_dataset.py scripts/inspect_dataset.py
```

Expected: exit code 0.

- [ ] **Step 5: Run config and helper tests**

Run:

```powershell
python -m pytest tests/test_bioprint_config.py tests/test_bioprint_fiftyone_io.py -v
```

Expected: all tests passed.

- [ ] **Step 6: Commit**

```powershell
git add scripts/init_fiftyone_dataset.py scripts/inspect_dataset.py data/raw/images/.gitkeep data/raw/metadata/.gitkeep data/processed/.gitkeep
git commit -m "feat: add FiftyOne initialization scripts"
```

---

### Task 4: Add Future Import Script Interface

**Files:**
- Create: `scripts/import_to_fiftyone.py`

- [ ] **Step 1: Add import script with clean no-data behavior**

Create `scripts/import_to_fiftyone.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from src.bioprint_data.config import load_dataset_config
from src.bioprint_data.fiftyone_io import create_or_load_dataset, ensure_dataset_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import image metadata into the configured FiftyOne dataset.")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Path to dataset YAML config.")
    parser.add_argument("--metadata", default=None, help="Optional CSV or Excel metadata table path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(Path(args.config))
    ensure_dataset_dirs(config)
    create_or_load_dataset(config)

    metadata_path = Path(args.metadata).resolve() if args.metadata else None
    if metadata_path is None:
        print("No metadata table provided. Dataset schema is ready for future import.")
        print(f"Expected image path column: {config.image_path_column}")
        print(f"Numeric fields: {', '.join(config.numeric_fields)}")
        print(f"Categorical fields: {', '.join(config.categorical_fields)}")
        return

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata table does not exist: {metadata_path}")

    raise NotImplementedError(
        "Metadata import will be implemented after the real CSV or Excel table format is available."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run syntax verification**

Run:

```powershell
python -m py_compile scripts/import_to_fiftyone.py
```

Expected: exit code 0.

- [ ] **Step 3: Commit**

```powershell
git add scripts/import_to_fiftyone.py
git commit -m "feat: add metadata import command interface"
```

---

### Task 5: Add PyTorch Dataset Skeleton

**Files:**
- Create: `src/bioprint_data/torch_dataset.py`
- Test: `tests/test_bioprint_torch_dataset.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bioprint_torch_dataset.py`:

```python
from pathlib import Path

from src.bioprint_data.config import DatasetConfig
from src.bioprint_data.torch_dataset import FiftyOneBioprintDataset


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
    dataset = FiftyOneBioprintDataset([sample], make_config(tmp_path), image_loader=lambda path: f"loaded:{path}")

    item = dataset[0]

    assert item["image"] == "loaded:image.png"
    assert item["numeric_features"] == [10.0, 0.5]
    assert item["categorical_features"] == {"print_path": "arc"}
    assert item["target"] == "success"
    assert item["filepath"] == "image.png"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_bioprint_torch_dataset.py -v
```

Expected: FAIL because `src.bioprint_data.torch_dataset` does not exist.

- [ ] **Step 3: Implement the PyTorch Dataset skeleton**

Create `src/bioprint_data/torch_dataset.py`:

```python
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .config import DatasetConfig


def default_image_loader(filepath: str) -> Any:
    from PIL import Image

    return Image.open(filepath).convert("RGB")


class FiftyOneBioprintDataset:
    def __init__(
        self,
        samples: Sequence[Any],
        config: DatasetConfig,
        image_loader: Callable[[str], Any] | None = None,
        image_transform: Callable[[Any], Any] | None = None,
    ):
        self.samples = list(samples)
        self.config = config
        self.image_loader = image_loader or default_image_loader
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = self.image_loader(sample.filepath)
        if self.image_transform is not None:
            image = self.image_transform(image)

        numeric_features = [
            float(sample.get_field(field_name))
            for field_name in self.config.numeric_fields
        ]
        categorical_features = {
            field_name: sample.get_field(field_name)
            for field_name in self.config.categorical_fields
        }
        target = sample.get_field(self.config.target_column) if self.config.target_column else None

        return {
            "image": image,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "target": target,
            "filepath": sample.filepath,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_bioprint_torch_dataset.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/bioprint_data/torch_dataset.py tests/test_bioprint_torch_dataset.py
git commit -m "feat: add PyTorch dataset skeleton"
```

---

### Task 6: Final Verification

**Files:**
- Verify all created files.

- [ ] **Step 1: Run all new tests**

Run:

```powershell
python -m pytest tests/test_bioprint_config.py tests/test_bioprint_fiftyone_io.py tests/test_bioprint_torch_dataset.py -v
```

Expected: all tests passed.

- [ ] **Step 2: Run syntax verification for scripts**

Run:

```powershell
python -m py_compile scripts/init_fiftyone_dataset.py scripts/import_to_fiftyone.py scripts/inspect_dataset.py
```

Expected: exit code 0.

- [ ] **Step 3: Run init script if FiftyOne is installed**

Run:

```powershell
python scripts/init_fiftyone_dataset.py --config configs/dataset.yaml
```

Expected when FiftyOne is installed:

```text
Dataset: bioprint_experiments
Samples: 0
Numeric fields: motion_speed, extrusion_speed, material_concentration, print_height, curvature_radius
Categorical fields: print_path
Target column: label
```

Expected when FiftyOne is not installed: Python reports `ModuleNotFoundError: No module named 'fiftyone'`. In that case, install FiftyOne later and keep unit tests passing now.

- [ ] **Step 4: Check unrelated user changes remain unstaged**

Run:

```powershell
git status --short
```

Expected: only files from this framework are staged or committed by this work; existing unrelated `控制代码/Motor.py` remains untouched.

- [ ] **Step 5: Commit final verification cleanup if needed**

```powershell
git add configs data scripts src/bioprint_data tests/test_bioprint_config.py tests/test_bioprint_fiftyone_io.py tests/test_bioprint_torch_dataset.py
git commit -m "chore: verify FiftyOne dataset framework"
```
