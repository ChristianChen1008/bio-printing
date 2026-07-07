# FiftyOne Dataset Framework Design

## Goal

Build a configuration-driven data management framework for a bioprinting multimodal machine learning dataset. The first version must work before real data exists: it should define the intended dataset schema, create an empty FiftyOne dataset, and provide clear extension points for importing future local experiment images and tabular experiment parameters.

The framework will support later PyTorch training by keeping image paths and experiment parameters in a consistent structure that can be converted into image tensors plus tabular features.

## Scope

This design covers:

- A stable project layout for dataset configuration, scripts, and reusable Python modules.
- A YAML configuration file that defines dataset name, data paths, image path column, parameter fields, and target fields.
- A script to initialize an empty FiftyOne dataset with the configured schema.
- A future import script interface for CSV or Excel metadata tables.
- A future PyTorch Dataset class that reads from FiftyOne and returns image data plus parameter features.

This design does not cover:

- Real data cleaning rules, because the actual metadata table format is not available yet.
- Model architecture or training loops beyond the data loading interface.
- Automated annotation or image feature extraction.

## Proposed Project Layout

```text
bioprint_project/
  configs/
    dataset.yaml
  data/
    raw/
      images/
      metadata/
    processed/
  scripts/
    init_fiftyone_dataset.py
    import_to_fiftyone.py
    inspect_dataset.py
  src/
    bioprint_data/
      __init__.py
      config.py
      fiftyone_io.py
      torch_dataset.py
```

## Configuration

The framework will use `configs/dataset.yaml` as the single source of truth for dataset setup.

The initial field design is:

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

`image_path` is reserved for the local experiment image path. `print_path` is an experiment parameter that represents the printing path or trajectory category.

## Components

### `src/bioprint_data/config.py`

Responsibilities:

- Load the YAML configuration.
- Resolve project-relative paths into absolute paths.
- Validate required sections such as `dataset`, `paths`, `columns`, and `fields`.

Consumers should use one function such as `load_dataset_config(config_path)` instead of reading YAML directly.

### `src/bioprint_data/fiftyone_io.py`

Responsibilities:

- Create or load the configured FiftyOne dataset.
- Add configured sample fields to the dataset schema.
- Provide reusable helpers for validating image paths and metadata rows.
- Later, convert rows from CSV or Excel into `fiftyone.Sample` objects.

The initial implementation should support creating an empty dataset with the schema only.

### `scripts/init_fiftyone_dataset.py`

Responsibilities:

- Read `configs/dataset.yaml`.
- Create or load the configured FiftyOne dataset.
- Apply the configured schema.
- Print a concise summary: dataset name, sample count, numeric fields, categorical fields.

Expected usage:

```powershell
python scripts/init_fiftyone_dataset.py --config configs/dataset.yaml
```

### `scripts/import_to_fiftyone.py`

Responsibilities:

- Read `configs/dataset.yaml`.
- Accept a metadata table path from the config or CLI.
- Support CSV first, with Excel support added once real files are available.
- Validate required columns before importing.
- Create one FiftyOne sample per metadata row.

The first implementation can print a clear message and exit successfully if no metadata file exists yet.

### `scripts/inspect_dataset.py`

Responsibilities:

- Load the configured FiftyOne dataset.
- Print dataset name, sample count, and schema.
- Optionally launch the FiftyOne App later.

### `src/bioprint_data/torch_dataset.py`

Responsibilities:

- Provide a PyTorch `Dataset` wrapper around a FiftyOne dataset.
- Return a dictionary shaped for multimodal training:

```python
{
    "image": image_tensor,
    "numeric_features": numeric_tensor,
    "categorical_features": categorical_values,
    "target": target_value,
    "filepath": filepath,
}
```

The first version may keep image transforms injectable and leave categorical encoding as a future step. This avoids locking the project into a specific model architecture too early.

## Data Flow

1. The user edits `configs/dataset.yaml` to define fields and paths.
2. `init_fiftyone_dataset.py` creates the empty FiftyOne dataset and schema.
3. Later, images are placed under `data/raw/images`.
4. Later, metadata tables are placed under `data/raw/metadata`.
5. `import_to_fiftyone.py` reads the table, validates image paths and parameter columns, and inserts samples.
6. `torch_dataset.py` loads the FiftyOne dataset and exposes samples to PyTorch.

## Error Handling

The scripts should fail with actionable messages when:

- The config file is missing or malformed.
- Required config sections are missing.
- A requested metadata table does not exist.
- A metadata table is missing required columns.
- A row points to an image file that does not exist.
- A configured dataset already exists and `overwrite` is false.

When no real data exists, initialization should still succeed and create an empty dataset.

## Testing

Initial tests should cover:

- Loading a valid YAML config.
- Rejecting a config with missing required sections.
- Creating or loading a FiftyOne dataset from config.
- Initializing schema fields without requiring real images.

Later tests should add temporary CSV metadata and tiny test images to verify import behavior.

## Open Decisions

These decisions are intentionally deferred until real data exists:

- Exact metadata table format: CSV, Excel, or both.
- Units for each numeric field.
- Whether `print_path` is a category, a file path to a path definition, or a structured trajectory representation.
- Target label type: classification label, regression value, or multiple targets.
- Categorical encoding strategy for PyTorch training.

## Acceptance Criteria

- The framework can be created without real experiment data.
- A user can run one command to initialize an empty FiftyOne dataset from YAML config.
- The six experiment parameter classes are represented explicitly:
  - motion speed
  - extrusion speed
  - material concentration
  - print height
  - curvature radius
  - print path
- Image file paths are separate from the print path experiment parameter.
- The structure leaves a clear path to future CSV/Excel import and PyTorch Dataset integration.
