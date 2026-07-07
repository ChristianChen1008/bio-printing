# Database Folder Reorganization Design

## Goal

Move all FiftyOne dataset framework files into one top-level `database/` folder so the data-management code, configuration, scripts, tests, and data directories are easy to find together.

## Scope

Move these database-related areas:

- `configs/dataset.yaml`
- `scripts/init_fiftyone_dataset.py`
- `scripts/import_to_fiftyone.py`
- `scripts/inspect_dataset.py`
- `src/bioprint_data/`
- `tests/test_bioprint_config.py`
- `tests/test_bioprint_fiftyone_io.py`
- `tests/test_bioprint_torch_dataset.py`
- `data/raw/images/`
- `data/raw/metadata/`
- `data/processed/`

Do not move:

- `控制代码/`
- `可视化(1)/`
- `tests/test_path_program.py`
- Existing docs except this design and the implementation plan.

## Target Layout

```text
database/
  configs/
    dataset.yaml
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
  tests/
    test_bioprint_config.py
    test_bioprint_fiftyone_io.py
    test_bioprint_torch_dataset.py
  data/
    raw/
      images/
      metadata/
    processed/
```

## Required Code Updates

- Update script path bootstrapping so scripts under `database/scripts/` import from `database/src/`.
- Update default config paths in the three database scripts to `database/configs/dataset.yaml` when run from the repository root.
- Update tests so `pytest database/tests` can import `database/src/bioprint_data`.
- Keep the package import name as `bioprint_data`.
- Update config path resolution so `database/configs/dataset.yaml` treats `database/` as the database project root. This keeps configured data paths under `database/data/...`.

## Verification

Run:

```powershell
python -m pytest database/tests tests/test_path_program.py -q
python -m py_compile database/scripts/init_fiftyone_dataset.py database/scripts/import_to_fiftyone.py database/scripts/inspect_dataset.py
python database/scripts/init_fiftyone_dataset.py --help
python database/scripts/import_to_fiftyone.py --help
python database/scripts/inspect_dataset.py --help
```

The real initialization command may still require installing `fiftyone`.

## Acceptance Criteria

- Database framework files are grouped under `database/`.
- Non-database control and visualization files remain in place.
- Existing tests still pass.
- Script help commands still work from the repository root.
- Unsupported import and inspect behavior remain non-destructive.
