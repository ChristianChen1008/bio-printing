# Database Folder Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the FiftyOne dataset framework into one top-level `database/` folder while keeping scripts and tests runnable from the repository root.

**Architecture:** Preserve the Python package name `bioprint_data`, but relocate its source root to `database/src`. Relocate database configs, scripts, tests, and data directories together, then update path bootstrapping and pytest configuration to point at the new source root.

**Tech Stack:** Python, pytest, PowerShell, existing FiftyOne framework code

---

## File Structure

- Move: `configs/dataset.yaml` -> `database/configs/dataset.yaml`
- Move: `scripts/init_fiftyone_dataset.py` -> `database/scripts/init_fiftyone_dataset.py`
- Move: `scripts/import_to_fiftyone.py` -> `database/scripts/import_to_fiftyone.py`
- Move: `scripts/inspect_dataset.py` -> `database/scripts/inspect_dataset.py`
- Move: `src/bioprint_data/` -> `database/src/bioprint_data/`
- Move: `tests/test_bioprint_config.py` -> `database/tests/test_bioprint_config.py`
- Move: `tests/test_bioprint_fiftyone_io.py` -> `database/tests/test_bioprint_fiftyone_io.py`
- Move: `tests/test_bioprint_torch_dataset.py` -> `database/tests/test_bioprint_torch_dataset.py`
- Move: `data/raw/images/.gitkeep` -> `database/data/raw/images/.gitkeep`
- Move: `data/raw/metadata/.gitkeep` -> `database/data/raw/metadata/.gitkeep`
- Move: `data/processed/.gitkeep` -> `database/data/processed/.gitkeep`
- Modify: `pyproject.toml`

---

### Task 1: Move Database Files Under `database/`

**Files:**
- Move all files listed in File Structure.

- [ ] **Step 1: Move the files**

Run:

```powershell
New-Item -ItemType Directory -Force database/configs,database/scripts,database/src,database/tests,database/data/raw/images,database/data/raw/metadata,database/data/processed | Out-Null
git mv configs/dataset.yaml database/configs/dataset.yaml
git mv scripts/init_fiftyone_dataset.py database/scripts/init_fiftyone_dataset.py
git mv scripts/import_to_fiftyone.py database/scripts/import_to_fiftyone.py
git mv scripts/inspect_dataset.py database/scripts/inspect_dataset.py
git mv src/bioprint_data database/src/bioprint_data
git mv tests/test_bioprint_config.py database/tests/test_bioprint_config.py
git mv tests/test_bioprint_fiftyone_io.py database/tests/test_bioprint_fiftyone_io.py
git mv tests/test_bioprint_torch_dataset.py database/tests/test_bioprint_torch_dataset.py
git mv data/raw/images/.gitkeep database/data/raw/images/.gitkeep
git mv data/raw/metadata/.gitkeep database/data/raw/metadata/.gitkeep
git mv data/processed/.gitkeep database/data/processed/.gitkeep
```

- [ ] **Step 2: Verify expected old and new paths**

Run:

```powershell
Test-Path database/configs/dataset.yaml
Test-Path database/scripts/init_fiftyone_dataset.py
Test-Path database/src/bioprint_data/config.py
Test-Path database/tests/test_bioprint_config.py
Test-Path configs/dataset.yaml
Test-Path scripts/init_fiftyone_dataset.py
Test-Path src/bioprint_data/config.py
```

Expected:

```text
True
True
True
True
False
False
False
```

---

### Task 2: Update Paths and Test Configuration

**Files:**
- Modify: `database/scripts/init_fiftyone_dataset.py`
- Modify: `database/scripts/import_to_fiftyone.py`
- Modify: `database/scripts/inspect_dataset.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Update script bootstraps and default config paths**

In each database script, set:

```python
DATABASE_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = DATABASE_ROOT / "src"
```

and update the default config argument to:

```python
default="database/configs/dataset.yaml"
```

- [ ] **Step 2: Update pytest pythonpath**

Replace `pyproject.toml` contents with:

```toml
[tool.pytest.ini_options]
pythonpath = ["database/src"]
```

- [ ] **Step 3: Verify script help**

Run:

```powershell
python database/scripts/init_fiftyone_dataset.py --help
python database/scripts/import_to_fiftyone.py --help
python database/scripts/inspect_dataset.py --help
```

Expected: each command prints argparse help and exits successfully without requiring `fiftyone`.

---

### Task 3: Verify Moved Tests and Commit

**Files:**
- Verify: `database/tests/*.py`
- Verify: `tests/test_path_program.py`

- [ ] **Step 1: Run database and existing tests**

Run:

```powershell
python -m pytest database/tests tests/test_path_program.py -q
```

Expected:

```text
17 passed
```

- [ ] **Step 2: Run script syntax checks**

Run:

```powershell
python -m py_compile database/scripts/init_fiftyone_dataset.py database/scripts/import_to_fiftyone.py database/scripts/inspect_dataset.py
```

Expected: exit code 0.

- [ ] **Step 3: Inspect git status**

Run:

```powershell
git status --short
```

Expected: only database reorganization moves, `pyproject.toml`, and this plan file are changed.

- [ ] **Step 4: Commit**

Run:

```powershell
git add database pyproject.toml docs/superpowers/plans/2026-07-07-database-folder-reorganization.md
git commit -m "Move database framework into database folder"
```
