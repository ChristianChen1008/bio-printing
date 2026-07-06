# Composite Path Program Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-version support for complex composite print programs that sequence multiple trajectories and non-printing motions, while preserving the existing single-trajectory workflow.

**Architecture:** Keep `Trajectory` focused on geometry generation and add a new `PathProgram` layer for process sequencing. Extend the existing print worker and preview pipeline to accept either a single trajectory or a composite program, with UI limited to preset program selection in v1.

**Tech Stack:** Python, PyQt5, NumPy, existing `TrajectoryFactory`, existing UR3 and motor controllers

---

## File Structure

- Create: `控制代码/PathProgram.py`
  - Define `ProgramStep`, `PathProgram`, and first-version preset builders.
- Modify: `可视化(1)/MINE_window.py`
  - Add composite program preview and execution support.
  - Preserve existing single-trajectory flow.
- Modify: `可视化(1)/MINE_ui.py`
  - Add minimal UI controls for selecting preset complex programs.
- Optional Modify: `控制代码/Trajectory.py`
  - Only if a tiny helper is needed for path flattening; otherwise leave untouched.

---

### Task 1: Add Composite Program Domain Layer

**Files:**
- Create: `控制代码/PathProgram.py`
- Test: one-off import and object-construction verification via inline Python

- [ ] **Step 1: Write the failing verification script**

```python
from 控制代码.PathProgram import PathProgram
from 控制代码.Trajectory import TrajectoryFactory

traj = TrajectoryFactory.line([0, 0], [10, 0], step_mm=1.0)
program = PathProgram("test")
program.add_travel([0, 0, 5])
program.add_print_trajectory(traj, name="line")
program.add_lift(3)
program.add_retract(200)

assert len(program.steps) == 4
assert program.steps[0].step_type == "travel"
assert program.steps[1].step_type == "print_trajectory"
assert program.steps[2].step_type == "lift"
assert program.steps[3].step_type == "retract"
```

- [ ] **Step 2: Run verification to confirm it fails before implementation**

Run:

```bash
@'
from 控制代码.PathProgram import PathProgram
'@ | python -
```

Expected: import failure because `PathProgram.py` does not exist yet.

- [ ] **Step 3: Implement the minimal domain layer**

Implement `控制代码/PathProgram.py` with:

```python
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ProgramStep:
    step_type: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)


class PathProgram:
    def __init__(self, name: str):
        self.name = name
        self.steps: list[ProgramStep] = []

    def add_step(self, step_type: str, name: str | None = None, **params):
        self.steps.append(ProgramStep(step_type, name or step_type, params))
        return self

    def add_travel(self, point_xyz, name: str | None = None):
        return self.add_step("travel", name, point_xyz=list(point_xyz))

    def add_print_trajectory(self, trajectory, name: str | None = None, **params):
        return self.add_step("print_trajectory", name, trajectory=trajectory, **params)

    def add_lift(self, delta_z, name: str | None = None):
        return self.add_step("lift", name, delta_z=float(delta_z))

    def add_retract(self, pulses, speed=50, name: str | None = None):
        return self.add_step("retract", name, pulses=int(pulses), speed=int(speed))


def build_rect_circle_program(factory):
    rect = factory.rectangle([95, 95], 12, 8, line_width=1.0, name="矩形填充")
    circle = factory.circle([115, 105], 5, name="圆形")
    return (
        PathProgram("矩形+圆形组合")
        .add_travel([95, 91, 5], name="移动到矩形区域")
        .add_print_trajectory(rect, name="打印矩形")
        .add_retract(1200, name="矩形后回抽")
        .add_lift(3, name="抬高避让")
        .add_travel([120, 105, 5], name="移动到圆形区域")
        .add_print_trajectory(circle, name="打印圆形")
    )
```

- [ ] **Step 4: Run the verification script to confirm it passes**

Run:

```bash
@'
from 控制代码.PathProgram import PathProgram
from 控制代码.Trajectory import TrajectoryFactory

traj = TrajectoryFactory.line([0, 0], [10, 0], step_mm=1.0)
program = PathProgram("test")
program.add_travel([0, 0, 5])
program.add_print_trajectory(traj, name="line")
program.add_lift(3)
program.add_retract(200)

assert len(program.steps) == 4
assert program.steps[0].step_type == "travel"
assert program.steps[1].step_type == "print_trajectory"
assert program.steps[2].step_type == "lift"
assert program.steps[3].step_type == "retract"
print("ok")
'@ | python -
```

Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add 控制代码/PathProgram.py
git commit -m "feat: add composite path program domain model"
```

---

### Task 2: Add Preview Support for Composite Programs

**Files:**
- Modify: `可视化(1)/MINE_window.py`
- Test: `python -m py_compile 可视化(1)/MINE_window.py`

- [ ] **Step 1: Write the failing verification script**

Add a temporary inline expectation that a composite program can be flattened into previewable points through a new helper such as `_build_preview_points_from_program(program)`.

```python
from 控制代码.PathProgram import build_rect_circle_program
from 控制代码.Trajectory import TrajectoryFactory

program = build_rect_circle_program(TrajectoryFactory)
assert len(program.steps) > 1
```

- [ ] **Step 2: Run a syntax/import verification before editing**

Run:

```bash
python -m py_compile "可视化(1)\MINE_window.py"
```

Expected: passes before change, confirming a stable baseline.

- [ ] **Step 3: Implement minimal preview support**

Add to `可视化(1)/MINE_window.py`:

- a helper that converts one `ProgramStep` into point sequences
- a helper that flattens a `PathProgram` into a total preview path
- branch logic in preview flow:
  - single trajectory -> existing behavior
  - preset composite program -> new flattening logic

Use straightforward rules:

```python
if step.step_type == "travel":
    pts.append(np.array(step.params["point_xyz"], dtype=float))
elif step.step_type == "lift":
    # derive current point and adjust z only
elif step.step_type == "print_trajectory":
    # reuse trajectory.generated_points or trajectory.generate_path_points(...)
```

- [ ] **Step 4: Verify syntax after implementation**

Run:

```bash
python -m py_compile "可视化(1)\MINE_window.py"
```

Expected: exit code 0.

- [ ] **Step 5: Commit**

```bash
git add "可视化(1)/MINE_window.py"
git commit -m "feat: add composite path preview support"
```

---

### Task 3: Add Composite Program Execution to PrintWorker

**Files:**
- Modify: `可视化(1)/MINE_window.py`
- Test: `python -m py_compile 可视化(1)/MINE_window.py`

- [ ] **Step 1: Write the failing verification target**

Document the expected new execution entry points inside `PrintWorker`:

- `_execute_single_trajectory(...)`
- `_execute_path_program(...)`
- `_execute_program_step(...)`

The failure condition before implementation is that only single-trajectory inline logic exists.

- [ ] **Step 2: Run baseline verification**

Run:

```bash
python -m py_compile "可视化(1)\MINE_window.py"
```

Expected: passes before refactor.

- [ ] **Step 3: Refactor print execution with minimal feature expansion**

In `可视化(1)/MINE_window.py`:

- extract current single-trajectory print sequence into a helper
- add a second helper that loops through `PathProgram.steps`
- per step:
  - `travel`: stop motor, move arm to point
  - `lift`: read current TCP and move z upward
  - `retract`: run `motor.backward(...)`
  - `print_trajectory`: reuse the extracted single-trajectory helper

Keep:

- `_abort` checks
- current emergency stop behavior
- current shared-device behavior
- current “skip motor polling while printing” protection

- [ ] **Step 4: Verify syntax after refactor**

Run:

```bash
python -m py_compile "可视化(1)\MINE_window.py"
```

Expected: exit code 0.

- [ ] **Step 5: Commit**

```bash
git add "可视化(1)/MINE_window.py"
git commit -m "feat: execute composite path programs in print worker"
```

---

### Task 4: Add Minimal UI for Preset Complex Programs

**Files:**
- Modify: `可视化(1)/MINE_ui.py`
- Modify: `可视化(1)/MINE_window.py`
- Test: `python -m py_compile 可视化(1)/MINE_ui.py 可视化(1)/MINE_window.py`

- [ ] **Step 1: Write the failing verification target**

Expected new UI surface:

- one mode selector or checkbox for `单一路径 / 复杂流程`
- one preset selector for complex programs

Before implementation, these controls do not exist.

- [ ] **Step 2: Run baseline syntax verification**

Run:

```bash
python -m py_compile "可视化(1)\MINE_ui.py" "可视化(1)\MINE_window.py"
```

Expected: exit code 0.

- [ ] **Step 3: Implement minimal preset selection UI**

In `可视化(1)/MINE_ui.py`:

- add a compact selector for print mode
- add a combo box for composite presets
- keep current layout conservative; do not redesign the whole window

In `可视化(1)/MINE_window.py`:

- populate preset list
- toggle relevant controls by mode
- route preview/start-print to:
  - existing single-trajectory behavior
  - composite program builder and executor

- [ ] **Step 4: Verify syntax after UI changes**

Run:

```bash
python -m py_compile "可视化(1)\MINE_ui.py" "可视化(1)\MINE_window.py"
```

Expected: exit code 0.

- [ ] **Step 5: Commit**

```bash
git add "可视化(1)/MINE_ui.py" "可视化(1)/MINE_window.py"
git commit -m "feat: add preset composite path program controls"
```

---

### Task 5: End-to-End Verification and Final Cleanup

**Files:**
- Verify: `控制代码/PathProgram.py`
- Verify: `可视化(1)/MINE_ui.py`
- Verify: `可视化(1)/MINE_window.py`

- [ ] **Step 1: Run full syntax verification**

Run:

```bash
python -m py_compile "控制代码\PathProgram.py" "可视化(1)\MINE_ui.py" "可视化(1)\MINE_window.py"
```

Expected: exit code 0.

- [ ] **Step 2: Run import verification for the new domain layer**

Run:

```bash
@'
from 控制代码.PathProgram import PathProgram, build_rect_circle_program
from 控制代码.Trajectory import TrajectoryFactory

program = build_rect_circle_program(TrajectoryFactory)
assert isinstance(program, PathProgram)
assert len(program.steps) >= 2
print(program.name, len(program.steps))
'@ | python -
```

Expected: prints program name and step count.

- [ ] **Step 3: Re-read spec checklist and verify feature coverage**

Checklist:

- complex program domain added
- at least one preset complex program added
- preview supports complex programs
- print worker supports complex programs
- single-trajectory mode still present
- no regressions to shared device handling and emergency stop logic

- [ ] **Step 4: Commit final cleanup if needed**

```bash
git add 控制代码/PathProgram.py "可视化(1)/MINE_ui.py" "可视化(1)/MINE_window.py"
git commit -m "chore: finalize composite path program support"
```
