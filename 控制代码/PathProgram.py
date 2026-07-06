from __future__ import annotations

"""
复合路径程序层

本文件用于把多个单独轨迹和工艺动作组合成一个完整打印流程。
当前已支持四类步骤：
- travel: 空走到目标点
- print_trajectory: 执行单条轨迹并出料
- lift: 只抬高 Z
- retract: 单独回抽

当前已接入可视化窗口：
- 可在界面中切换“单条轨迹 / 复合程序”
- 可预览并执行预设复合程序
- 第一版预设示例为“矩形 + 圆形”
"""

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Iterable, Sequence

try:
    from .Trajectory import TrajectoryFactory
except ImportError:  # pragma: no cover - supports direct script-style imports
    from Trajectory import TrajectoryFactory


STEP_TYPE_TRAVEL = "travel"
STEP_TYPE_PRINT_TRAJECTORY = "print_trajectory"
STEP_TYPE_LIFT = "lift"
STEP_TYPE_RETRACT = "retract"

SUPPORTED_STEP_TYPES = {
    STEP_TYPE_TRAVEL,
    STEP_TYPE_PRINT_TRAJECTORY,
    STEP_TYPE_LIFT,
    STEP_TYPE_RETRACT,
}


@dataclass(slots=True)
class ProgramStep:
    step_type: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_type not in SUPPORTED_STEP_TYPES:
            raise ValueError(f"Unsupported step type: {self.step_type}")
        self.params = _validate_step_params(self.step_type, self.params)


class PathProgram:
    def __init__(self, name: str):
        self.name = name
        self.steps: list[ProgramStep] = []

    def add_step(self, step_type: str, name: str | None = None, **params: Any) -> "PathProgram":
        step_name = name or step_type
        self.steps.append(ProgramStep(step_type=step_type, name=step_name, params=params))
        return self

    def add_travel(self, point_xyz: Sequence[float], name: str | None = None) -> "PathProgram":
        return self.add_step(
            STEP_TYPE_TRAVEL,
            name,
            point_xyz=_as_point(point_xyz, expected_length=3, param_name="point_xyz"),
        )

    def add_print_trajectory(
        self,
        trajectory: Any,
        name: str | None = None,
        **params: Any,
    ) -> "PathProgram":
        return self.add_step(
            STEP_TYPE_PRINT_TRAJECTORY,
            name,
            trajectory=trajectory,
            **params,
        )

    def add_lift(self, delta_z: float, name: str | None = None) -> "PathProgram":
        return self.add_step(STEP_TYPE_LIFT, name, delta_z=delta_z)

    def add_retract(
        self,
        pulses: int,
        speed: int = 50,
        name: str | None = None,
    ) -> "PathProgram":
        return self.add_step(
            STEP_TYPE_RETRACT,
            name,
            pulses=pulses,
            speed=speed,
        )


def build_rectangle_circle_program(
    factory: Any = TrajectoryFactory,
    *,
    safe_z: float = 5.0,
    lift_z: float = 3.0,
) -> PathProgram:
    rectangle = factory.rectangle(
        [95, 95],
        12,
        8,
        line_width=1.0,
        name="rectangle_fill",
    )
    circle = factory.circle(
        [115, 105],
        5,
        name="circle",
    )

    rectangle_start = _trajectory_start(rectangle, fallback_xy=[89, 91])
    circle_start = _trajectory_start(circle, fallback_xy=[120, 105])

    return (
        PathProgram("rectangle_circle_program")
        .add_travel([rectangle_start[0], rectangle_start[1], safe_z], name="travel_to_rectangle")
        .add_print_trajectory(rectangle, name="print_rectangle")
        .add_retract(1200, name="retract_after_rectangle")
        .add_lift(lift_z, name="lift_clearance")
        .add_travel([circle_start[0], circle_start[1], safe_z], name="travel_to_circle")
        .add_print_trajectory(circle, name="print_circle")
    )


def _trajectory_start(trajectory: Any, fallback_xy: Iterable[float]) -> list[float]:
    points = getattr(trajectory, "generated_points", None)
    if points is not None:
        try:
            point_count = len(points)
        except TypeError:
            point_count = 0
        if point_count > 0:
            first_point = points[0]
            return _as_point(first_point, expected_length=2, param_name="trajectory start point")
    return _as_point(fallback_xy, expected_length=2, param_name="fallback_xy")


def _validate_step_params(step_type: str, params: dict[str, Any]) -> dict[str, Any]:
    validated = dict(params)

    if step_type == STEP_TYPE_TRAVEL:
        if "point_xyz" not in validated:
            raise ValueError("travel step requires point_xyz")
        validated["point_xyz"] = _as_point(validated["point_xyz"], expected_length=3, param_name="point_xyz")
        return validated

    if step_type == STEP_TYPE_PRINT_TRAJECTORY:
        if validated.get("trajectory") is None:
            raise ValueError("print_trajectory step requires trajectory")
        return validated

    if step_type == STEP_TYPE_LIFT:
        if "delta_z" not in validated:
            raise ValueError("lift step requires delta_z")
        validated["delta_z"] = _as_number(validated["delta_z"], param_name="delta_z")
        return validated

    if step_type == STEP_TYPE_RETRACT:
        if "pulses" not in validated:
            raise ValueError("retract step requires pulses")
        validated["pulses"] = int(_as_number(validated["pulses"], param_name="pulses"))
        if "speed" in validated:
            validated["speed"] = _as_number(validated["speed"], param_name="speed")
        return validated

    return validated


def _as_point(values: Iterable[float], *, expected_length: int, param_name: str) -> list[float]:
    if isinstance(values, (str, bytes, bytearray, memoryview)):
        raise ValueError(f"{param_name} must be an iterable of {expected_length} numeric values")
    try:
        point = [float(value) for value in values]
    except TypeError as exc:
        raise ValueError(f"{param_name} must be an iterable of {expected_length} numeric values") from exc

    if len(point) != expected_length:
        raise ValueError(f"{param_name} must contain exactly {expected_length} values")
    return point


def _as_number(value: Any, *, param_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{param_name} must be numeric")
    return float(value)


__all__ = [
    "ProgramStep",
    "PathProgram",
    "STEP_TYPE_TRAVEL",
    "STEP_TYPE_PRINT_TRAJECTORY",
    "STEP_TYPE_LIFT",
    "STEP_TYPE_RETRACT",
    "build_rectangle_circle_program",
]
