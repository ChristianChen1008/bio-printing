from __future__ import annotations

from collections.abc import Sequence


DEFAULT_SAFE_POINT = [100.0, 100.0, 50.0]


def build_safe_lift_path(
    current_point: Sequence[float],
    safe_point: Sequence[float] = DEFAULT_SAFE_POINT,
) -> list[list[float]]:
    current = [float(value) for value in current_point[:3]]
    safe = [float(value) for value in safe_point[:3]]

    if current[2] < safe[2]:
        vertical_lift = [current[0], current[1], safe[2]]
        return [vertical_lift, safe]

    return [safe]
