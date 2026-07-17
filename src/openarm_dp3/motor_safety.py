from __future__ import annotations

import math
from collections.abc import Sequence

VECTOR_SIZE = 16
GRIPPER_INDICES = (7, 15)


def gripper_feedback_violations(
    state: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    margin: float,
) -> list[str]:
    """Return gripper readings that cannot represent this dataset's operating range."""
    if len(state) != VECTOR_SIZE or len(lower) != VECTOR_SIZE or len(upper) != VECTOR_SIZE:
        raise ValueError("state and action bounds must each contain 16 values")
    if margin < 0 or not math.isfinite(margin):
        raise ValueError("feedback margin must be finite and non-negative")
    violations: list[str] = []
    for index in GRIPPER_INDICES:
        value = float(state[index])
        minimum = float(lower[index]) - margin
        maximum = float(upper[index]) + margin
        if not math.isfinite(value) or value < minimum or value > maximum:
            violations.append(
                f"index {index}: feedback={value:.4f}, expected=[{minimum:.4f}, {maximum:.4f}]"
            )
    return violations
