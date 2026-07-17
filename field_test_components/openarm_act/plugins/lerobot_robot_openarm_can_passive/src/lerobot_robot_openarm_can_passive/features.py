from __future__ import annotations

MOTOR_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper")


def motor_features() -> dict[str, type]:
    return {f"{side}_{motor}.pos": float for side in ("left", "right") for motor in MOTOR_NAMES}


def motor_values(left: list[float], right: list[float]) -> dict[str, float]:
    if len(left) != 8 or len(right) != 8:
        raise ValueError("left and right motor vectors must each contain eight values")
    return {
        f"{side}_{motor}.pos": float(value)
        for side, values in (("left", left), ("right", right))
        for motor, value in zip(MOTOR_NAMES, values, strict=True)
    }
