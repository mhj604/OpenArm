from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rollout_profile import VECTOR_SIZE, RolloutProfile

STATE_ORDER = (
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_joint_6.pos",
    "left_joint_7.pos",
    "left_gripper.pos",
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_joint_6.pos",
    "right_joint_7.pos",
    "right_gripper.pos",
)


class ZeroPoseError(ValueError):
    """Raised when the configured OpenArm zero pose is invalid."""


@dataclass(frozen=True)
class ZeroPose:
    target: np.ndarray
    source: str

    @classmethod
    def load(cls, path: Path) -> ZeroPose:
        if not path.is_file():
            raise ZeroPoseError(f"missing zero-pose config: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ZeroPoseError("unsupported zero-pose config")
        if raw.get("unit") != "radian" or tuple(raw.get("state_order", ())) != STATE_ORDER:
            raise ZeroPoseError("zero-pose unit or state order does not match OpenArm")
        target = np.asarray(raw.get("target"), dtype=np.float64)
        if target.shape != (VECTOR_SIZE,) or not np.isfinite(target).all():
            raise ZeroPoseError(f"zero-pose target must contain {VECTOR_SIZE} finite values")
        source = raw.get("source")
        if not isinstance(source, str) or not source:
            raise ZeroPoseError("zero-pose source must be documented")
        return cls(target=target, source=source)

    def action_target(self, profile: RolloutProfile) -> np.ndarray:
        """Return the zero pose clipped only to demonstrated motor command bounds."""
        return np.ascontiguousarray(
            np.clip(self.target, profile.action_min, profile.action_max), dtype=np.float32
        )
