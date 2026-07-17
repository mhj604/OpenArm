from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

VECTOR_SIZE = 16
GRIPPER_INDICES = frozenset({7, 15})


class RolloutProfileError(ValueError):
    """Raised when rollout limits cannot be derived or validated safely."""


def _finite_vector(value: Any, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (VECTOR_SIZE,) or not np.isfinite(vector).all():
        raise RolloutProfileError(f"{label} must contain {VECTOR_SIZE} finite values")
    return vector


@dataclass(frozen=True)
class RolloutProfile:
    action_min: np.ndarray
    action_max: np.ndarray
    start_min: np.ndarray
    start_max: np.ndarray
    start_mean: np.ndarray

    @classmethod
    def load(cls, path: Path) -> RolloutProfile:
        if not path.is_file():
            raise RolloutProfileError(f"missing rollout profile: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise RolloutProfileError(f"unsupported rollout profile: {path}")
        action = raw.get("action")
        start = raw.get("start_state")
        if not isinstance(action, dict) or not isinstance(start, dict):
            raise RolloutProfileError("rollout profile is missing action or start_state")
        profile = cls(
            action_min=_finite_vector(action.get("min"), label="action.min"),
            action_max=_finite_vector(action.get("max"), label="action.max"),
            start_min=_finite_vector(start.get("min"), label="start_state.min"),
            start_max=_finite_vector(start.get("max"), label="start_state.max"),
            start_mean=_finite_vector(start.get("mean"), label="start_state.mean"),
        )
        if np.any(profile.action_min > profile.action_max):
            raise RolloutProfileError("action bounds are reversed")
        if np.any(profile.start_min > profile.start_max):
            raise RolloutProfileError("start-state bounds are reversed")
        return profile

    def start_violations(
        self,
        state: Sequence[float],
        *,
        arm_margin: float,
        gripper_margin: float,
    ) -> list[str]:
        current = _finite_vector(state, label="live state")
        if arm_margin < 0 or gripper_margin < 0:
            raise RolloutProfileError("start-state margins cannot be negative")
        violations: list[str] = []
        for index, value in enumerate(current):
            margin = gripper_margin if index in GRIPPER_INDICES else arm_margin
            lower = self.start_min[index] - margin
            upper = self.start_max[index] + margin
            if value < lower or value > upper:
                violations.append(
                    f"index {index}: live={value:.4f}, allowed=[{lower:.4f}, {upper:.4f}]"
                )
        return violations


def create_profile(dataset_path: Path) -> dict[str, Any]:
    if not dataset_path.is_dir():
        raise RolloutProfileError(f"missing processed dataset: {dataset_path}")
    root = zarr.open_group(str(dataset_path), mode="r")
    state = np.asarray(root["data/state"][:], dtype=np.float64)
    action = np.asarray(root["data/action"][:], dtype=np.float64)
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if state.ndim != 2 or state.shape[1] != VECTOR_SIZE or state.shape != action.shape:
        raise RolloutProfileError(
            f"state/action must share shape (T, {VECTOR_SIZE}), "
            f"got {state.shape} and {action.shape}"
        )
    if len(state) == 0 or not np.isfinite(state).all() or not np.isfinite(action).all():
        raise RolloutProfileError("state/action data must be non-empty and finite")
    if (
        episode_ends.ndim != 1
        or len(episode_ends) == 0
        or episode_ends[-1] != len(state)
        or np.any(np.diff(episode_ends) <= 0)
    ):
        raise RolloutProfileError("episode_ends is malformed")
    start_indices = np.concatenate((np.asarray([0], dtype=np.int64), episode_ends[:-1]))
    start_state = state[start_indices]
    unit = str(root.attrs.get("state_action_unit", ""))
    if unit != "radian":
        raise RolloutProfileError(f"expected radian state/action data, got {unit!r}")
    return {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "task_id": str(root.attrs.get("task_id", "")),
        "episode_count": int(len(episode_ends)),
        "frame_count": int(len(state)),
        "sample_fps": int(root.attrs.get("sample_fps", 0)),
        "state_action_unit": unit,
        "action": {
            "min": action.min(axis=0).tolist(),
            "max": action.max(axis=0).tolist(),
        },
        "start_state": {
            "min": start_state.min(axis=0).tolist(),
            "max": start_state.max(axis=0).tolist(),
            "mean": start_state.mean(axis=0).tolist(),
        },
    }


def write_profile(dataset_path: Path, output_path: Path) -> dict[str, Any]:
    profile = create_profile(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp_{os.getpid()}")
    temporary.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    RolloutProfile.load(output_path)
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = write_profile(args.dataset, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episode_count": profile["episode_count"],
                "frame_count": profile["frame_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
