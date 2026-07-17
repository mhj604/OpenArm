from __future__ import annotations

import math

import numpy as np

VECTOR_SIZE = 16
ARM_INDICES = np.asarray([*range(7), *range(8, 15)], dtype=np.int64)


class LearnedControlError(ValueError):
    """Raised when learned-action stabilization inputs are invalid."""


def _finite_vector(value: np.ndarray, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (VECTOR_SIZE,) or not np.isfinite(vector).all():
        raise LearnedControlError(f"{label} must contain {VECTOR_SIZE} finite values")
    return vector


def slew_zero_target(
    previous_target: np.ndarray,
    zero_target: np.ndarray,
    *,
    maximum_arm_step_rad: float,
) -> np.ndarray:
    """Move an arm target toward zero without a one-frame position jump."""
    previous = _finite_vector(previous_target, label="previous zero-pose target")
    target = _finite_vector(zero_target, label="zero-pose target")
    if maximum_arm_step_rad <= 0 or not math.isfinite(maximum_arm_step_rad):
        raise LearnedControlError("zero-pose arm step must be finite and positive")
    result = target.copy()
    arm_delta = np.clip(
        target[ARM_INDICES] - previous[ARM_INDICES],
        -maximum_arm_step_rad,
        maximum_arm_step_rad,
    )
    result[ARM_INDICES] = previous[ARM_INDICES] + arm_delta
    return np.ascontiguousarray(result, dtype=np.float32)


def amplify_learned_chunk(
    raw_actions: np.ndarray,
    reference_target: np.ndarray,
    *,
    arm_gain: float,
) -> np.ndarray:
    """Amplify learned arm excursions around a fixed task reference.

    The DP3 trajectory remains the sole motion source. Gripper channels are
    deliberately unchanged because their position range is not symmetric
    around the arm zero pose.
    """
    actions = np.asarray(raw_actions, dtype=np.float32)
    reference = _finite_vector(reference_target, label="learned action reference")
    if actions.ndim != 2 or actions.shape[1] != VECTOR_SIZE or not np.isfinite(actions).all():
        raise LearnedControlError("raw learned actions must be finite and shaped (T, 16)")
    if len(actions) == 0:
        raise LearnedControlError("raw learned action chunk cannot be empty")
    if arm_gain <= 0 or not math.isfinite(arm_gain):
        raise LearnedControlError("learned arm gain must be finite and positive")

    amplified = actions.copy()
    amplified[:, ARM_INDICES] = reference[ARM_INDICES] + arm_gain * (
        actions[:, ARM_INDICES] - reference[ARM_INDICES]
    )
    return np.ascontiguousarray(amplified, dtype=np.float32)


def stabilize_learned_chunk(
    raw_actions: np.ndarray,
    previous_target: np.ndarray,
    *,
    elapsed_s: float,
    fps: float,
    initial_ramp_s: float,
    initial_alpha: float,
    normal_alpha: float,
    initial_max_arm_step_rad: float,
    normal_max_arm_step_rad: float,
) -> np.ndarray:
    """Continuously join a DP3 chunk to the last command while retaining its direction."""
    actions = np.asarray(raw_actions, dtype=np.float32)
    previous = _finite_vector(previous_target, label="previous learned target").copy()
    if actions.ndim != 2 or actions.shape[1] != VECTOR_SIZE or not np.isfinite(actions).all():
        raise LearnedControlError("raw learned actions must be finite and shaped (T, 16)")
    if len(actions) == 0:
        raise LearnedControlError("raw learned action chunk cannot be empty")
    positive = (fps, initial_max_arm_step_rad, normal_max_arm_step_rad)
    if any(value <= 0 or not math.isfinite(value) for value in positive):
        raise LearnedControlError("learned stabilization rates must be finite and positive")
    if elapsed_s < 0 or initial_ramp_s < 0:
        raise LearnedControlError("learned stabilization timing cannot be negative")
    if not 0 < initial_alpha <= 1 or not 0 < normal_alpha <= 1:
        raise LearnedControlError("learned stabilization alpha must be in (0, 1]")

    stabilized: list[np.ndarray] = []
    for index, raw in enumerate(actions):
        action_time = elapsed_s + index / fps
        in_initial_ramp = action_time < initial_ramp_s
        alpha = initial_alpha if in_initial_ramp else normal_alpha
        step_limit = (
            initial_max_arm_step_rad if in_initial_ramp else normal_max_arm_step_rad
        )
        result = raw.copy()
        filtered_delta = (raw[ARM_INDICES] - previous[ARM_INDICES]) * alpha
        filtered_delta = np.clip(filtered_delta, -step_limit, step_limit)
        result[ARM_INDICES] = previous[ARM_INDICES] + filtered_delta
        stabilized.append(result)
        previous = result
    return np.ascontiguousarray(np.stack(stabilized), dtype=np.float32)
