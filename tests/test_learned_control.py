import numpy as np

from openarm_dp3.learned_control import (
    amplify_learned_chunk,
    slew_zero_target,
    stabilize_learned_chunk,
)


def test_zero_pose_slew_limits_only_arm_channels() -> None:
    previous = np.ones(16, dtype=np.float32)
    target = np.zeros(16, dtype=np.float32)
    result = slew_zero_target(previous, target, maximum_arm_step_rad=0.03)
    arm_indices = [*range(7), *range(8, 15)]
    np.testing.assert_allclose(result[arm_indices], 0.97)
    np.testing.assert_allclose(result[[7, 15]], 0.0)


def test_learned_gain_amplifies_arms_around_reference_without_changing_grippers() -> None:
    reference = np.zeros(16, dtype=np.float32)
    reference[[0, 8]] = [0.2, -0.1]
    raw = np.tile(reference, (2, 1))
    raw[:, 0] += [0.1, -0.2]
    raw[:, 8] += [0.3, -0.4]
    raw[:, [7, 15]] = [[-0.2, -0.1], [-0.4, -0.3]]
    result = amplify_learned_chunk(raw, reference, arm_gain=1.8)
    np.testing.assert_allclose(result[:, 0], [0.38, -0.16], atol=1e-6)
    np.testing.assert_allclose(result[:, 8], [0.44, -0.82], atol=1e-6)
    np.testing.assert_allclose(result[:, [7, 15]], raw[:, [7, 15]])


def test_learned_chunk_is_softly_joined_then_uses_normal_step_limit() -> None:
    raw = np.ones((8, 16), dtype=np.float32)
    previous = np.zeros(16, dtype=np.float32)
    result = stabilize_learned_chunk(
        raw,
        previous,
        elapsed_s=0.0,
        fps=4.0,
        initial_ramp_s=1.0,
        initial_alpha=0.25,
        normal_alpha=0.70,
        initial_max_arm_step_rad=0.025,
        normal_max_arm_step_rad=0.08,
    )
    transitions = np.diff(np.vstack((previous, result)), axis=0)
    assert np.max(np.abs(transitions[:4, :7])) <= 0.025 + 1e-7
    assert np.max(np.abs(transitions[4:, :7])) <= 0.08 + 1e-7
    assert np.all(np.diff(result[:, 0]) >= 0)
    # Grippers remain learned values; the separate initial-open guard owns them.
    np.testing.assert_allclose(result[:, [7, 15]], 1.0)


def test_new_chunk_cannot_jump_away_from_previous_learned_target() -> None:
    previous = np.full(16, 0.5, dtype=np.float32)
    raw = np.full((3, 16), -0.5, dtype=np.float32)
    result = stabilize_learned_chunk(
        raw,
        previous,
        elapsed_s=2.0,
        fps=15.0,
        initial_ramp_s=1.0,
        initial_alpha=0.25,
        normal_alpha=0.70,
        initial_max_arm_step_rad=0.025,
        normal_max_arm_step_rad=0.08,
    )
    first_delta = result[0] - previous
    assert np.max(np.abs(first_delta[:7])) <= 0.08 + 1e-7
    assert np.max(np.abs(first_delta[8:15])) <= 0.08 + 1e-7
