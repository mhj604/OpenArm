import pytest

from openarm_dp3.motor_safety import gripper_feedback_violations


def test_enable_ack_sentinel_is_rejected_for_both_grippers() -> None:
    state = [0.0] * 16
    state[7] = -12.46757457846952
    state[15] = -12.46757457846952
    lower = [-1.0] * 16
    upper = [1.0] * 16
    violations = gripper_feedback_violations(state, lower, upper, margin=0.2)
    assert len(violations) == 2
    assert violations[0].startswith("index 7:")
    assert violations[1].startswith("index 15:")


def test_demonstrated_gripper_feedback_is_accepted() -> None:
    state = [0.0] * 16
    state[7] = -0.0238
    state[15] = -0.0162
    lower = [-1.0] * 16
    upper = [1.0] * 16
    assert not gripper_feedback_violations(state, lower, upper, margin=0.2)


def test_invalid_feedback_margin_is_rejected() -> None:
    with pytest.raises(ValueError, match="margin"):
        gripper_feedback_violations([0.0] * 16, [-1.0] * 16, [1.0] * 16, margin=-0.1)
