from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


CONTROLLER_MATH_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "openarm_testusd_task"
    / "openarm_testusd_task"
    / "tasks"
    / "testusd_pick_place"
    / "controller_math.py"
)
SPEC = importlib.util.spec_from_file_location("openarm_controller_math", CONTROLLER_MATH_PATH)
assert SPEC is not None and SPEC.loader is not None
controller_math = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller_math)
jerk_limited_cartesian_velocity = controller_math.jerk_limited_cartesian_velocity
active_translation_progress_ratio = controller_math.active_translation_progress_ratio
apply_stopping_speed_envelope = controller_math.apply_stopping_speed_envelope
build_prioritized_dls_solver = controller_math.build_prioritized_dls_solver
constrain_target_lead = controller_math.constrain_target_lead
preserve_safe_goal_directed_step = controller_math.preserve_safe_goal_directed_step
project_goal_directed_step = controller_math.project_goal_directed_step
quintic_smoothstep = controller_math.quintic_smoothstep
realized_translation_progress = controller_math.realized_translation_progress
solve_prioritized_dls_delta = controller_math.solve_prioritized_dls_delta
target_lead_constraints_valid = controller_math.target_lead_constraints_valid
target_lead_constraint_intersection = (
    controller_math.target_lead_constraint_intersection
)
translation_progress_ratios = controller_math.translation_progress_ratios


def test_quintic_blend_has_exact_endpoints_and_is_monotonic():
    progress = torch.linspace(0.0, 1.0, 101)
    blend = quintic_smoothstep(progress)

    assert blend[0].item() == 0.0
    assert blend[-1].item() == 1.0
    assert torch.all(blend[1:] >= blend[:-1])


def test_xy_motion_is_not_throttled_by_inactive_z():
    candidate = torch.tensor([[0.0030, 0.0010, 0.0040]])
    error = torch.tensor([[0.10, 0.00, 0.0005]])

    projected = project_goal_directed_step(
        candidate,
        error,
        maximum_xy=0.0025,
        maximum_up=0.0025,
        maximum_down=0.0030,
        horizontal_deadband=0.005,
        vertical_deadband=0.002,
    )

    assert torch.allclose(projected[:, :2], torch.tensor([[0.0025, 0.0]]))
    assert projected[0, 2].item() == 0.0


def test_z_motion_is_not_throttled_by_inactive_xy():
    candidate = torch.tensor([[0.0020, 0.0010, -0.0040]])
    error = torch.tensor([[0.0005, 0.0005, -0.05]])

    projected = project_goal_directed_step(
        candidate,
        error,
        maximum_xy=0.0025,
        maximum_up=0.0025,
        maximum_down=0.0030,
        horizontal_deadband=0.005,
        vertical_deadband=0.002,
    )

    assert torch.allclose(projected[:, :2], torch.zeros((1, 2)))
    assert torch.allclose(projected[:, 2], torch.tensor([-0.0030]))


def test_away_motion_is_removed_per_axis_without_stopping_safe_axis():
    candidate = torch.tensor([[-0.0020, 0.0, -0.0020]])
    error = torch.tensor([[0.10, 0.0, -0.05]])

    projected = project_goal_directed_step(
        candidate,
        error,
        maximum_xy=0.0025,
        maximum_up=0.0025,
        maximum_down=0.0030,
        horizontal_deadband=0.005,
        vertical_deadband=0.002,
    )

    assert torch.allclose(projected[:, :2], torch.zeros((1, 2)))
    assert torch.allclose(projected[:, 2], torch.tensor([-0.0020]))


def test_projection_stops_at_deadband_instead_of_crossing_it():
    candidate = torch.tensor([[0.0020, 0.0, -0.0020]])
    error = torch.tensor([[0.0055, 0.0, -0.0025]])

    projected = project_goal_directed_step(
        candidate,
        error,
        maximum_xy=0.0025,
        maximum_up=0.0025,
        maximum_down=0.0030,
        horizontal_deadband=0.005,
        vertical_deadband=0.002,
    )

    assert torch.allclose(projected, torch.tensor([[0.0005, 0.0, -0.0005]]))


def test_safe_tangential_entry_motion_is_preserved():
    candidate = torch.tensor([[0.0020, 0.0010, 0.0]])
    error = torch.tensor([[0.10, 0.02, 0.0]])

    preserved = preserve_safe_goal_directed_step(
        candidate,
        error,
        maximum_xy=0.0025,
        maximum_up=0.0025,
        maximum_down=0.0030,
        horizontal_deadband=0.005,
        vertical_deadband=0.002,
    )

    assert torch.allclose(preserved, candidate)


def test_stopping_envelope_starts_before_the_deadband():
    velocity = torch.tensor([[0.075, 0.0, -0.090]])
    error = torch.tensor([[0.006, 0.0, -0.006]])

    bounded = apply_stopping_speed_envelope(
        velocity,
        error,
        horizontal_deadband=0.005,
        vertical_deadband=0.002,
        maximum_horizontal_acceleration=0.60,
        maximum_vertical_acceleration=0.45,
        maximum_horizontal_jerk=6.0,
        maximum_vertical_jerk=4.5,
    )

    assert torch.norm(bounded[:, :2], dim=1).item() < 0.075
    assert torch.abs(bounded[:, 2]).item() < 0.090


def test_target_lead_applies_previous_target_continuity_last():
    generator = torch.Generator().manual_seed(33)
    batch_size = 4096
    lower = -0.8 + 0.2 * torch.rand((batch_size, 7), generator=generator)
    upper = 0.6 + 0.2 * torch.rand((batch_size, 7), generator=generator)
    limits = torch.stack((lower, upper), dim=2)
    current = lower + (upper - lower) * torch.rand(
        (batch_size, 7), generator=generator
    )
    previous_offset = 0.24 * torch.rand(
        (batch_size, 7), generator=generator
    ) - 0.12
    previous = torch.clamp(current + previous_offset, min=lower, max=upper)
    raw_lead = 0.6 * torch.randn((batch_size, 7), generator=generator)

    lead = constrain_target_lead(
        raw_lead,
        current,
        previous,
        limits,
        maximum_lead=0.12,
        maximum_target_change=0.03,
    )
    target = current + lead

    assert torch.amax(torch.abs(target - previous)).item() <= 0.030001
    assert torch.all(
        target_lead_constraints_valid(
            lead,
            current,
            previous,
            limits,
            maximum_lead=0.12,
            maximum_target_change=0.03,
        )
    )


def test_target_lead_regression_near_joint_limit_does_not_reopen_continuity():
    current = torch.tensor(
        [[-0.8416008, 0.8293270, -0.4304750, 0.9996859, -0.6435938, -0.6340439, 0.9083976]]
    )
    previous = torch.tensor(
        [[-0.7855523, 0.7859160, -0.3974628, 0.9885018, -0.6198741, -0.6535927, 0.9588839]]
    )
    raw_lead = torch.tensor(
        [[0.0106348, 0.0079568, 0.0071751, 0.0114982, 0.0138148, 0.0022692, 0.0018343]]
    )
    limits = torch.tensor([[[-1.0, 1.0]] * 7])

    lead = constrain_target_lead(
        raw_lead,
        current,
        previous,
        limits,
        maximum_lead=0.12,
        maximum_target_change=0.03,
    )
    target = current + lead

    assert torch.amax(torch.abs(target - previous)).item() <= 0.030001
    assert target_lead_constraints_valid(
        lead,
        current,
        previous,
        limits,
        maximum_lead=0.12,
        maximum_target_change=0.03,
    ).item()


def test_target_lead_projects_into_the_full_feasible_intersection():
    current = torch.tensor([[0.0]])
    previous = torch.tensor([[0.145]])
    limits = torch.tensor([[[-1.0, 1.0]]])

    lower, upper, feasible = target_lead_constraint_intersection(
        current,
        previous,
        limits,
        maximum_lead=0.12,
        maximum_target_change=0.03,
    )
    lead = constrain_target_lead(
        torch.tensor([[0.20]]),
        current,
        previous,
        limits,
        maximum_lead=0.12,
        maximum_target_change=0.03,
    )

    assert feasible.item()
    assert torch.allclose(lower, torch.tensor([[0.115]]))
    assert torch.allclose(upper, torch.tensor([[0.120]]))
    assert torch.allclose(lead, torch.tensor([[0.120]]))
    assert target_lead_constraints_valid(
        lead, current, previous, limits, 0.12, 0.03
    ).item()


def test_infeasible_target_lead_rebases_then_recovers_on_the_next_control():
    current = torch.tensor([[0.0]])
    previous = torch.tensor([[0.20]])
    limits = torch.tensor([[[-1.0, 1.0]]])

    lower, upper, feasible = target_lead_constraint_intersection(
        current, previous, limits, 0.12, 0.03
    )
    lead = constrain_target_lead(
        torch.tensor([[0.03]]), current, previous, limits, 0.12, 0.03
    )

    assert not feasible.item()
    assert lower.item() > upper.item()
    assert torch.equal(lead, torch.zeros_like(lead))
    assert not target_lead_constraints_valid(
        lead, current, previous, limits, 0.12, 0.03
    ).item()

    rebased_previous = current + lead
    recovered = constrain_target_lead(
        torch.tensor([[0.03]]),
        current,
        rebased_previous,
        limits,
        0.12,
        0.03,
    )
    assert torch.allclose(recovered, torch.tensor([[0.03]]))
    assert target_lead_constraints_valid(
        recovered, current, rebased_previous, limits, 0.12, 0.03
    ).item()


def test_one_infeasible_joint_rebases_the_whole_environment_row():
    current = torch.zeros((2, 3))
    previous = torch.tensor([[0.0, 0.20, 0.0], [0.0, 0.14, 0.0]])
    limits = torch.tensor([[[-1.0, 1.0]] * 3, [[-1.0, 1.0]] * 3])
    raw_lead = torch.full((2, 3), 0.03)

    lead = constrain_target_lead(
        raw_lead, current, previous, limits, 0.12, 0.03
    )

    assert torch.equal(lead[0], torch.zeros(3))
    assert not target_lead_constraints_valid(
        lead, current, previous, limits, 0.12, 0.03
    )[0].item()
    assert target_lead_constraints_valid(
        lead, current, previous, limits, 0.12, 0.03
    )[1].item()


def test_three_control_lookahead_is_current_relative_not_unbounded_cumulative():
    limits = torch.tensor([[[-1.0, 1.0]]])
    raw_lookahead = torch.tensor([[0.03]])

    current = torch.tensor([[0.0]])
    previous = current.clone()
    for _ in range(10):
        lead = constrain_target_lead(
            raw_lookahead, current, previous, limits, 0.12, 0.03
        )
        target = current + lead
        assert torch.allclose(target, torch.tensor([[0.03]]))
        previous = target

    current = torch.tensor([[0.0]])
    previous = current.clone()
    previous_target = None
    for _ in range(10):
        lead = constrain_target_lead(
            raw_lookahead, current, previous, limits, 0.12, 0.03
        )
        target = current + lead
        assert torch.allclose(lead, torch.tensor([[0.03]]))
        if previous_target is not None:
            assert torch.allclose(target - previous_target, torch.tensor([[0.01]]))
        previous_target = target
        previous = target
        current = current + 0.01


def test_axis_progress_ratios_are_reported_independently():
    predicted = torch.tensor([[0.005, 0.0, -0.001]])
    requested = torch.tensor([[0.010, 0.0, -0.004]])
    xy_ratio, z_ratio = translation_progress_ratios(
        predicted,
        requested,
        active_xy=torch.tensor([True]),
        active_z=torch.tensor([True]),
    )

    assert torch.allclose(xy_ratio, torch.tensor([0.5]))
    assert torch.allclose(z_ratio, torch.tensor([0.25]))

    inactive_xy, regressive_z = translation_progress_ratios(
        predicted_translation=torch.tensor([[0.0, 0.0, 0.001]]),
        requested_translation=torch.tensor([[0.0, 0.0, -0.004]]),
        active_xy=torch.tensor([False]),
        active_z=torch.tensor([True]),
    )
    assert torch.allclose(inactive_xy, torch.ones(1))
    assert torch.allclose(regressive_z, torch.tensor([-0.25]))


def test_realized_progress_uses_the_previous_requested_direction():
    actual = torch.tensor([[0.003, 0.004, -0.002], [-0.001, 0.0, 0.001]])
    requested = torch.tensor([[0.006, 0.008, -0.004], [0.002, 0.0, -0.002]])

    requested_xy, requested_z, realized_xy, realized_z = (
        realized_translation_progress(actual, requested)
    )

    assert torch.allclose(requested_xy, torch.tensor([0.010, 0.002]))
    assert torch.allclose(requested_z, torch.tensor([0.004, 0.002]))
    assert torch.allclose(realized_xy, torch.tensor([0.005, -0.001]))
    assert torch.allclose(realized_z, torch.tensor([0.002, -0.001]))


def test_jerk_limiter_respects_independent_xy_and_z_limits():
    previous_velocity = torch.zeros((2, 3))
    previous_acceleration = torch.zeros((2, 3))
    desired_velocity = torch.tensor([[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]])
    step_dt = 1.0 / 30.0

    velocity, acceleration = jerk_limited_cartesian_velocity(
        previous_velocity,
        previous_acceleration,
        desired_velocity,
        step_dt,
        maximum_horizontal_acceleration=0.60,
        maximum_vertical_acceleration=0.45,
        maximum_horizontal_jerk=6.0,
        maximum_vertical_jerk=4.5,
    )

    assert torch.all(torch.norm(acceleration[:, :2], dim=1) <= 6.0 * step_dt + 1.0e-6)
    assert torch.all(torch.abs(acceleration[:, 2]) <= 4.5 * step_dt + 1.0e-6)
    assert torch.allclose(velocity, acceleration * step_dt)


def test_prioritized_dls_preserves_active_progress_while_canceling_feasible_leak():
    jacobian = torch.tensor([[[1.0, 0.0], [0.2, 1.0]]])
    target = torch.tensor([[0.0025, 0.0]])
    primary_mask = torch.tensor([[True, False]])
    solver = build_prioritized_dls_solver(jacobian, primary_mask, damping=0.08)

    delta = solve_prioritized_dls_delta(
        target,
        jacobian,
        primary_mask,
        solver,
        maximum_joint_step=0.03,
    )
    predicted = (jacobian @ delta.unsqueeze(-1)).squeeze(-1)

    assert predicted[0, 0].item() >= 0.99 * target[0, 0].item()
    assert torch.abs(predicted[0, 1]).item() < 1.0e-5


def test_prioritized_dls_does_not_trade_away_active_motion_when_leak_is_unavoidable():
    jacobian = torch.tensor([[[1.0], [1.0]]])
    target = torch.tensor([[0.0025, 0.0]])
    primary_mask = torch.tensor([[True, False]])
    solver = build_prioritized_dls_solver(jacobian, primary_mask, damping=0.08)

    delta = solve_prioritized_dls_delta(
        target,
        jacobian,
        primary_mask,
        solver,
        maximum_joint_step=0.03,
    )
    predicted = (jacobian @ delta.unsqueeze(-1)).squeeze(-1)
    ratio = active_translation_progress_ratio(
        torch.cat((predicted, torch.zeros((1, 1))), dim=1),
        torch.tensor([[0.0025, 0.0, 0.0]]),
        active_xy=torch.tensor([True]),
        active_z=torch.tensor([False]),
    )

    assert ratio.item() > 0.95
    assert predicted[0, 1].item() > 0.001


def test_progress_ratio_remains_well_conditioned_for_tiny_requested_steps():
    requested = torch.tensor([[1.0e-8, 0.0, -1.0e-8]])
    ratio = active_translation_progress_ratio(
        requested.clone(),
        requested,
        active_xy=torch.tensor([True]),
        active_z=torch.tensor([True]),
    )

    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1.0e-6)


def test_jerk_aware_envelope_reaches_deadband_without_emergency_projection():
    step_dt = 1.0 / 30.0
    position = torch.zeros((1, 3))
    target = torch.tensor([[0.0, 0.0, 0.030]])
    velocity = torch.tensor([[0.0, 0.0, 0.075]])
    acceleration = torch.zeros_like(velocity)
    maximum_projection_override = 0.0

    for _ in range(100):
        error = target - position
        desired = apply_stopping_speed_envelope(
            torch.tensor([[0.0, 0.0, 0.075]]),
            error,
            horizontal_deadband=0.005,
            vertical_deadband=0.002,
            maximum_horizontal_acceleration=0.60,
            maximum_vertical_acceleration=0.45,
            maximum_horizontal_jerk=6.0,
            maximum_vertical_jerk=4.5,
        )
        limited_velocity, limited_acceleration = jerk_limited_cartesian_velocity(
            velocity,
            acceleration,
            desired,
            step_dt,
            maximum_horizontal_acceleration=0.60,
            maximum_vertical_acceleration=0.45,
            maximum_horizontal_jerk=6.0,
            maximum_vertical_jerk=4.5,
        )
        step = preserve_safe_goal_directed_step(
            limited_velocity * step_dt,
            error,
            maximum_xy=0.0025,
            maximum_up=0.0025,
            maximum_down=0.0030,
            horizontal_deadband=0.005,
            vertical_deadband=0.002,
        )
        applied_velocity = step / step_dt
        projection_override = torch.norm(applied_velocity - limited_velocity, dim=1)
        maximum_projection_override = max(
            maximum_projection_override, projection_override.item()
        )
        position += step
        velocity = applied_velocity
        acceleration = limited_acceleration

    final_error = torch.abs(target[0, 2] - position[0, 2]).item()
    assert abs(final_error - 0.002) < 1.0e-6
    assert maximum_projection_override < 1.0e-6
