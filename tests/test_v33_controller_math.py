from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
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
advance_post_center_descent_progress = (
    controller_math.advance_post_center_descent_progress
)
actor_owned_carry_phase = controller_math.actor_owned_carry_phase
apply_stopping_speed_envelope = controller_math.apply_stopping_speed_envelope
automatic_release_ready = controller_math.automatic_release_ready
build_prioritized_dls_solver = controller_math.build_prioritized_dls_solver
carry_clearance_fraction = controller_math.carry_clearance_fraction
capped_raw_coupled_lift_rise = controller_math.capped_raw_coupled_lift_rise
saturating_lift_bootstrap_equivalent_rise = (
    controller_math.saturating_lift_bootstrap_equivalent_rise
)
constrain_target_lead = controller_math.constrain_target_lead
coupled_lift_progress = controller_math.coupled_lift_progress
low_clearance_inward_progress_cost = controller_math.low_clearance_inward_progress_cost
prelift_inward_progress_increment = (
    controller_math.prelift_inward_progress_increment
)
persistent_best_signed_progress = controller_math.persistent_best_signed_progress
preserve_safe_goal_directed_step = controller_math.preserve_safe_goal_directed_step
project_goal_directed_step = controller_math.project_goal_directed_step
quintic_smoothstep = controller_math.quintic_smoothstep
realized_translation_progress = controller_math.realized_translation_progress
required_axis_progress_valid = controller_math.required_axis_progress_valid
phase_primary_axis_progress_valid = controller_math.phase_primary_axis_progress_valid
phase_primary_translation_progress_ratio = (
    controller_math.phase_primary_translation_progress_ratio
)
post_grasp_actor_authority = controller_math.post_grasp_actor_authority
post_center_descent_readiness = controller_math.post_center_descent_readiness
scale_delta_to_cartesian_bounds = controller_math.scale_delta_to_cartesian_bounds
select_higher_phase_primary_candidate = (
    controller_math.select_higher_phase_primary_candidate
)
select_phase_axis_fallback = controller_math.select_phase_axis_fallback
select_phase_progress_lead_fallback = (
    controller_math.select_phase_progress_lead_fallback
)
select_phase_primary_axis = controller_math.select_phase_primary_axis
solve_prioritized_dls_delta = controller_math.solve_prioritized_dls_delta
target_lead_constraints_valid = controller_math.target_lead_constraints_valid
target_lead_constraint_intersection = (
    controller_math.target_lead_constraint_intersection
)
translation_progress_ratios = controller_math.translation_progress_ratios


def _persistent_progress_trace(
    qualities: list[float], active: list[bool] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if active is None:
        active = [True] * len(qualities)
    previous = torch.tensor([0.0])
    best = torch.tensor([0.0])
    was_active = torch.tensor([False])
    deltas = []
    best_history = []
    for quality_value, active_value in zip(qualities, active, strict=True):
        quality = torch.tensor([quality_value])
        is_active = torch.tensor([active_value])
        delta, best = persistent_best_signed_progress(
            quality, previous, best, is_active, was_active
        )
        deltas.append(delta)
        best_history.append(best.clone())
        previous = quality
        was_active = is_active
    return torch.cat(deltas), torch.cat(best_history)


@pytest.mark.parametrize(
    "qualities",
    (
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2],
        [0.2, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
    ),
)
def test_pregrasp_progress_cycle_has_zero_undiscounted_return(qualities):
    deltas, _ = _persistent_progress_trace(qualities)

    assert torch.sum(deltas).item() == pytest.approx(0.0, abs=1.0e-6)


def test_pregrasp_progress_does_not_repay_recovery_to_an_old_best():
    deltas, _ = _persistent_progress_trace([0.2, 0.8, 0.2, 0.8, 0.2])

    assert deltas.tolist() == pytest.approx([0.0, 0.6, -0.6, 0.0, -0.6])


def test_pregrasp_progress_keeps_best_across_inactive_retry_periods():
    deltas, best = _persistent_progress_trace(
        [0.2, 0.8, 0.2, 0.2, 0.8],
        [True, True, False, True, True],
    )

    assert best.tolist() == pytest.approx([0.2, 0.8, 0.8, 0.8, 0.8])
    assert deltas.tolist() == pytest.approx([0.0, 0.6, 0.0, 0.0, 0.0])


def test_pregrasp_progress_absorbs_inactive_improvement_without_credit():
    deltas, best = _persistent_progress_trace(
        [0.2, 0.8, 0.8, 0.8],
        [True, False, True, True],
    )

    assert best.tolist() == pytest.approx([0.2, 0.2, 0.8, 0.8])
    assert deltas.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_pregrasp_progress_does_not_pay_repeated_dwell_at_the_best():
    deltas, _ = _persistent_progress_trace([0.2, 0.8, 0.8, 0.8])

    assert deltas.tolist() == pytest.approx([0.0, 0.6, 0.0, 0.0])


def test_pregrasp_progress_keeps_vectorized_environment_masks_independent():
    current = torch.tensor([0.8, 0.7, 0.4])
    previous = torch.tensor([0.2, 0.8, 0.6])
    best = torch.tensor([0.5, 0.9, 0.6])
    active = torch.tensor([True, False, True])
    was_active = torch.tensor([True, True, False])

    delta, updated_best = persistent_best_signed_progress(
        current, previous, best, active, was_active
    )

    assert delta.tolist() == pytest.approx([0.3, 0.0, 0.0])
    assert updated_best.tolist() == pytest.approx([0.8, 0.9, 0.6])


def test_pregrasp_progress_preserves_a_large_one_step_improvement():
    deltas, _ = _persistent_progress_trace([0.2, 0.8])

    assert deltas.tolist() == pytest.approx([0.0, 0.6])


def test_quintic_blend_has_exact_endpoints_and_is_monotonic():
    progress = torch.linspace(0.0, 1.0, 101)
    blend = quintic_smoothstep(progress)

    assert blend[0].item() == 0.0
    assert blend[-1].item() == 1.0
    assert torch.all(blend[1:] >= blend[:-1])


def test_actor_owned_carry_phase_excludes_every_automatic_control_path():
    phase = actor_owned_carry_phase(
        grasp_acquired=torch.tensor([False, True, True, True, True, True]),
        assist_owns_arm=torch.tensor([False, False, True, False, False, False]),
        released=torch.tensor([False, False, False, True, False, False]),
        failed=torch.tensor([False, False, False, False, True, False]),
    )

    assert torch.equal(
        phase,
        torch.tensor([False, True, False, False, False, True]),
    )


def test_automatic_release_requires_every_shared_gate():
    true = torch.ones(6, dtype=torch.bool)
    gates = [true.clone() for _ in range(5)]
    for missing_gate, gate in enumerate(gates, start=1):
        gate[missing_gate] = False

    ready = automatic_release_ready(*gates)

    assert torch.equal(
        ready,
        torch.tensor([True, False, False, False, False, False]),
    )


def test_actor_authority_excludes_only_post_grasp_automatic_control():
    false = torch.zeros(8, dtype=torch.bool)
    gates = [false.clone() for _ in range(7)]
    for automatic_case, gate in enumerate(gates, start=1):
        gate[automatic_case] = True

    authority = post_grasp_actor_authority(*gates)

    assert torch.equal(
        authority,
        torch.tensor([True, False, False, False, False, False, False, False]),
    )


def test_coupled_lift_progress_requires_object_and_tcp_to_rise_together():
    object_progress, tcp_progress, coupled = coupled_lift_progress(
        object_height=torch.tensor([0.680, 0.680, 0.680, 0.740]),
        tcp_height=torch.tensor([0.680, 0.640, 0.740, 0.740]),
        object_origin_height=torch.full((4,), 0.640),
        tcp_origin_height=torch.full((4,), 0.640),
        target_object_height=0.720,
    )

    assert object_progress.tolist() == pytest.approx([0.5, 0.5, 0.5, 1.0])
    assert tcp_progress.tolist() == pytest.approx([0.5, 0.0, 1.0, 1.0])
    assert coupled.tolist() == pytest.approx([0.5, 0.0, 0.5, 1.0])


def test_coupled_lift_progress_handles_a_grasp_near_the_target_height():
    _, _, coupled = coupled_lift_progress(
        object_height=torch.tensor([0.7306]),
        tcp_height=torch.tensor([0.7311]),
        object_origin_height=torch.tensor([0.7295]),
        tcp_origin_height=torch.tensor([0.7300]),
        target_object_height=0.730,
    )

    assert coupled.item() == pytest.approx(1.0)


def test_capped_raw_coupled_lift_rise_uses_the_smaller_physical_rise():
    rise = capped_raw_coupled_lift_rise(
        object_height=torch.tensor([0.657, 0.757, 0.750, 0.7398, 0.741]),
        tcp_height=torch.tensor([0.667, 0.757, 0.750, 0.7398, 0.741]),
        object_origin_height=torch.tensor([0.637, 0.637, 0.700, 0.7298, 0.731]),
        tcp_origin_height=torch.tensor([0.637, 0.637, 0.700, 0.7298, 0.731]),
        target_object_height=0.730,
        maximum_credit_height=0.093,
    )

    assert rise.tolist() == pytest.approx(
        [0.020, 0.093, 0.030, 0.0002, 0.0], abs=1.0e-6
    )


def test_capped_raw_coupled_lift_rise_rejects_one_sided_or_negative_motion():
    rise = capped_raw_coupled_lift_rise(
        object_height=torch.tensor([0.660, 0.640, 0.630]),
        tcp_height=torch.tensor([0.640, 0.660, 0.630]),
        object_origin_height=torch.full((3,), 0.640),
        tcp_origin_height=torch.full((3,), 0.640),
        target_object_height=0.730,
        maximum_credit_height=0.093,
    )

    assert rise.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_capped_raw_coupled_lift_rise_rejects_a_negative_credit_cap():
    with pytest.raises(ValueError, match="non-negative"):
        capped_raw_coupled_lift_rise(
            object_height=torch.tensor([0.640]),
            tcp_height=torch.tensor([0.640]),
            object_origin_height=torch.tensor([0.640]),
            tcp_origin_height=torch.tensor([0.640]),
            target_object_height=0.730,
            maximum_credit_height=-0.001,
        )


def test_saturating_lift_bootstrap_is_smooth_monotonic_and_bounded():
    rise = torch.tensor([0.0, 1.0e-6, 0.001, 0.010, 0.020, 0.093, 1.0])
    bonus = saturating_lift_bootstrap_equivalent_rise(
        rise,
        length_scale=0.010,
        multiplier=4.0,
    )

    assert bonus[0].item() == 0.0
    assert torch.all(bonus[1:] > bonus[:-1])
    assert torch.all(bonus < 0.0400001)
    assert bonus[3].item() == pytest.approx(
        0.04 * (1.0 - torch.exp(torch.tensor(-1.0)).item())
    )
    assert (bonus[1] / rise[1]).item() == pytest.approx(4.0, rel=1.0e-4)


@pytest.mark.parametrize(
    ("length_scale", "multiplier", "message"),
    [(0.0, 4.0, "positive"), (0.010, -1.0, "non-negative")],
)
def test_saturating_lift_bootstrap_rejects_invalid_parameters(
    length_scale,
    multiplier,
    message,
):
    with pytest.raises(ValueError, match=message):
        saturating_lift_bootstrap_equivalent_rise(
            torch.tensor([0.01]),
            length_scale=length_scale,
            multiplier=multiplier,
        )


def test_low_clearance_cost_is_monotonic_and_local_to_the_basket():
    height = torch.linspace(0.700, 0.740, 81)
    clearance = carry_clearance_fraction(height, 0.710, 0.730)

    assert clearance[0].item() == 0.0
    assert clearance[-1].item() == 1.0
    assert torch.all(clearance[1:] >= clearance[:-1])

    previous = torch.tensor([0.22, 0.20, 0.20, 0.20, 0.20, 0.20, 0.16, 0.14, 0.20, 0.20])
    current = torch.tensor(
        [0.19, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, float("nan"), 0.10]
    )
    fraction = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0])
    active = torch.tensor([True, True, True, False, True, True, True, True, True, True])
    was_active = torch.tensor([True, True, False, True, True, True, True, True, True, True])
    cost = low_clearance_inward_progress_cost(
        previous,
        current,
        fraction,
        active,
        was_active,
        near_basket_xy_threshold=0.17,
        max_step_change=0.03,
    )

    assert torch.allclose(
        cost,
        torch.tensor([0.0, 0.01, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0, 0.0, 0.03]),
    )


def test_prelift_inward_progress_charges_only_new_episode_maximum():
    maximum = torch.zeros(1)
    increments = []
    initial_distance = torch.tensor([0.222])
    for distance in (0.222, 0.217, 0.210, 0.172, 0.192, 0.162):
        increment, maximum = prelift_inward_progress_increment(
            initial_distance,
            torch.tensor([distance]),
            maximum,
            torch.tensor([True]),
            free_tolerance=0.010,
            maximum_charged_excursion=0.050,
        )
        increments.append(increment.item())

    assert increments == pytest.approx(
        [0.0, 0.0, 0.002, 0.038, 0.0, 0.010], abs=1.0e-7
    )
    assert sum(increments) == pytest.approx(0.050, abs=1.0e-7)
    assert maximum.item() == pytest.approx(0.060, abs=1.0e-7)


def test_prelift_inward_progress_ignores_outward_tangential_and_inactive_motion():
    initial = torch.full((5,), 0.22)
    current = torch.tensor([0.24, 0.22, 0.18, 0.18, float("nan")])
    previous = torch.zeros(5)
    active = torch.tensor([True, True, True, False, True])

    increment, updated = prelift_inward_progress_increment(
        initial,
        current,
        previous,
        active,
        free_tolerance=0.010,
        maximum_charged_excursion=0.050,
    )

    assert increment.tolist() == pytest.approx([0.0, 0.0, 0.03, 0.0, 0.0])
    assert updated.tolist() == pytest.approx([0.0, 0.0, 0.04, 0.0, 0.0])


def test_prelift_inward_progress_charge_is_bounded_but_raw_max_is_preserved():
    increment, updated = prelift_inward_progress_increment(
        torch.tensor([0.22]),
        torch.tensor([0.10]),
        torch.tensor([0.0]),
        torch.tensor([True]),
        free_tolerance=0.010,
        maximum_charged_excursion=0.050,
    )

    assert increment.item() == pytest.approx(0.050)
    assert updated.item() == pytest.approx(0.120)


@pytest.mark.parametrize(
    ("free_tolerance", "maximum_charged_excursion", "message"),
    [(-0.001, 0.050, "non-negative"), (0.010, 0.0, "positive")],
)
def test_prelift_inward_progress_rejects_invalid_bounds(
    free_tolerance,
    maximum_charged_excursion,
    message,
):
    with pytest.raises(ValueError, match=message):
        prelift_inward_progress_increment(
            torch.tensor([0.22]),
            torch.tensor([0.20]),
            torch.tensor([0.0]),
            torch.tensor([True]),
            free_tolerance,
            maximum_charged_excursion,
        )


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


def test_cartesian_scaling_applies_request_cap_and_error_nonincrease_bound():
    joint_delta = torch.tensor([[0.040, 0.0], [0.0, -0.030]])
    predicted = torch.tensor([[0.004, 0.0, 0.0], [0.0, 0.0, -0.006]])
    error = torch.tensor([[0.100, 0.0, 0.0], [0.0, 0.0, -0.002]])

    scaled, scale, feasible = scale_delta_to_cartesian_bounds(
        joint_delta,
        predicted,
        error,
        requested_translation=torch.tensor(
            [[0.0025, 0.0, 0.0], [0.0, 0.0, -0.006]]
        ),
        require_xy_progress=torch.tensor([True, False]),
        require_z_progress=torch.tensor([False, True]),
        maximum_xy=torch.tensor([0.0025, 0.0100]),
        maximum_up=torch.tensor([0.0025, 0.0100]),
        maximum_down=torch.tensor([0.0030, 0.0100]),
        horizontal_deadband=0.005,
        vertical_deadband=0.001,
        inactive_leak_tolerance=0.001,
    )

    assert torch.all(feasible)
    assert torch.allclose(scale, torch.tensor([0.625, 2.0 / 3.0]))
    assert torch.allclose(scaled, joint_delta * scale[:, None])
    scaled_prediction = predicted * scale[:, None]
    assert torch.allclose(torch.norm(scaled_prediction[:, :2], dim=1)[0], torch.tensor(0.0025))
    assert torch.allclose(
        torch.abs(error[1, 2] - scaled_prediction[1, 2]),
        torch.abs(error[1, 2]),
    )


def test_cartesian_scaling_limits_inactive_axis_leak_with_margin():
    joint_delta = torch.tensor([[0.020, 0.010], [-0.020, 0.010]])
    predicted = torch.tensor(
        [[0.002, 0.0, 0.004], [0.003, 0.004, -0.002]]
    )
    error = torch.tensor([[0.100, 0.0, 0.0], [0.0, 0.0, -0.100]])

    scaled, scale, feasible = scale_delta_to_cartesian_bounds(
        joint_delta,
        predicted,
        error,
        requested_translation=torch.tensor(
            [[0.002, 0.0, 0.0], [0.0, 0.0, -0.002]]
        ),
        require_xy_progress=torch.tensor([True, False]),
        require_z_progress=torch.tensor([False, True]),
        maximum_xy=0.010,
        maximum_up=0.010,
        maximum_down=0.010,
        horizontal_deadband=0.005,
        vertical_deadband=0.002,
        inactive_leak_tolerance=torch.tensor([0.001, 0.001]),
    )

    expected = torch.tensor([0.999 * 0.001 / 0.004, 0.999 * 0.001 / 0.005])
    assert torch.all(feasible)
    assert torch.allclose(scale, expected)
    assert torch.allclose(scaled, joint_delta * scale[:, None])
    scaled_prediction = predicted * scale[:, None]
    assert torch.abs(scaled_prediction[0, 2]).item() < 0.001
    assert torch.norm(scaled_prediction[1, :2]).item() < 0.001


def test_cartesian_scaling_rejects_regressive_active_axes():
    joint_delta = torch.tensor([[0.020, 0.0], [0.0, 0.020]])
    predicted = torch.tensor([[-0.002, 0.0, 0.0], [0.0, 0.0, 0.002]])
    error = torch.tensor([[0.100, 0.0, 0.0], [0.0, 0.0, -0.100]])

    scaled, scale, feasible = scale_delta_to_cartesian_bounds(
        joint_delta,
        predicted,
        error,
        requested_translation=torch.tensor(
            [[0.002, 0.0, 0.0], [0.0, 0.0, -0.002]]
        ),
        require_xy_progress=torch.tensor([True, False]),
        require_z_progress=torch.tensor([False, True]),
        maximum_xy=0.0025,
        maximum_up=0.0025,
        maximum_down=0.0030,
        horizontal_deadband=0.005,
        vertical_deadband=0.002,
        inactive_leak_tolerance=0.001,
    )

    assert not torch.any(feasible)
    assert torch.equal(scale, torch.zeros(2))
    assert torch.equal(scaled, torch.zeros_like(joint_delta))


def test_required_axis_progress_distinguishes_coupled_and_single_axis_ownership():
    requested = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.002, 0.0, 0.0],
            [0.002, 0.0, -0.002],
            [0.002, 0.0, -0.002],
            [0.002, 0.0, -0.002],
            [0.002, 0.0, 0.0],
        ]
    )
    predicted = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0001],
            [0.001, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.0, 0.0, -0.001],
            [-0.001, 0.0, 0.0],
        ]
    )
    require_xy = torch.tensor([False, True, True, True, False, True])
    require_z = torch.tensor([False, False, True, False, True, False])

    valid, xy_ratio, z_ratio = required_axis_progress_valid(
        predicted,
        requested,
        require_xy,
        require_z,
    )

    assert torch.equal(
        valid,
        torch.tensor([True, False, False, True, True, False]),
    )
    assert xy_ratio[3] > 0.0
    assert z_ratio[4] > 0.0


def test_required_axis_progress_rejects_numerical_noops_and_nonfinite_values():
    requested = torch.tensor([[0.002, 0.0, 0.0]] * 4)
    predicted = torch.tensor(
        [
            [0.002 * 0.5e-6, 0.0, 0.0],
            [0.002 * 2.0e-6, 0.0, 0.0],
            [float("nan"), 0.0, 0.0],
            [float("inf"), 0.0, 0.0],
        ]
    )
    required = torch.ones(4, dtype=torch.bool)

    valid, _, _ = required_axis_progress_valid(
        predicted,
        requested,
        required,
        torch.zeros_like(required),
    )

    assert torch.equal(valid, torch.tensor([False, True, False, False]))


def test_required_axis_progress_applies_the_configured_candidate_floor_per_axis():
    requested = torch.tensor([[0.002, 0.0, -0.002]] * 4)
    predicted = torch.tensor(
        [
            [0.002 * 0.249, 0.0, -0.002 * 0.80],
            [0.002 * 0.25, 0.0, -0.002 * 0.80],
            [0.002 * 0.80, 0.0, -0.002 * 0.249],
            [0.002 * 0.80, 0.0, -0.002 * 0.25],
        ]
    )
    required = torch.ones(4, dtype=torch.bool)

    valid, _, _ = required_axis_progress_valid(
        predicted,
        requested,
        required,
        required,
        minimum_progress_ratio=0.25,
    )

    assert torch.equal(valid, torch.tensor([False, True, False, True]))


def test_v39_phase_primary_progress_uses_smaller_positive_companion_floor():
    requested = torch.tensor([[0.002, 0.0, -0.003]] * 7)
    predicted = torch.tensor(
        [
            [0.0015, 0.0, -0.0003],
            [0.000498, 0.0, -0.0024],
            [0.0016, 0.0, -0.000147],
            [0.0002, 0.0, -0.0021],
            [0.0016, 0.0, -0.000747],
            [0.0016, 0.0, 0.0003],
            [float("nan"), 0.0, -0.0021],
        ]
    )
    active = torch.ones(7, dtype=torch.bool)
    prefer_z = torch.tensor([False, False, False, True, True, False, True])

    valid, xy_ratio, z_ratio = phase_primary_axis_progress_valid(
        predicted,
        requested,
        active,
        active,
        prefer_z,
        minimum_primary_progress_ratio=0.25,
        minimum_companion_progress_ratio=0.05,
    )

    assert torch.equal(
        valid,
        torch.tensor([True, False, False, True, False, False, False]),
    )
    assert torch.isclose(xy_ratio[0], torch.tensor(0.75))
    assert torch.isclose(z_ratio[3], torch.tensor(0.70))


def test_v40_phase_primary_score_does_not_rank_by_the_weaker_companion_axis():
    requested = torch.tensor([[0.002, 0.0, -0.003]] * 2)
    predicted = torch.tensor(
        [
            [0.00011, 0.0, -0.0024],
            [0.0004, 0.0, -0.00075],
        ]
    )
    active = torch.ones(2, dtype=torch.bool)
    prefer_z = torch.ones(2, dtype=torch.bool)

    valid, _, _ = phase_primary_axis_progress_valid(
        predicted,
        requested,
        active,
        active,
        prefer_z,
        minimum_primary_progress_ratio=0.25,
        minimum_companion_progress_ratio=0.05,
    )
    phase_score = phase_primary_translation_progress_ratio(
        predicted,
        requested,
        active,
        active,
        prefer_z,
    )
    weakest_axis_score = active_translation_progress_ratio(
        predicted,
        requested,
        active,
        active,
    )

    assert torch.all(valid)
    assert torch.allclose(phase_score, torch.tensor([0.80, 0.25]))
    assert torch.allclose(weakest_axis_score, torch.tensor([0.055, 0.20]))
    assert phase_score[0] > phase_score[1]
    assert weakest_axis_score[0] < weakest_axis_score[1]


def test_v40_z_priority_upgrade_requires_a_safe_strict_primary_improvement():
    upgrade = select_higher_phase_primary_candidate(
        current_valid=torch.tensor([True, True, True, False, True]),
        current_ratio=torch.tensor([0.25, 0.80, 0.25, 0.25, float("nan")]),
        candidate_valid=torch.tensor([True, True, False, True, True]),
        candidate_ratio=torch.tensor([0.80, 0.25, 0.90, 0.90, 0.90]),
        phase_allows_candidate=torch.tensor([True, True, True, True, True]),
    )

    assert torch.equal(
        upgrade,
        torch.tensor([True, False, False, False, False]),
    )


def test_v47_lead_fallback_upgrades_only_z_priority_to_better_progress():
    (
        use_stabilization,
        use_incremental,
        use_hold,
        progress_upgrade,
    ) = select_phase_progress_lead_fallback(
        preferred_valid=torch.tensor([True, False, False, False, False, False]),
        stabilization_valid=torch.tensor([True, True, True, True, False, False]),
        stabilization_ratio=torch.tensor([0.10, 0.10, 0.10, 0.80, 0.0, 0.0]),
        incremental_valid=torch.tensor([True, True, True, True, True, False]),
        incremental_ratio=torch.tensor([0.90, 0.90, 0.90, 0.25, 0.90, 0.0]),
        prefer_incremental_progress=torch.tensor(
            [True, False, True, True, True, True]
        ),
    )

    assert torch.equal(
        use_stabilization,
        torch.tensor([False, True, False, True, False, False]),
    )
    assert torch.equal(
        use_incremental,
        torch.tensor([False, False, True, False, True, False]),
    )
    assert torch.equal(
        use_hold,
        torch.tensor([False, False, False, False, False, True]),
    )
    assert torch.equal(
        progress_upgrade,
        torch.tensor([False, False, True, False, False, False]),
    )


def test_v47_lead_fallback_keeps_stabilization_on_equal_progress():
    use_stabilization, use_incremental, use_hold, progress_upgrade = (
        select_phase_progress_lead_fallback(
            preferred_valid=torch.tensor([False]),
            stabilization_valid=torch.tensor([True]),
            stabilization_ratio=torch.tensor([0.50]),
            incremental_valid=torch.tensor([True]),
            incremental_ratio=torch.tensor([0.50]),
            prefer_incremental_progress=torch.tensor([True]),
        )
    )

    assert use_stabilization.item()
    assert not use_incremental.item()
    assert not use_hold.item()
    assert not progress_upgrade.item()


def test_v47_non_z_lead_fallback_is_identical_to_the_previous_priority():
    generator = torch.Generator().manual_seed(47)
    sample_count = 4096
    preferred_valid = torch.rand(sample_count, generator=generator) > 0.5
    stabilization_valid = torch.rand(sample_count, generator=generator) > 0.5
    incremental_valid = torch.rand(sample_count, generator=generator) > 0.5
    stabilization_ratio = torch.rand(sample_count, generator=generator)
    incremental_ratio = torch.rand(sample_count, generator=generator)

    use_stabilization, use_incremental, use_hold, progress_upgrade = (
        select_phase_progress_lead_fallback(
            preferred_valid,
            stabilization_valid,
            stabilization_ratio,
            incremental_valid,
            incremental_ratio,
            prefer_incremental_progress=torch.zeros(
                sample_count, dtype=torch.bool
            ),
        )
    )

    unresolved = ~preferred_valid
    assert torch.equal(
        use_stabilization, unresolved & stabilization_valid
    )
    assert torch.equal(
        use_incremental,
        unresolved & ~stabilization_valid & incremental_valid,
    )
    assert torch.equal(
        use_hold,
        unresolved & ~stabilization_valid & ~incremental_valid,
    )
    assert not torch.any(progress_upgrade)


def test_v39_single_axis_still_requires_the_full_primary_floor():
    requested = torch.tensor(
        [[0.002, 0.0, -0.003], [0.002, 0.0, -0.003]]
    )
    predicted = torch.tensor(
        [[0.0004, 0.0, 0.0], [0.0, 0.0, -0.00075]]
    )
    require_xy = torch.tensor([True, False])
    require_z = torch.tensor([False, True])

    valid, _, _ = phase_primary_axis_progress_valid(
        predicted,
        requested,
        require_xy,
        require_z,
        prefer_z=torch.tensor([False, True]),
        minimum_primary_progress_ratio=0.25,
        minimum_companion_progress_ratio=0.05,
    )

    assert torch.equal(valid, torch.tensor([False, True]))


def test_v39_phase_progress_rejects_an_unowned_stabilization_request():
    valid, _, _ = phase_primary_axis_progress_valid(
        predicted_translation=torch.zeros((1, 3)),
        requested_translation=torch.tensor([[0.002, 0.0, -0.003]]),
        active_xy=torch.zeros(1, dtype=torch.bool),
        active_z=torch.zeros(1, dtype=torch.bool),
        prefer_z=torch.zeros(1, dtype=torch.bool),
        minimum_primary_progress_ratio=0.25,
        minimum_companion_progress_ratio=0.05,
    )

    assert not valid.item()


def test_v39_phase_primary_axis_switches_only_in_the_descent_safe_region():
    active_xy = torch.tensor([True, True, True, False, False])
    active_z = torch.tensor([True, True, False, True, False])
    prefer_z = torch.tensor([False, True, True, False, True])

    primary_xy, primary_z = select_phase_primary_axis(
        active_xy, active_z, prefer_z
    )

    assert torch.equal(primary_xy, torch.tensor([True, False, True, False, False]))
    assert torch.equal(primary_z, torch.tensor([False, True, False, True, False]))


def test_phase_axis_fallback_requires_the_current_release_basin_for_z_priority():
    unresolved = torch.ones(5, dtype=torch.bool)
    xy_valid = torch.tensor([True, True, True, False, False])
    z_valid = torch.tensor([True, True, True, True, False])
    centered = torch.tensor([False, True, True, False, True])
    inside_release = torch.tensor([False, True, False, False, True])

    use_xy, use_z = select_phase_axis_fallback(
        unresolved, xy_valid, z_valid, centered & inside_release
    )

    assert torch.equal(use_xy, torch.tensor([True, False, True, False, False]))
    assert torch.equal(use_z, torch.tensor([False, True, False, True, False]))


def test_v40_post_center_descent_ramp_is_continuous_pausable_and_bounded():
    ramp_steps = 12
    progress = torch.tensor([0.7795])
    xy_progress = progress.clone()
    start = torch.zeros_like(progress)
    elapsed = torch.zeros(1, dtype=torch.long)
    started = torch.zeros(1, dtype=torch.bool)
    history = [progress.item()]

    for _ in range(4):
        progress, start, elapsed, started = advance_post_center_descent_progress(
            progress,
            xy_progress,
            start,
            elapsed,
            started,
            ramp_candidate=torch.ones(1, dtype=torch.bool),
            ramp_active=torch.ones(1, dtype=torch.bool),
            ramp_total_steps=ramp_steps,
        )
        history.append(progress.item())

    paused = progress.clone()
    for paused_xy_progress in (0.82, 0.86, 0.92, 0.98):
        progress, start, elapsed, started = advance_post_center_descent_progress(
            progress,
            torch.tensor([paused_xy_progress]),
            start,
            elapsed,
            started,
            ramp_candidate=torch.ones(1, dtype=torch.bool),
            ramp_active=torch.zeros(1, dtype=torch.bool),
            ramp_total_steps=ramp_steps,
        )
    assert torch.equal(progress, paused)
    assert elapsed.item() == 4

    for _ in range(8):
        progress, start, elapsed, started = advance_post_center_descent_progress(
            progress,
            xy_progress,
            start,
            elapsed,
            started,
            ramp_candidate=torch.ones(1, dtype=torch.bool),
            ramp_active=torch.ones(1, dtype=torch.bool),
            ramp_total_steps=ramp_steps,
        )
        history.append(progress.item())

    assert started.item()
    assert elapsed.item() == ramp_steps
    assert progress.item() == 1.0
    assert all(after >= before for before, after in zip(history, history[1:]))
    heights = 0.75 + torch.tensor(history) * (0.67 - 0.75)
    assert torch.max(torch.abs(heights[1:] - heights[:-1])) <= 0.003


def test_v39_post_center_descent_freezes_before_start_inside_the_candidate_basin():
    candidate, ready, _, _ = post_center_descent_readiness(
        descent_active=torch.ones(1, dtype=torch.bool),
        place_centered=torch.zeros(1, dtype=torch.bool),
        distance_xy=torch.tensor([0.02]),
        object_xy_speed=torch.tensor([0.01]),
        target_tracking_error=torch.tensor([0.001]),
        maximum_xy_distance=0.03,
        maximum_xy_speed=0.05,
        maximum_tracking_error=0.012,
    )
    progress, start, elapsed, started = advance_post_center_descent_progress(
        current_progress=torch.tensor([0.80]),
        xy_progress=torch.tensor([0.98]),
        ramp_start_progress=torch.zeros(1),
        ramp_elapsed_steps=torch.zeros(1, dtype=torch.long),
        ramp_started=torch.zeros(1, dtype=torch.bool),
        ramp_candidate=candidate,
        ramp_active=ready,
        ramp_total_steps=18,
    )

    assert progress.item() == pytest.approx(0.80)
    assert candidate.item()
    assert not ready.item()
    assert start.item() == 0.0
    assert elapsed.item() == 0
    assert not started.item()

    progress, _, _, _ = advance_post_center_descent_progress(
        current_progress=torch.tensor([0.80]),
        xy_progress=torch.tensor([0.90]),
        ramp_start_progress=torch.zeros(1),
        ramp_elapsed_steps=torch.zeros(1, dtype=torch.long),
        ramp_started=torch.zeros(1, dtype=torch.bool),
        ramp_candidate=torch.zeros(1, dtype=torch.bool),
        ramp_active=torch.zeros(1, dtype=torch.bool),
        ramp_total_steps=18,
    )

    assert progress.item() == pytest.approx(0.90)


def test_v39_post_center_descent_waits_for_stability_and_target_tracking():
    candidate, ready, stability_pause, tracking_pause = post_center_descent_readiness(
        descent_active=torch.tensor([False, True, True, True, True, True]),
        place_centered=torch.tensor([True, False, True, True, True, True]),
        distance_xy=torch.tensor([0.02, 0.02, 0.04, 0.02, 0.02, 0.02]),
        object_xy_speed=torch.tensor([0.01, 0.01, 0.01, 0.06, 0.01, 0.06]),
        target_tracking_error=torch.tensor([0.001, 0.001, 0.001, 0.001, 0.013, 0.013]),
        maximum_xy_distance=0.03,
        maximum_xy_speed=0.05,
        maximum_tracking_error=0.012,
    )

    assert torch.equal(
        candidate, torch.tensor([False, True, False, True, True, True])
    )
    assert torch.equal(
        ready, torch.tensor([False, False, False, False, False, False])
    )
    assert torch.equal(
        stability_pause, torch.tensor([False, False, False, True, False, True])
    )
    assert torch.equal(
        tracking_pause, torch.tensor([False, False, False, False, True, False])
    )

    candidate, ready, stability_pause, tracking_pause = post_center_descent_readiness(
        descent_active=torch.ones(1, dtype=torch.bool),
        place_centered=torch.ones(1, dtype=torch.bool),
        distance_xy=torch.tensor([0.03]),
        object_xy_speed=torch.tensor([0.05]),
        target_tracking_error=torch.tensor([0.012]),
        maximum_xy_distance=0.03,
        maximum_xy_speed=0.05,
        maximum_tracking_error=0.012,
    )

    assert candidate.item()
    assert ready.item()
    assert not stability_pause.item()
    assert not tracking_pause.item()


def test_cartesian_scaling_allows_zero_prediction_on_the_other_active_axis():
    joint_delta = torch.tensor([[0.020, 0.010], [-0.020, 0.010]])
    predicted = torch.tensor([[0.002, 0.0, 0.0], [0.0, 0.0, -0.002]])
    error = torch.tensor([[0.100, 0.0, -0.050], [0.100, 0.0, -0.050]])

    scaled, scale, feasible = scale_delta_to_cartesian_bounds(
        joint_delta,
        predicted,
        error,
        requested_translation=torch.tensor(
            [[0.002, 0.0, -0.002], [0.002, 0.0, -0.002]]
        ),
        require_xy_progress=torch.tensor([True, False]),
        require_z_progress=torch.tensor([False, True]),
        maximum_xy=0.0025,
        maximum_up=0.0025,
        maximum_down=0.0030,
        horizontal_deadband=0.005,
        vertical_deadband=0.002,
        inactive_leak_tolerance=0.001,
    )

    assert torch.all(feasible)
    assert torch.allclose(scale, torch.ones(2))
    assert torch.equal(scaled, joint_delta)


def test_cartesian_scaling_rejects_candidates_below_the_progress_floor():
    joint_delta = torch.tensor([[0.020, 0.010], [0.020, 0.010]])
    predicted = torch.tensor([[0.0004, 0.0, 0.0], [0.0006, 0.0, 0.0]])
    requested = torch.tensor([[0.002, 0.0, 0.0]] * 2)

    scaled, scale, feasible = scale_delta_to_cartesian_bounds(
        joint_delta,
        predicted,
        position_error=torch.tensor([[0.100, 0.0, 0.0]] * 2),
        requested_translation=requested,
        require_xy_progress=torch.ones(2, dtype=torch.bool),
        require_z_progress=torch.zeros(2, dtype=torch.bool),
        maximum_xy=0.0025,
        maximum_up=0.0025,
        maximum_down=0.0030,
        horizontal_deadband=0.005,
        vertical_deadband=0.002,
        inactive_leak_tolerance=0.001,
        minimum_progress_ratio=0.25,
    )

    assert torch.equal(feasible, torch.tensor([False, True]))
    assert torch.equal(scale, torch.tensor([0.0, 1.0]))
    assert torch.equal(scaled[0], torch.zeros_like(scaled[0]))
    assert torch.equal(scaled[1], joint_delta[1])


def test_v39_cartesian_scaling_honors_phase_primary_and_companion_floors():
    joint_delta = torch.tensor([[0.020, 0.010], [0.020, -0.010]])
    requested = torch.tensor([[0.002, 0.0, -0.003]] * 2)
    predicted = torch.tensor(
        [[0.0015, 0.0, -0.0003], [0.0002, 0.0, -0.0021]]
    )

    scaled, scale, feasible = scale_delta_to_cartesian_bounds(
        joint_delta,
        predicted,
        position_error=torch.tensor([[0.100, 0.0, -0.050]] * 2),
        requested_translation=requested,
        require_xy_progress=torch.ones(2, dtype=torch.bool),
        require_z_progress=torch.ones(2, dtype=torch.bool),
        maximum_xy=0.0025,
        maximum_up=0.0025,
        maximum_down=0.0030,
        horizontal_deadband=0.005,
        vertical_deadband=0.002,
        inactive_leak_tolerance=0.001,
        minimum_progress_ratio=0.25,
        prefer_z_progress=torch.tensor([False, True]),
        minimum_companion_progress_ratio=0.05,
    )

    assert torch.all(feasible)
    assert torch.allclose(scale, torch.ones(2))
    assert torch.equal(scaled, joint_delta)


def test_v39_cartesian_scaling_rejects_incomplete_phase_options():
    common = dict(
        joint_delta=torch.tensor([[0.02, 0.01]]),
        predicted_translation=torch.tensor([[0.0015, 0.0, -0.0003]]),
        position_error=torch.tensor([[0.10, 0.0, -0.05]]),
        requested_translation=torch.tensor([[0.002, 0.0, -0.003]]),
        require_xy_progress=torch.ones(1, dtype=torch.bool),
        require_z_progress=torch.ones(1, dtype=torch.bool),
        maximum_xy=0.0025,
        maximum_up=0.0025,
        maximum_down=0.0030,
        horizontal_deadband=0.005,
        vertical_deadband=0.002,
        inactive_leak_tolerance=0.001,
        minimum_progress_ratio=0.25,
    )

    with pytest.raises(ValueError, match="must be provided together"):
        scale_delta_to_cartesian_bounds(
            **common,
            prefer_z_progress=torch.zeros(1, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="must be provided together"):
        scale_delta_to_cartesian_bounds(
            **common,
            minimum_companion_progress_ratio=0.05,
        )


def test_cartesian_scaling_rejects_missing_required_axis_and_leak_only_motion():
    joint_delta = torch.tensor([[0.020, 0.010], [0.010, 0.0]])
    predicted = torch.tensor([[0.002, 0.0, 0.0], [0.0, 0.0, 0.0001]])
    error = torch.tensor([[0.100, 0.0, -0.050], [0.100, 0.0, 0.0]])
    requested = torch.tensor([[0.002, 0.0, -0.002], [0.002, 0.0, 0.0]])

    scaled, scale, feasible = scale_delta_to_cartesian_bounds(
        joint_delta,
        predicted,
        error,
        requested_translation=requested,
        require_xy_progress=torch.tensor([True, True]),
        require_z_progress=torch.tensor([True, False]),
        maximum_xy=0.0025,
        maximum_up=0.0025,
        maximum_down=0.0030,
        horizontal_deadband=0.005,
        vertical_deadband=0.002,
        inactive_leak_tolerance=0.001,
    )

    assert not torch.any(feasible)
    assert torch.equal(scale, torch.zeros(2))
    assert torch.equal(scaled, torch.zeros_like(joint_delta))


def test_cartesian_scaling_rejects_nonfinite_and_zero_candidates():
    joint_delta = torch.tensor(
        [[0.0, 0.0], [float("nan"), 0.010], [0.010, 0.0], [0.010, 0.0]]
    )
    predicted = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.002, 0.0, 0.0],
            [float("inf"), 0.0, 0.0],
            [0.002, 0.0, 0.0],
        ]
    )
    error = torch.tensor(
        [
            [0.100, 0.0, 0.0],
            [0.100, 0.0, 0.0],
            [0.100, 0.0, 0.0],
            [float("nan"), 0.0, 0.0],
        ]
    )

    scaled, scale, feasible = scale_delta_to_cartesian_bounds(
        joint_delta,
        predicted,
        error,
        requested_translation=torch.tensor(
            [[0.002, 0.0, 0.0]] * 4
        ),
        require_xy_progress=torch.ones(4, dtype=torch.bool),
        require_z_progress=torch.zeros(4, dtype=torch.bool),
        maximum_xy=0.0025,
        maximum_up=0.0025,
        maximum_down=0.0030,
        horizontal_deadband=0.005,
        vertical_deadband=0.002,
        inactive_leak_tolerance=0.001,
    )

    assert not torch.any(feasible)
    assert torch.equal(scale, torch.zeros(4))
    assert torch.equal(scaled, torch.zeros_like(joint_delta))


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


def test_v40_phase_primary_mask_preserves_the_selected_dls_axis_first():
    jacobian = torch.zeros((2, 6, 2))
    jacobian[:, 0, :] = torch.tensor([1.0, 1.0])
    jacobian[:, 2, :] = torch.tensor([1.0, -1.0])
    target = torch.zeros((2, 6))
    target[:, 0] = 0.006
    target[:, 2] = 0.002
    active = torch.ones(2, dtype=torch.bool)
    primary_xy, primary_z = select_phase_primary_axis(
        active,
        active,
        prefer_z=torch.tensor([False, True]),
    )
    primary_mask = torch.zeros_like(target, dtype=torch.bool)
    primary_mask[:, :2] = primary_xy[:, None]
    primary_mask[:, 2] = primary_z

    solver = build_prioritized_dls_solver(
        jacobian,
        primary_mask,
        damping=1.0e-4,
    )
    delta = solve_prioritized_dls_delta(
        target,
        jacobian,
        primary_mask,
        solver,
        maximum_joint_step=0.003,
    )
    predicted = (jacobian @ delta.unsqueeze(-1)).squeeze(-1)
    xy_ratio, z_ratio = translation_progress_ratios(
        predicted[:, :3],
        target[:, :3],
        active,
        active,
    )

    assert xy_ratio[0] > xy_ratio[1]
    assert z_ratio[1] > z_ratio[0]
    assert torch.all(xy_ratio > 0.0)
    assert torch.all(z_ratio > 0.0)


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
