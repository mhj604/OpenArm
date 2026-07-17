"""Pure tensor helpers for the staged pick-place Cartesian controller."""

from __future__ import annotations

import torch


def quintic_smoothstep(progress: torch.Tensor) -> torch.Tensor:
    """Return a C2-continuous blend coefficient in the closed interval [0, 1]."""

    progress = torch.clamp(progress, min=0.0, max=1.0)
    return progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))


def limit_vector_norm(vector: torch.Tensor, maximum: float | torch.Tensor) -> torch.Tensor:
    """Limit row-wise vector norms without changing their direction."""

    maximum_tensor = torch.as_tensor(maximum, device=vector.device, dtype=vector.dtype)
    if maximum_tensor.ndim == 0:
        maximum_tensor = maximum_tensor.expand(vector.shape[0])
    maximum_tensor = maximum_tensor.reshape(vector.shape[0], 1)
    norm = torch.norm(vector, dim=1, keepdim=True)
    scale = torch.clamp(maximum_tensor / torch.clamp(norm, min=1.0e-9), max=1.0)
    return vector * scale


def scale_by_max_abs(vector: torch.Tensor, maximum: float) -> torch.Tensor:
    """Uniformly cap row-wise absolute components while preserving direction."""

    largest = torch.amax(torch.abs(vector), dim=1, keepdim=True)
    scale = torch.clamp(maximum / torch.clamp(largest, min=1.0e-9), max=1.0)
    return vector * scale


def scale_delta_to_joint_limits(
    current_position: torch.Tensor,
    delta: torch.Tensor,
    joint_limits: torch.Tensor,
) -> torch.Tensor:
    """Uniformly scale a delta so its endpoint does not cross a joint limit."""

    infinite = torch.full_like(delta, float("inf"))
    upper_scale = torch.where(
        delta > 0.0,
        (joint_limits[:, :, 1] - current_position)
        / torch.clamp(delta, min=1.0e-9),
        infinite,
    )
    lower_scale = torch.where(
        delta < 0.0,
        (joint_limits[:, :, 0] - current_position)
        / torch.clamp(delta, max=-1.0e-9),
        infinite,
    )
    scale = torch.amin(torch.minimum(upper_scale, lower_scale), dim=1, keepdim=True)
    return delta * torch.clamp(scale, min=0.0, max=1.0)


def carry_clearance_fraction(
    object_height: torch.Tensor,
    clearance_start_height: float,
    clearance_full_height: float,
) -> torch.Tensor:
    """Return a smooth low-to-full carry-clearance fraction."""

    if clearance_full_height <= clearance_start_height:
        raise ValueError(
            "clearance_full_height must be greater than clearance_start_height"
        )
    normalized_height = (object_height - clearance_start_height) / (
        clearance_full_height - clearance_start_height
    )
    return quintic_smoothstep(normalized_height)


def actor_owned_carry_phase(
    grasp_acquired: torch.Tensor,
    assist_owns_arm: torch.Tensor,
    released: torch.Tensor,
    failed: torch.Tensor,
) -> torch.Tensor:
    """Return post-grasp motion still owned by the learned actor."""

    return grasp_acquired & ~assist_owns_arm & ~released & ~failed


def automatic_release_ready(
    grasp_acquired: torch.Tensor,
    lift_completed: torch.Tensor,
    validated_transport: torch.Tensor,
    release_ready_reached: torch.Tensor,
    release_ready: torch.Tensor,
) -> torch.Tensor:
    """Return the shared release gate used by the arm and gripper action terms."""

    return (
        grasp_acquired
        & lift_completed
        & validated_transport
        & release_ready_reached
        & release_ready
    )


def post_grasp_actor_authority(
    assist_active: torch.Tensor,
    hold_recovery: torch.Tensor,
    handoff_pending: torch.Tensor,
    imminent_automatic_release: torch.Tensor,
    release_commanded: torch.Tensor,
    released: torch.Tensor,
    failed: torch.Tensor,
) -> torch.Tensor:
    """Return actor authority after excluding post-grasp automatic arm control."""

    return ~(
        assist_active
        | hold_recovery
        | handoff_pending
        | imminent_automatic_release
        | release_commanded
        | released
        | failed
    )


def coupled_lift_progress(
    object_height: torch.Tensor,
    tcp_height: torch.Tensor,
    object_origin_height: torch.Tensor,
    tcp_origin_height: torch.Tensor,
    target_object_height: float,
    minimum_height_span: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized object, TCP, and coupled post-grasp lift progress."""

    if minimum_height_span <= 0.0:
        raise ValueError("minimum lift height span must be positive")
    height_span = torch.clamp(
        target_object_height - object_origin_height,
        min=minimum_height_span,
    )
    object_progress = torch.clamp(
        (object_height - object_origin_height) / height_span,
        min=0.0,
        max=1.0,
    )
    tcp_progress = torch.clamp(
        (tcp_height - tcp_origin_height) / height_span,
        min=0.0,
        max=1.0,
    )
    return object_progress, tcp_progress, torch.minimum(object_progress, tcp_progress)


def capped_raw_coupled_lift_rise(
    object_height: torch.Tensor,
    tcp_height: torch.Tensor,
    object_origin_height: torch.Tensor,
    tcp_origin_height: torch.Tensor,
    target_object_height: float,
    maximum_credit_height: float,
) -> torch.Tensor:
    """Return bounded lift distance achieved jointly by the object and TCP."""

    if maximum_credit_height < 0.0:
        raise ValueError("maximum_credit_height must be non-negative")
    credit_cap = torch.clamp(
        target_object_height - object_origin_height,
        min=0.0,
        max=maximum_credit_height,
    )
    object_rise = torch.clamp(object_height - object_origin_height, min=0.0)
    tcp_rise = torch.clamp(tcp_height - tcp_origin_height, min=0.0)
    return torch.minimum(torch.minimum(object_rise, tcp_rise), credit_cap)


def saturating_lift_bootstrap_equivalent_rise(
    coupled_rise: torch.Tensor,
    length_scale: float,
    multiplier: float,
) -> torch.Tensor:
    """Return a smooth bounded equivalent-rise bonus for early lift motion."""

    if not length_scale > 0.0:
        raise ValueError("length_scale must be positive")
    if not multiplier >= 0.0:
        raise ValueError("multiplier must be non-negative")
    nonnegative_rise = torch.clamp(coupled_rise, min=0.0)
    return -multiplier * length_scale * torch.expm1(
        -nonnegative_rise / length_scale
    )


def persistent_best_signed_progress(
    current_quality: torch.Tensor,
    previous_quality: torch.Tensor,
    best_quality: torch.Tensor,
    active: torch.Tensor,
    was_active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pay new episode-best quality once and charge every active regression."""

    continued = active & was_active
    new_best_improvement = torch.clamp(current_quality - best_quality, min=0.0)
    signed_regression = torch.clamp(current_quality - previous_quality, max=0.0)
    progress = torch.where(
        continued,
        new_best_improvement + signed_regression,
        torch.zeros_like(current_quality),
    )
    updated_best = torch.where(
        active,
        torch.maximum(best_quality, current_quality),
        best_quality,
    )
    return progress, updated_best


def low_clearance_inward_progress_cost(
    previous_distance_xy: torch.Tensor,
    distance_xy: torch.Tensor,
    clearance_fraction: torch.Tensor,
    preassist_active: torch.Tensor,
    was_preassist_active: torch.Tensor,
    near_basket_xy_threshold: float,
    max_step_change: float,
) -> torch.Tensor:
    """Measure low-clearance inward progress only near the basket."""

    if near_basket_xy_threshold <= 0.0:
        raise ValueError("near_basket_xy_threshold must be positive")

    finite = (
        torch.isfinite(previous_distance_xy)
        & torch.isfinite(distance_xy)
        & torch.isfinite(clearance_fraction)
    )
    previous_local_distance = torch.minimum(
        previous_distance_xy,
        torch.full_like(previous_distance_xy, near_basket_xy_threshold),
    )
    local_distance = torch.minimum(
        distance_xy,
        torch.full_like(distance_xy, near_basket_xy_threshold),
    )
    inward_progress = torch.clamp(
        previous_local_distance - local_distance,
        min=0.0,
        max=max_step_change,
    )
    clearance_weight = 1.0 - torch.clamp(clearance_fraction, min=0.0, max=1.0)
    return torch.where(
        finite & preassist_active & was_preassist_active,
        clearance_weight * inward_progress,
        torch.zeros_like(distance_xy),
    )


def required_axis_progress_valid(
    predicted_translation: torch.Tensor,
    requested_translation: torch.Tensor,
    require_xy: torch.Tensor,
    require_z: torch.Tensor,
    minimum_progress_ratio: float = 1.0e-6,
    request_active_epsilon: float = 1.0e-9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate the minimum progress ratio on each axis a candidate owns."""

    if minimum_progress_ratio <= 0.0:
        raise ValueError("minimum_progress_ratio must be positive")

    batch_size = predicted_translation.shape[0]
    require_xy = torch.as_tensor(
        require_xy, device=predicted_translation.device, dtype=torch.bool
    ).reshape(batch_size)
    require_z = torch.as_tensor(
        require_z, device=predicted_translation.device, dtype=torch.bool
    ).reshape(batch_size)

    requested_xy_squared = torch.sum(requested_translation[:, :2] ** 2, dim=1)
    requested_z_squared = requested_translation[:, 2] ** 2
    xy_active = requested_xy_squared > request_active_epsilon**2
    z_active = torch.abs(requested_translation[:, 2]) > request_active_epsilon
    xy_ratio = torch.sum(
        predicted_translation[:, :2] * requested_translation[:, :2], dim=1
    ) / torch.clamp(requested_xy_squared, min=request_active_epsilon**2)
    z_ratio = (
        predicted_translation[:, 2]
        * requested_translation[:, 2]
        / torch.clamp(requested_z_squared, min=request_active_epsilon**2)
    )

    finite = (
        torch.all(torch.isfinite(predicted_translation), dim=1)
        & torch.all(torch.isfinite(requested_translation), dim=1)
        & torch.isfinite(xy_ratio)
        & torch.isfinite(z_ratio)
    )
    requirements_consistent = (~require_xy | xy_active) & (~require_z | z_active)
    required_progress = (
        (~require_xy | (xy_ratio >= minimum_progress_ratio))
        & (~require_z | (z_ratio >= minimum_progress_ratio))
    )
    return finite & requirements_consistent & required_progress, xy_ratio, z_ratio


def select_phase_primary_axis(
    active_xy: torch.Tensor,
    active_z: torch.Tensor,
    prefer_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose one meaningful-progress axis while keeping both axes active."""

    active_xy = torch.as_tensor(active_xy, dtype=torch.bool)
    active_z = torch.as_tensor(active_z, device=active_xy.device, dtype=torch.bool)
    prefer_z = torch.as_tensor(prefer_z, device=active_xy.device, dtype=torch.bool)
    primary_z = active_z & (prefer_z | ~active_xy)
    primary_xy = active_xy & ~primary_z
    return primary_xy, primary_z


def phase_primary_axis_progress_valid(
    predicted_translation: torch.Tensor,
    requested_translation: torch.Tensor,
    active_xy: torch.Tensor,
    active_z: torch.Tensor,
    prefer_z: torch.Tensor,
    minimum_primary_progress_ratio: float,
    minimum_companion_progress_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Require strong phase-primary progress and smaller forward companion progress."""

    if not 0.0 < minimum_companion_progress_ratio <= minimum_primary_progress_ratio <= 1.0:
        raise ValueError(
            "progress ratios must satisfy 0 < companion <= primary <= 1"
        )
    active_xy = torch.as_tensor(
        active_xy, device=predicted_translation.device, dtype=torch.bool
    ).reshape(predicted_translation.shape[0])
    active_z = torch.as_tensor(
        active_z, device=predicted_translation.device, dtype=torch.bool
    ).reshape(predicted_translation.shape[0])
    prefer_z = torch.as_tensor(
        prefer_z, device=predicted_translation.device, dtype=torch.bool
    ).reshape(predicted_translation.shape[0])
    primary_xy, primary_z = select_phase_primary_axis(active_xy, active_z, prefer_z)
    primary_present = primary_xy | primary_z
    primary_valid, xy_ratio, z_ratio = required_axis_progress_valid(
        predicted_translation,
        requested_translation,
        primary_xy,
        primary_z,
        minimum_primary_progress_ratio,
    )
    requested_xy_active = torch.sum(requested_translation[:, :2] ** 2, dim=1) > 1.0e-18
    requested_z_active = torch.abs(requested_translation[:, 2]) > 1.0e-9
    active_requests_valid = (~active_xy | requested_xy_active) & (
        ~active_z | requested_z_active
    )
    companion_xy = active_xy & ~primary_xy
    companion_z = active_z & ~primary_z
    companion_progress_valid = (
        (~companion_xy | (xy_ratio >= minimum_companion_progress_ratio))
        & (~companion_z | (z_ratio >= minimum_companion_progress_ratio))
    )
    return (
        primary_present
        & primary_valid
        & active_requests_valid
        & companion_progress_valid,
        xy_ratio,
        z_ratio,
    )


def phase_primary_translation_progress_ratio(
    predicted_translation: torch.Tensor,
    requested_translation: torch.Tensor,
    active_xy: torch.Tensor,
    active_z: torch.Tensor,
    prefer_z: torch.Tensor,
) -> torch.Tensor:
    """Return progress on the phase-primary axis, not the weaker companion axis."""

    active_xy = torch.as_tensor(
        active_xy, device=predicted_translation.device, dtype=torch.bool
    ).reshape(predicted_translation.shape[0])
    active_z = torch.as_tensor(
        active_z, device=predicted_translation.device, dtype=torch.bool
    ).reshape(predicted_translation.shape[0])
    prefer_z = torch.as_tensor(
        prefer_z, device=predicted_translation.device, dtype=torch.bool
    ).reshape(predicted_translation.shape[0])
    xy_ratio, z_ratio = translation_progress_ratios(
        predicted_translation,
        requested_translation,
        active_xy,
        active_z,
    )
    primary_xy, primary_z = select_phase_primary_axis(
        active_xy, active_z, prefer_z
    )
    ratio = torch.where(primary_z, z_ratio, xy_ratio)
    return torch.where(
        primary_xy | primary_z,
        ratio,
        torch.ones_like(ratio),
    )


def select_higher_phase_primary_candidate(
    current_valid: torch.Tensor,
    current_ratio: torch.Tensor,
    candidate_valid: torch.Tensor,
    candidate_ratio: torch.Tensor,
    phase_allows_candidate: torch.Tensor,
    minimum_improvement: float = 1.0e-6,
) -> torch.Tensor:
    """Select a safe phase-specific candidate only when it improves primary progress."""

    if minimum_improvement < 0.0:
        raise ValueError("minimum improvement must be non-negative")
    device = current_ratio.device
    current_valid = torch.as_tensor(current_valid, device=device, dtype=torch.bool)
    candidate_valid = torch.as_tensor(candidate_valid, device=device, dtype=torch.bool)
    phase_allows_candidate = torch.as_tensor(
        phase_allows_candidate, device=device, dtype=torch.bool
    )
    finite = torch.isfinite(current_ratio) & torch.isfinite(candidate_ratio)
    return (
        phase_allows_candidate
        & current_valid
        & candidate_valid
        & finite
        & (candidate_ratio > current_ratio + minimum_improvement)
    )


def advance_post_center_descent_progress(
    current_progress: torch.Tensor,
    xy_progress: torch.Tensor,
    ramp_start_progress: torch.Tensor,
    ramp_elapsed_steps: torch.Tensor,
    ramp_started: torch.Tensor,
    ramp_candidate: torch.Tensor,
    ramp_active: torch.Tensor,
    ramp_total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance a pausable, monotonic descent ramp without changing its start target."""

    if ramp_total_steps < 1:
        raise ValueError("post-center descent ramp must contain at least one step")
    ramp_started = torch.as_tensor(
        ramp_started, device=current_progress.device, dtype=torch.bool
    )
    ramp_candidate = torch.as_tensor(
        ramp_candidate, device=current_progress.device, dtype=torch.bool
    )
    ramp_active = torch.as_tensor(
        ramp_active, device=current_progress.device, dtype=torch.bool
    )
    pre_ramp_baseline = torch.maximum(current_progress, xy_progress)
    baseline = torch.where(
        ramp_started | ramp_candidate,
        current_progress,
        pre_ramp_baseline,
    ).clamp(0.0, 1.0)
    start_now = ramp_active & ~ramp_started
    updated_start = torch.where(start_now, baseline, ramp_start_progress).clamp(
        0.0, 1.0
    )
    updated_started = ramp_started | start_now
    advance = ramp_active & updated_started
    updated_elapsed = torch.where(
        advance,
        torch.clamp(ramp_elapsed_steps + 1, max=ramp_total_steps),
        ramp_elapsed_steps,
    )
    ramp_fraction = updated_elapsed.to(dtype=baseline.dtype) / ramp_total_steps
    timed_progress = updated_start + (1.0 - updated_start) * quintic_smoothstep(
        ramp_fraction
    )
    updated_progress = torch.where(
        updated_started,
        torch.maximum(baseline, timed_progress),
        baseline,
    )
    return updated_progress, updated_start, updated_elapsed, updated_started


def post_center_descent_readiness(
    descent_active: torch.Tensor,
    place_centered: torch.Tensor,
    distance_xy: torch.Tensor,
    object_xy_speed: torch.Tensor,
    target_tracking_error: torch.Tensor,
    maximum_xy_distance: float,
    maximum_xy_speed: float,
    maximum_tracking_error: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gate descent target advancement and expose mutually exclusive pause causes."""

    if maximum_xy_distance <= 0.0:
        raise ValueError("maximum XY distance must be positive")
    if maximum_xy_speed <= 0.0:
        raise ValueError("maximum XY speed must be positive")
    if maximum_tracking_error <= 0.0:
        raise ValueError("maximum tracking error must be positive")
    device = distance_xy.device
    descent_active = torch.as_tensor(
        descent_active, device=device, dtype=torch.bool
    )
    place_centered = torch.as_tensor(
        place_centered, device=device, dtype=torch.bool
    )
    candidate = (
        descent_active
        & torch.isfinite(distance_xy)
        & (distance_xy <= maximum_xy_distance)
    )
    centered_candidate = candidate & place_centered
    stable = torch.isfinite(object_xy_speed) & (
        object_xy_speed <= maximum_xy_speed
    )
    tracking = torch.isfinite(target_tracking_error) & (
        target_tracking_error <= maximum_tracking_error
    )
    stability_pause = centered_candidate & ~stable
    tracking_pause = centered_candidate & stable & ~tracking
    ready = centered_candidate & stable & tracking
    return candidate, ready, stability_pause, tracking_pause


def scale_delta_to_cartesian_bounds(
    joint_delta: torch.Tensor,
    predicted_translation: torch.Tensor,
    position_error: torch.Tensor,
    requested_translation: torch.Tensor,
    require_xy_progress: torch.Tensor,
    require_z_progress: torch.Tensor,
    maximum_xy: float | torch.Tensor,
    maximum_up: float | torch.Tensor,
    maximum_down: float | torch.Tensor,
    horizontal_deadband: float | torch.Tensor,
    vertical_deadband: float | torch.Tensor,
    inactive_leak_tolerance: float | torch.Tensor,
    minimum_progress_ratio: float = 1.0e-6,
    prefer_z_progress: torch.Tensor | None = None,
    minimum_companion_progress_ratio: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniformly shrink a joint delta to satisfy Cartesian safety bounds.

    ``predicted_translation`` is the Cartesian translation produced by
    ``joint_delta`` under the local Jacobian. Scaling both by the same value
    preserves the joint-space direction and any convex feasibility constraints
    already satisfied by the input delta.
    """

    batch_size = joint_delta.shape[0]
    phase_options_complete = (prefer_z_progress is None) == (
        minimum_companion_progress_ratio is None
    )
    if not phase_options_complete:
        raise ValueError(
            "prefer_z_progress and minimum_companion_progress_ratio must be provided together"
        )

    def as_batch(value: float | torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(
            value, device=joint_delta.device, dtype=joint_delta.dtype
        )
        if tensor.ndim == 0:
            tensor = tensor.expand(batch_size)
        return tensor.reshape(batch_size)

    xy_limit = as_batch(maximum_xy)
    up_limit = as_batch(maximum_up)
    down_limit = as_batch(maximum_down)
    xy_deadband = as_batch(horizontal_deadband)
    z_deadband = as_batch(vertical_deadband)
    leak_tolerance = as_batch(inactive_leak_tolerance)
    require_xy_progress = torch.as_tensor(
        require_xy_progress, device=joint_delta.device, dtype=torch.bool
    ).reshape(batch_size)
    require_z_progress = torch.as_tensor(
        require_z_progress, device=joint_delta.device, dtype=torch.bool
    ).reshape(batch_size)

    xy_error = position_error[:, :2]
    xy_step = predicted_translation[:, :2]
    xy_error_norm = torch.norm(xy_error, dim=1)
    xy_step_norm = torch.norm(xy_step, dim=1)
    xy_step_squared = torch.sum(xy_step**2, dim=1)
    xy_progress = torch.sum(xy_error * xy_step, dim=1)
    xy_active = xy_error_norm > xy_deadband

    z_error = position_error[:, 2]
    z_step = predicted_translation[:, 2]
    z_error_abs = torch.abs(z_error)
    z_step_abs = torch.abs(z_step)
    z_progress = z_error * z_step
    z_active = z_error_abs > z_deadband
    z_limit = torch.where(z_error >= 0.0, up_limit, down_limit)

    infinite = torch.full(
        (batch_size,),
        float("inf"),
        device=joint_delta.device,
        dtype=joint_delta.dtype,
    )
    xy_request_bound = torch.where(
        xy_step_norm > 0.0, xy_limit / xy_step_norm, infinite
    )
    xy_overshoot_bound = torch.where(
        xy_step_squared > 0.0,
        2.0 * xy_progress / xy_step_squared,
        infinite,
    )
    xy_leak_bound = torch.where(
        xy_step_norm > 0.0,
        0.999 * leak_tolerance / xy_step_norm,
        infinite,
    )
    xy_bound = torch.where(
        xy_active,
        torch.minimum(xy_request_bound, xy_overshoot_bound),
        xy_leak_bound,
    )

    z_request_bound = torch.where(z_step_abs > 0.0, z_limit / z_step_abs, infinite)
    z_overshoot_bound = torch.where(
        z_step_abs > 0.0,
        2.0 * z_progress / (z_step**2),
        infinite,
    )
    z_leak_bound = torch.where(
        z_step_abs > 0.0,
        0.999 * leak_tolerance / z_step_abs,
        infinite,
    )
    z_bound = torch.where(
        z_active,
        torch.minimum(z_request_bound, z_overshoot_bound),
        z_leak_bound,
    )

    scale = torch.clamp(torch.minimum(xy_bound, z_bound), min=0.0, max=1.0)
    scaled_prediction = predicted_translation * scale[:, None]
    if prefer_z_progress is None or minimum_companion_progress_ratio is None:
        progress_valid, _, _ = required_axis_progress_valid(
            scaled_prediction,
            requested_translation,
            require_xy_progress,
            require_z_progress,
            minimum_progress_ratio,
        )
    else:
        progress_valid, _, _ = phase_primary_axis_progress_valid(
            scaled_prediction,
            requested_translation,
            require_xy_progress,
            require_z_progress,
            prefer_z_progress,
            minimum_progress_ratio,
            minimum_companion_progress_ratio,
        )
    required_motion = require_xy_progress | require_z_progress
    finite = (
        torch.all(torch.isfinite(joint_delta), dim=1)
        & torch.all(torch.isfinite(predicted_translation), dim=1)
        & torch.all(torch.isfinite(position_error), dim=1)
        & torch.isfinite(xy_limit)
        & torch.isfinite(up_limit)
        & torch.isfinite(down_limit)
        & torch.isfinite(xy_deadband)
        & torch.isfinite(z_deadband)
        & torch.isfinite(leak_tolerance)
    )
    parameters_valid = (
        (xy_limit >= 0.0)
        & (up_limit >= 0.0)
        & (down_limit >= 0.0)
        & (xy_deadband >= 0.0)
        & (z_deadband >= 0.0)
        & (leak_tolerance >= 0.0)
    )
    forward = (~xy_active | (xy_step_norm == 0.0) | (xy_progress > 0.0)) & (
        ~z_active | (z_step_abs == 0.0) | (z_progress > 0.0)
    )
    nonzero_delta = torch.amax(torch.abs(joint_delta), dim=1) > 0.0
    nonzero_prediction = torch.amax(torch.abs(predicted_translation), dim=1) > 0.0
    feasible = (
        finite
        & parameters_valid
        & (xy_active | z_active)
        & required_motion
        & progress_valid
        & forward
        & nonzero_delta
        & nonzero_prediction
        & torch.isfinite(scale)
        & (scale > 0.0)
    )
    scale = torch.where(feasible, scale, torch.zeros_like(scale))
    scaled_delta = torch.where(
        feasible[:, None], joint_delta * scale[:, None], torch.zeros_like(joint_delta)
    )
    return scaled_delta, scale, feasible


def select_phase_axis_fallback(
    unresolved: torch.Tensor,
    xy_valid: torch.Tensor,
    z_valid: torch.Tensor,
    prefer_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefer the current phase axis, falling back to the other valid axis."""

    use_z = unresolved & z_valid & (prefer_z | ~xy_valid)
    use_xy = unresolved & xy_valid & ~use_z
    return use_xy, use_z


def constrain_target_lead(
    raw_lead: torch.Tensor,
    current_position: torch.Tensor,
    previous_target: torch.Tensor,
    joint_limits: torch.Tensor,
    maximum_lead: float,
    maximum_target_change: float,
) -> torch.Tensor:
    """Project a desired target into the joint, lead, and continuity intersection.

    Applying the continuity cap after the lead cap can move the final target back
    outside ``current_position +/- maximum_lead``.  Projecting the target into the
    common component-wise interval keeps every constraint true at the same time.
    When the interval is empty, the current pose is the safest explicit rebase;
    callers can detect the continuity break with
    :func:`target_lead_constraints_valid` and count a safety fallback.
    """

    desired_lead = scale_by_max_abs(raw_lead, maximum_lead)
    desired_target = current_position + desired_lead
    lower, upper, feasible = target_lead_constraint_intersection(
        current_position,
        previous_target,
        joint_limits,
        maximum_lead,
        maximum_target_change,
    )
    projected_target = torch.maximum(torch.minimum(desired_target, upper), lower)
    safe_rebase_target = torch.maximum(
        torch.minimum(current_position, joint_limits[:, :, 1]),
        joint_limits[:, :, 0],
    )
    projected_target = torch.where(
        feasible[:, None], projected_target, safe_rebase_target
    )
    return projected_target - current_position


def target_lead_constraint_intersection(
    current_position: torch.Tensor,
    previous_target: torch.Tensor,
    joint_limits: torch.Tensor,
    maximum_lead: float,
    maximum_target_change: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the common target interval and whether every joint is feasible."""

    lower = torch.maximum(
        joint_limits[:, :, 0],
        torch.maximum(
            current_position - maximum_lead,
            previous_target - maximum_target_change,
        ),
    )
    upper = torch.minimum(
        joint_limits[:, :, 1],
        torch.minimum(
            current_position + maximum_lead,
            previous_target + maximum_target_change,
        ),
    )
    feasible = torch.all(lower <= upper, dim=1)
    return lower, upper, feasible


def target_lead_constraints_valid(
    lead: torch.Tensor,
    current_position: torch.Tensor,
    previous_target: torch.Tensor,
    joint_limits: torch.Tensor,
    maximum_lead: float,
    maximum_target_change: float,
    tolerance: float = 1.0e-6,
) -> torch.Tensor:
    """Check finite, lead, continuity, and joint-limit constraints after finalization."""

    target = current_position + lead
    finite = torch.all(torch.isfinite(lead), dim=1) & torch.all(
        torch.isfinite(target), dim=1
    )
    lead_safe = torch.amax(torch.abs(lead), dim=1) <= maximum_lead + tolerance
    change_safe = (
        torch.amax(torch.abs(target - previous_target), dim=1)
        <= maximum_target_change + tolerance
    )
    limits_safe = torch.all(
        (target >= joint_limits[:, :, 0] - tolerance)
        & (target <= joint_limits[:, :, 1] + tolerance),
        dim=1,
    )
    return finite & lead_safe & change_safe & limits_safe


def project_goal_directed_step(
    candidate_step: torch.Tensor,
    position_error: torch.Tensor,
    maximum_xy: float | torch.Tensor,
    maximum_up: float | torch.Tensor,
    maximum_down: float | torch.Tensor,
    horizontal_deadband: float,
    vertical_deadband: float,
) -> torch.Tensor:
    """Project XY and Z independently onto bounded, non-overshooting goal motion."""

    projected = torch.zeros_like(candidate_step)
    xy_error = position_error[:, :2]
    xy_distance = torch.norm(xy_error, dim=1, keepdim=True)
    xy_direction = xy_error / torch.clamp(xy_distance, min=1.0e-9)
    xy_progress = torch.sum(candidate_step[:, :2] * xy_direction, dim=1, keepdim=True)
    xy_limit = torch.as_tensor(maximum_xy, device=candidate_step.device, dtype=candidate_step.dtype)
    if xy_limit.ndim == 0:
        xy_limit = xy_limit.expand(candidate_step.shape[0])
    xy_remaining = torch.clamp(xy_distance - horizontal_deadband, min=0.0)
    xy_limit = torch.minimum(xy_limit.reshape(-1, 1), xy_remaining)
    xy_progress = torch.minimum(torch.clamp(xy_progress, min=0.0), xy_limit)
    xy_active = xy_distance > horizontal_deadband
    projected[:, :2] = torch.where(xy_active, xy_direction * xy_progress, 0.0)

    z_error = position_error[:, 2]
    z_distance = torch.abs(z_error)
    z_direction = torch.sign(z_error)
    z_progress = candidate_step[:, 2] * z_direction
    maximum_up_tensor = torch.as_tensor(
        maximum_up, device=candidate_step.device, dtype=candidate_step.dtype
    )
    maximum_down_tensor = torch.as_tensor(
        maximum_down, device=candidate_step.device, dtype=candidate_step.dtype
    )
    if maximum_up_tensor.ndim == 0:
        maximum_up_tensor = maximum_up_tensor.expand(candidate_step.shape[0])
    if maximum_down_tensor.ndim == 0:
        maximum_down_tensor = maximum_down_tensor.expand(candidate_step.shape[0])
    z_limit = torch.where(z_error >= 0.0, maximum_up_tensor, maximum_down_tensor)
    z_remaining = torch.clamp(z_distance - vertical_deadband, min=0.0)
    z_limit = torch.minimum(z_limit, z_remaining)
    z_progress = torch.minimum(torch.clamp(z_progress, min=0.0), z_limit)
    projected[:, 2] = torch.where(
        z_distance > vertical_deadband, z_direction * z_progress, 0.0
    )
    return projected


def preserve_safe_goal_directed_step(
    candidate_step: torch.Tensor,
    position_error: torch.Tensor,
    maximum_xy: float | torch.Tensor,
    maximum_up: float | torch.Tensor,
    maximum_down: float | torch.Tensor,
    horizontal_deadband: float,
    vertical_deadband: float,
) -> torch.Tensor:
    """Keep safe incoming direction, projecting only axes that move away or overshoot."""

    projected = project_goal_directed_step(
        candidate_step,
        position_error,
        maximum_xy,
        maximum_up,
        maximum_down,
        horizontal_deadband,
        vertical_deadband,
    )
    result = projected.clone()
    xy_error = position_error[:, :2]
    xy_distance = torch.norm(xy_error, dim=1)
    xy_limit = torch.as_tensor(maximum_xy, device=candidate_step.device, dtype=candidate_step.dtype)
    if xy_limit.ndim == 0:
        xy_limit = xy_limit.expand(candidate_step.shape[0])
    candidate_xy_norm = torch.norm(candidate_step[:, :2], dim=1)
    candidate_xy_progress = torch.sum(candidate_step[:, :2] * xy_error, dim=1)
    candidate_xy_remaining = torch.norm(xy_error - candidate_step[:, :2], dim=1)
    safe_xy = (
        (xy_distance > horizontal_deadband)
        & (candidate_xy_progress >= 0.0)
        & (candidate_xy_remaining <= xy_distance)
        & (candidate_xy_remaining >= horizontal_deadband)
        & (candidate_xy_norm <= xy_limit)
    )
    result[:, :2] = torch.where(
        safe_xy[:, None], candidate_step[:, :2], projected[:, :2]
    )

    z_error = position_error[:, 2]
    up_limit = torch.as_tensor(maximum_up, device=candidate_step.device, dtype=candidate_step.dtype)
    down_limit = torch.as_tensor(
        maximum_down, device=candidate_step.device, dtype=candidate_step.dtype
    )
    if up_limit.ndim == 0:
        up_limit = up_limit.expand(candidate_step.shape[0])
    if down_limit.ndim == 0:
        down_limit = down_limit.expand(candidate_step.shape[0])
    z_limit = torch.where(z_error >= 0.0, up_limit, down_limit)
    safe_z = (
        (torch.abs(z_error) > vertical_deadband)
        & (candidate_step[:, 2] * z_error >= 0.0)
        & (torch.abs(z_error - candidate_step[:, 2]) <= torch.abs(z_error))
        & (torch.abs(z_error - candidate_step[:, 2]) >= vertical_deadband)
        & (torch.abs(candidate_step[:, 2]) <= z_limit)
    )
    result[:, 2] = torch.where(safe_z, candidate_step[:, 2], projected[:, 2])
    return result


def apply_stopping_speed_envelope(
    velocity: torch.Tensor,
    position_error: torch.Tensor,
    horizontal_deadband: float,
    vertical_deadband: float,
    maximum_horizontal_acceleration: float,
    maximum_vertical_acceleration: float,
    maximum_horizontal_jerk: float,
    maximum_vertical_jerk: float,
) -> torch.Tensor:
    """Bound speed for a conservative jerk ramp followed by bounded deceleration."""

    def stopping_speed(
        remaining: torch.Tensor,
        maximum_acceleration: float,
        maximum_jerk: float,
    ) -> torch.Tensor:
        # d >= v*a/j + v^2/(2a). This reserves one full jerk-ramp interval
        # before constant deceleration and therefore starts braking before a
        # constant-acceleration envelope would.
        ramp_velocity = maximum_acceleration**2 / maximum_jerk
        return -ramp_velocity + torch.sqrt(
            ramp_velocity**2 + 2.0 * maximum_acceleration * remaining
        )

    bounded = velocity.clone()
    xy_remaining = torch.clamp(
        torch.norm(position_error[:, :2], dim=1) - horizontal_deadband,
        min=0.0,
    )
    xy_stopping_speed = stopping_speed(
        xy_remaining,
        maximum_horizontal_acceleration,
        maximum_horizontal_jerk,
    )
    bounded[:, :2] = limit_vector_norm(bounded[:, :2], xy_stopping_speed)
    z_remaining = torch.clamp(
        torch.abs(position_error[:, 2]) - vertical_deadband, min=0.0
    )
    z_stopping_speed = stopping_speed(
        z_remaining,
        maximum_vertical_acceleration,
        maximum_vertical_jerk,
    )
    bounded[:, 2] = torch.clamp(
        bounded[:, 2], min=-z_stopping_speed, max=z_stopping_speed
    )
    return bounded


def damped_pseudoinverse(
    task_jacobian: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    """Return a batched damped least-squares pseudoinverse."""

    jacobian_t = torch.transpose(task_jacobian, 1, 2)
    identity = torch.eye(
        task_jacobian.shape[1],
        device=task_jacobian.device,
        dtype=task_jacobian.dtype,
    ).unsqueeze(0)
    damped = (
        task_jacobian @ jacobian_t
        + damping**2 * identity.expand(task_jacobian.shape[0], -1, -1)
    )
    damped_inverse = torch.linalg.solve(
        damped, identity.expand(task_jacobian.shape[0], -1, -1)
    )
    return jacobian_t @ damped_inverse


def build_prioritized_dls_solver(
    task_jacobian: torch.Tensor,
    primary_row_mask: torch.Tensor,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a translation-first DLS solver with secondary rows in its nullspace."""

    primary_mask = primary_row_mask.to(dtype=task_jacobian.dtype)
    secondary_mask = 1.0 - primary_mask
    primary_jacobian = task_jacobian * primary_mask.unsqueeze(-1)
    primary_pinv = damped_pseudoinverse(primary_jacobian, damping)
    identity = torch.eye(
        task_jacobian.shape[2],
        device=task_jacobian.device,
        dtype=task_jacobian.dtype,
    ).unsqueeze(0)
    primary_nullspace = identity - primary_pinv @ primary_jacobian
    secondary_jacobian = task_jacobian * secondary_mask.unsqueeze(-1)
    secondary_effective_jacobian = secondary_jacobian @ primary_nullspace
    secondary_pinv = damped_pseudoinverse(secondary_effective_jacobian, damping)
    return primary_pinv, primary_nullspace, secondary_pinv, secondary_mask


def solve_primary_dls_delta(
    target_twist: torch.Tensor,
    task_jacobian: torch.Tensor,
    primary_row_mask: torch.Tensor,
    primary_pinv: torch.Tensor,
    maximum_joint_step: float,
    iterations: int = 2,
) -> torch.Tensor:
    """Solve only the active translation rows without secondary-axis tradeoffs."""

    primary_mask = primary_row_mask.to(dtype=target_twist.dtype)
    primary_target = target_twist * primary_mask
    delta = (primary_pinv @ primary_target.unsqueeze(-1)).squeeze(-1)
    for _ in range(iterations):
        residual = (
            target_twist
            - (task_jacobian @ delta.unsqueeze(-1)).squeeze(-1)
        ) * primary_mask
        delta = delta + (primary_pinv @ residual.unsqueeze(-1)).squeeze(-1)
    return scale_by_max_abs(delta, maximum_joint_step)


def solve_prioritized_dls_delta(
    target_twist: torch.Tensor,
    task_jacobian: torch.Tensor,
    primary_row_mask: torch.Tensor,
    solver: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    maximum_joint_step: float,
) -> torch.Tensor:
    """Solve active translation first, then reduce orientation/inactive leak in nullspace."""

    primary_pinv, primary_nullspace, secondary_pinv, secondary_mask = solver
    primary_delta = solve_primary_dls_delta(
        target_twist,
        task_jacobian,
        primary_row_mask,
        primary_pinv,
        maximum_joint_step,
    )
    secondary_residual = (
        target_twist
        - (task_jacobian @ primary_delta.unsqueeze(-1)).squeeze(-1)
    ) * secondary_mask
    secondary_seed = (secondary_pinv @ secondary_residual.unsqueeze(-1)).squeeze(-1)
    secondary_delta = (primary_nullspace @ secondary_seed.unsqueeze(-1)).squeeze(-1)
    combined = scale_by_max_abs(primary_delta + secondary_delta, maximum_joint_step)

    # Restore any primary progress lost to damping in the approximate nullspace.
    primary_mask = primary_row_mask.to(dtype=target_twist.dtype)
    primary_residual = (
        target_twist
        - (task_jacobian @ combined.unsqueeze(-1)).squeeze(-1)
    ) * primary_mask
    combined = combined + (primary_pinv @ primary_residual.unsqueeze(-1)).squeeze(-1)
    return scale_by_max_abs(combined, maximum_joint_step)


def active_translation_progress_ratio(
    predicted_translation: torch.Tensor,
    requested_translation: torch.Tensor,
    active_xy: torch.Tensor,
    active_z: torch.Tensor,
) -> torch.Tensor:
    """Return the weakest requested-axis progress ratio for each environment."""

    xy_ratio, z_ratio = translation_progress_ratios(
        predicted_translation,
        requested_translation,
        active_xy,
        active_z,
    )
    ratio = torch.where(
        active_xy & active_z,
        torch.minimum(xy_ratio, z_ratio),
        torch.where(active_xy, xy_ratio, z_ratio),
    )
    return torch.where(active_xy | active_z, ratio, torch.ones_like(ratio))


def translation_progress_ratios(
    predicted_translation: torch.Tensor,
    requested_translation: torch.Tensor,
    active_xy: torch.Tensor,
    active_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return independent XY and Z progress ratios, using one for inactive axes."""

    requested_xy_squared = torch.sum(requested_translation[:, :2] ** 2, dim=1)
    xy_ratio = torch.sum(
        predicted_translation[:, :2] * requested_translation[:, :2], dim=1
    ) / torch.clamp(requested_xy_squared, min=1.0e-18)
    z_ratio = (
        predicted_translation[:, 2]
        * requested_translation[:, 2]
        / torch.clamp(requested_translation[:, 2] ** 2, min=1.0e-18)
    )
    infinite = torch.full_like(xy_ratio, float("inf"))
    xy_ratio = torch.where(active_xy, xy_ratio, infinite)
    z_ratio = torch.where(active_z, z_ratio, infinite)
    ones = torch.ones_like(xy_ratio)
    return torch.where(active_xy, xy_ratio, ones), torch.where(active_z, z_ratio, ones)


def realized_translation_progress(
    actual_translation: torch.Tensor,
    requested_translation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attribute measured translation to the direction requested one control earlier."""

    requested_xy = requested_translation[:, :2]
    requested_xy_norm = torch.norm(requested_xy, dim=1)
    requested_xy_direction = requested_xy / torch.clamp(
        requested_xy_norm[:, None], min=1.0e-9
    )
    realized_xy_progress = torch.sum(
        actual_translation[:, :2] * requested_xy_direction, dim=1
    )
    requested_z = torch.abs(requested_translation[:, 2])
    realized_z_progress = (
        torch.sign(requested_translation[:, 2]) * actual_translation[:, 2]
    )
    return (
        requested_xy_norm,
        requested_z,
        realized_xy_progress,
        realized_z_progress,
    )


def jerk_limited_cartesian_velocity(
    previous_velocity: torch.Tensor,
    previous_acceleration: torch.Tensor,
    desired_velocity: torch.Tensor,
    step_dt: float,
    maximum_horizontal_acceleration: float,
    maximum_vertical_acceleration: float,
    maximum_horizontal_jerk: float,
    maximum_vertical_jerk: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance a Cartesian command with independent XY and Z acceleration/jerk limits."""

    desired_acceleration = (desired_velocity - previous_velocity) / step_dt
    desired_acceleration[:, :2] = limit_vector_norm(
        desired_acceleration[:, :2], maximum_horizontal_acceleration
    )
    desired_acceleration[:, 2] = torch.clamp(
        desired_acceleration[:, 2],
        min=-maximum_vertical_acceleration,
        max=maximum_vertical_acceleration,
    )

    acceleration_change = desired_acceleration - previous_acceleration
    acceleration_change[:, :2] = limit_vector_norm(
        acceleration_change[:, :2], maximum_horizontal_jerk * step_dt
    )
    acceleration_change[:, 2] = torch.clamp(
        acceleration_change[:, 2],
        min=-maximum_vertical_jerk * step_dt,
        max=maximum_vertical_jerk * step_dt,
    )
    acceleration = previous_acceleration + acceleration_change
    acceleration[:, :2] = limit_vector_norm(
        acceleration[:, :2], maximum_horizontal_acceleration
    )
    acceleration[:, 2] = torch.clamp(
        acceleration[:, 2],
        min=-maximum_vertical_acceleration,
        max=maximum_vertical_acceleration,
    )
    velocity = previous_velocity + acceleration * step_dt
    return velocity, acceleration
