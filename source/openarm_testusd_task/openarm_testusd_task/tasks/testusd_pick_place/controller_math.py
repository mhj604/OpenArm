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
