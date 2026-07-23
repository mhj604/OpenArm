#!/usr/bin/env python3
"""Pure geometry helpers shared by the ROMO-B builder and validators."""

from __future__ import annotations

import math
from collections.abc import Sequence


def positive_body_half_width_at_x(
    outline_xy: Sequence[Sequence[float] | tuple[float, float]],
    x: float,
) -> float:
    """Interpolate the positive-Y exterior of a symmetric plan outline."""

    positive = sorted(
        (float(point[0]), float(point[1]))
        for point in outline_xy
        if float(point[1]) > 0.0
    )
    if len(positive) < 2:
        raise ValueError("The plan outline needs at least two positive-Y points")
    if x < positive[0][0] - 1e-12 or x > positive[-1][0] + 1e-12:
        raise ValueError(f"X={x} is outside the plan outline")
    for (x0, y0), (x1, y1) in zip(positive, positive[1:]):
        if x <= x1 + 1e-12:
            if math.isclose(x0, x1, rel_tol=0.0, abs_tol=1e-15):
                return max(y0, y1)
            fraction = (x - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)
    return positive[-1][1]


def circular_arch_top_z(
    *,
    x: float,
    centre_x: float,
    centre_z: float,
    radius: float,
) -> float | None:
    """Return the upper half of one circular wheel-house opening."""

    if radius <= 0.0:
        raise ValueError("A wheel-house radius must be positive")
    horizontal = float(x) - float(centre_x)
    if abs(horizontal) > radius + 1e-12:
        return None
    if math.isclose(
        abs(horizontal),
        radius,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return float(centre_z)
    return float(centre_z) + math.sqrt(
        max(0.0, radius * radius - horizontal * horizontal)
    )


def single_circular_wheel_house_profile(
    *,
    body_x_values: Sequence[float],
    wheel_centres_xz: Sequence[Sequence[float] | tuple[float, float]],
    opening_radius: float,
    body_bottom_z: float,
    arc_segments: int,
) -> list[tuple[float, float]]:
    """Build one circular, bottom-open arch for each wheel.

    The duplicate X samples at each circle's horizontal tangent form the
    vertical lower legs visible in the manual side elevation. No steering
    sweep is applied to this external profile.
    """

    if len(body_x_values) < 2:
        raise ValueError("The body profile needs at least two X values")
    if opening_radius <= 0.0:
        raise ValueError("A wheel-house radius must be positive")
    if arc_segments < 8:
        raise ValueError("A circular wheel house needs at least eight segments")

    minimum_x = min(float(value) for value in body_x_values)
    maximum_x = max(float(value) for value in body_x_values)
    centres = sorted(
        (float(centre[0]), float(centre[1]))
        for centre in wheel_centres_xz
    )
    if not centres:
        raise ValueError("At least one wheel centre is required")

    intervals: list[tuple[float, float]] = []
    sample_x_values = [float(value) for value in body_x_values]
    for centre_x, _ in centres:
        left = centre_x - opening_radius
        right = centre_x + opening_radius
        if left <= minimum_x or right >= maximum_x:
            raise ValueError("A wheel-house arch must remain inside the body")
        intervals.append((left, right))
        sample_x_values.extend((left, right))
        for index in range(arc_segments + 1):
            angle = math.pi - math.pi * index / arc_segments
            sample_x_values.append(centre_x + opening_radius * math.cos(angle))

    for (_, previous_right), (following_left, _) in zip(
        intervals,
        intervals[1:],
    ):
        if previous_right >= following_left:
            raise ValueError("Wheel-house arches must not overlap")

    ordered_x: list[float] = []
    for x in sorted(sample_x_values):
        if not ordered_x or not math.isclose(
            x,
            ordered_x[-1],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            ordered_x.append(x)

    profile: list[tuple[float, float]] = []
    for x in ordered_x:
        active_tops = [
            top
            for centre_x, centre_z in centres
            if (
                top := circular_arch_top_z(
                    x=x,
                    centre_x=centre_x,
                    centre_z=centre_z,
                    radius=opening_radius,
                )
            )
            is not None
        ]
        top_z = max([float(body_bottom_z), *active_tops])
        left_boundary = any(
            math.isclose(
                x,
                centre_x - opening_radius,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for centre_x, _ in centres
        )
        right_boundary = any(
            math.isclose(
                x,
                centre_x + opening_radius,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for centre_x, _ in centres
        )
        if left_boundary:
            profile.append((x, float(body_bottom_z)))
        profile.append((x, top_z))
        if right_boundary:
            profile.append((x, float(body_bottom_z)))
    return profile


def steered_cylinder_section_top_z(
    *,
    x: float,
    side_plane_y: float,
    steering_pivot_xyz: Sequence[float],
    neutral_wheel_xyz: Sequence[float],
    cylinder_radius: float,
    cylinder_half_width: float,
    steering_limit_deg: float,
    steering_sample_count: int,
) -> float | None:
    """Return the highest steered-wheel section at an exterior side surface.

    The wheel is a finite cylinder whose neutral axis runs from the steering
    pivot toward the wheel centre. The centre and axis rotate together about
    the pivot's +Z axis. Sampling covers the complete configured steering
    interval, including both endpoints and neutral.
    """

    if cylinder_radius <= 0.0 or cylinder_half_width <= 0.0:
        raise ValueError("Cylinder radius and half width must be positive")
    if steering_limit_deg < 0.0:
        raise ValueError("Steering limit must be non-negative")
    if steering_sample_count < 3 or steering_sample_count % 2 == 0:
        raise ValueError("Steering sample count must be an odd integer >= 3")

    pivot_x, pivot_y, _ = (float(value) for value in steering_pivot_xyz)
    neutral_x, neutral_y, wheel_z = (
        float(value) for value in neutral_wheel_xyz
    )
    offset_x = neutral_x - pivot_x
    offset_y = neutral_y - pivot_y
    offset_length = math.hypot(offset_x, offset_y)
    if offset_length <= 0.0:
        raise ValueError("Wheel centre must be offset from the steering pivot")
    neutral_axis_x = offset_x / offset_length
    neutral_axis_y = offset_y / offset_length

    highest: float | None = None
    for index in range(steering_sample_count):
        fraction = index / (steering_sample_count - 1)
        angle_deg = -steering_limit_deg + 2.0 * steering_limit_deg * fraction
        angle = math.radians(angle_deg)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        axis_x = neutral_axis_x * cosine - neutral_axis_y * sine
        axis_y = neutral_axis_x * sine + neutral_axis_y * cosine
        centre_x = pivot_x + offset_length * axis_x
        centre_y = pivot_y + offset_length * axis_y

        delta_x = x - centre_x
        delta_y = side_plane_y - centre_y
        axial = delta_x * axis_x + delta_y * axis_y
        if abs(axial) > cylinder_half_width:
            continue
        radial_horizontal = delta_x * axis_y - delta_y * axis_x
        if abs(radial_horizontal) > cylinder_radius:
            continue
        radial_vertical = math.sqrt(
            max(
                0.0,
                cylinder_radius * cylinder_radius
                - radial_horizontal * radial_horizontal,
            )
        )
        top_z = wheel_z + radial_vertical
        highest = top_z if highest is None else max(highest, top_z)
    return highest
