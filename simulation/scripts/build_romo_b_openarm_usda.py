#!/usr/bin/env python3
"""Build the reviewable ROMO-B + OpenArm Isaac Sim USD from the audited JSON spec.

Run this with an Isaac Sim 5.1 Python environment so that ``pxr`` is available.
The OpenArm payload is intentionally left unloaded while authoring; the generated
stage therefore does not need network access.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from romo_b_geometry import (
    positive_body_half_width_at_x,
    single_circular_wheel_house_profile,
)


SIMULATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = SIMULATION_ROOT / "config" / "romo_b_openarm_spec.json"
DEFAULT_OUTPUT = SIMULATION_ROOT / "assets" / "romo_b_openarm_manual_proxy.usda"


def value(node: dict[str, Any]) -> float:
    """Return the numeric value of a provenance-bearing spec node."""

    return float(node["value"])


def set_translate(prim: Usd.Prim, xyz: tuple[float, float, float] | list[float]) -> None:
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*xyz))


def set_scale(prim: Usd.Prim, xyz: tuple[float, float, float]) -> None:
    UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(*xyz))


def set_orient_wxyz(prim: Usd.Prim, wxyz: tuple[float, float, float, float] | list[float]) -> None:
    UsdGeom.Xformable(prim).AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Quatf(wxyz[0], wxyz[1], wxyz[2], wxyz[3])
    )


def custom_attr(prim: Usd.Prim, name: str, type_name: Sdf.ValueTypeName, attr_value: Any) -> None:
    prim.CreateAttribute(name, type_name, custom=True).Set(attr_value)


def add_mass(prim: Usd.Prim, mass_kg: float) -> None:
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass_kg)


def add_box(
    stage: Usd.Stage,
    path: str,
    dimensions: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = False,
) -> UsdGeom.Cube:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    set_scale(cube.GetPrim(), dimensions)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    return cube


def add_translated_box(
    stage: Usd.Stage,
    path: str,
    dimensions: tuple[float, float, float],
    centre: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = False,
) -> UsdGeom.Cube:
    """Create a unit cube with an unscaled translation followed by scale."""

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    set_translate(cube.GetPrim(), centre)
    set_scale(cube.GetPrim(), dimensions)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    return cube


def add_cylinder(
    stage: Usd.Stage,
    path: str,
    axis: str,
    radius: float,
    height: float,
    centre: tuple[float, float, float],
    color: tuple[float, float, float],
) -> UsdGeom.Cylinder:
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(axis)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    set_translate(cylinder.GetPrim(), centre)
    cylinder.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return cylinder


def add_extruded_polygon_mesh(
    stage: Usd.Stage,
    path: str,
    xy_points: list[tuple[float, float]],
    z_min: float,
    z_max: float,
    color: tuple[float, float, float],
    skip_side_edges: set[int] | None = None,
) -> UsdGeom.Mesh:
    """Extrude a counter-clockwise XY polygon into a deterministic visual mesh."""

    if len(xy_points) < 3:
        raise ValueError("An extruded polygon needs at least three XY points")
    count = len(xy_points)
    points = [
        Gf.Vec3f(float(x), float(y), float(z))
        for z in (z_min, z_max)
        for x, y in xy_points
    ]
    skipped = skip_side_edges or set()
    if any(index < 0 or index >= count for index in skipped):
        raise ValueError("Skipped polygon edge index is outside the contour")
    face_counts = [count, count] + [4] * (count - len(skipped))
    face_indices = list(reversed(range(count))) + list(range(count, 2 * count))
    for index in range(count):
        if index in skipped:
            continue
        following = (index + 1) % count
        face_indices.extend(
            [index, following, count + following, count + index]
        )

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return mesh


def add_yz_panel_mesh(
    stage: Usd.Stage,
    path: str,
    x: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    color: tuple[float, float, float],
) -> UsdGeom.Mesh:
    """Create a double-sided end-panel quad in a constant-X plane."""

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(x, y_min, z_min),
            Gf.Vec3f(x, y_max, z_min),
            Gf.Vec3f(x, y_max, z_max),
            Gf.Vec3f(x, y_min, z_max),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return mesh


def add_xy_panel_mesh(
    stage: Usd.Stage,
    path: str,
    xy_points: list[tuple[float, float]],
    z: float,
    color: tuple[float, float, float],
) -> UsdGeom.Mesh:
    """Create a double-sided horizontal panel from a plan contour."""

    if len(xy_points) < 3:
        raise ValueError("An XY panel needs at least three points")
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [Gf.Vec3f(float(x), float(y), float(z)) for x, y in xy_points]
    )
    mesh.CreateFaceVertexCountsAttr([len(xy_points)])
    mesh.CreateFaceVertexIndicesAttr(list(range(len(xy_points))))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return mesh


def add_profiled_side_panel_mesh(
    stage: Usd.Stage,
    path: str,
    x_values: list[float],
    y_values: list[float],
    lower_z_values: list[float],
    upper_z: float,
    color: tuple[float, float, float],
) -> UsdGeom.Mesh:
    """Create a side panel whose lower edge follows open wheel wells."""

    count = len(x_values)
    if count < 2 or not (
        len(y_values) == count and len(lower_z_values) == count
    ):
        raise ValueError("A profiled side panel needs matching sampled arrays")
    if any(lower_z > upper_z for lower_z in lower_z_values):
        raise ValueError("A side-panel lower edge cannot exceed its upper edge")

    points: list[Gf.Vec3f] = []
    for x, y, lower_z in zip(x_values, y_values, lower_z_values):
        points.extend(
            (
                Gf.Vec3f(float(x), float(y), float(lower_z)),
                Gf.Vec3f(float(x), float(y), float(upper_z)),
            )
        )
    face_counts: list[int] = []
    face_indices: list[int] = []
    for index in range(count - 1):
        lower = 2 * index
        following = lower + 2
        if math.isclose(
            x_values[index],
            x_values[index + 1],
            rel_tol=0.0,
            abs_tol=1e-12,
        ) and math.isclose(
            y_values[index],
            y_values[index + 1],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            continue
        face_counts.append(4)
        face_indices.extend((lower, following, following + 1, lower + 1))

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return mesh


def add_extruded_polygon_ring_mesh(
    stage: Usd.Stage,
    path: str,
    outer_xy: list[tuple[float, float]],
    inner_xy: list[tuple[float, float]],
    z_min: float,
    z_max: float,
    color: tuple[float, float, float],
) -> UsdGeom.Mesh:
    """Extrude matching outer/inner polygons while keeping the centre open."""

    if len(outer_xy) < 3 or len(outer_xy) != len(inner_xy):
        raise ValueError("A polygon ring needs matching outer and inner contours")
    count = len(outer_xy)
    points: list[Gf.Vec3f] = []
    for z in (z_min, z_max):
        for contour in (outer_xy, inner_xy):
            points.extend(
                Gf.Vec3f(float(x), float(y), float(z))
                for x, y in contour
            )

    def vertex(z_index: int, contour_index: int, point_index: int) -> int:
        return z_index * 2 * count + contour_index * count + point_index

    face_counts: list[int] = []
    face_indices: list[int] = []

    def quad(indices: tuple[int, int, int, int]) -> None:
        face_counts.append(4)
        face_indices.extend(indices)

    for index in range(count):
        following = (index + 1) % count
        # Recessed top/bottom annulus.
        quad(
            (
                vertex(1, 0, index),
                vertex(1, 0, following),
                vertex(1, 1, following),
                vertex(1, 1, index),
            )
        )
        quad(
            (
                vertex(0, 1, index),
                vertex(0, 1, following),
                vertex(0, 0, following),
                vertex(0, 0, index),
            )
        )
        # Outer and inner vertical walls.
        quad(
            (
                vertex(0, 0, index),
                vertex(0, 0, following),
                vertex(1, 0, following),
                vertex(1, 0, index),
            )
        )
        quad(
            (
                vertex(0, 1, following),
                vertex(0, 1, index),
                vertex(1, 1, index),
                vertex(1, 1, following),
            )
        )

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return mesh


def add_revolute_joint(
    stage: Usd.Stage,
    path: str,
    body0: str,
    body1: str,
    axis: str,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float],
    lower_deg: float | None = None,
    upper_deg: float | None = None,
) -> UsdPhysics.RevoluteJoint:
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateAxisAttr(axis)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*local_pos1))
    if lower_deg is not None:
        joint.CreateLowerLimitAttr(lower_deg)
    if upper_deg is not None:
        joint.CreateUpperLimitAttr(upper_deg)
    return joint


def build(spec_path: Path, output_path: Path) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    dims = spec["romo_b"]["dimensions_m"]
    masses = spec["simulation_proxy"]["mass_allocation_kg"]
    limits = spec["romo_b"]["limits"]
    proxy = spec["simulation_proxy"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=output_path.suffix, dir=output_path.parent
    )
    os.close(handle)
    os.unlink(temporary_name)

    def remove_incomplete_output() -> None:
        Path(temporary_name).unlink(missing_ok=True)

    atexit.register(remove_incomplete_output)

    stage = Usd.Stage.CreateNew(temporary_name)
    if stage is None:
        raise RuntimeError(f"Could not create USD stage: {temporary_name}")
    stage.SetLoadRules(Usd.StageLoadRules.LoadNone())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetTimeCodesPerSecond(60.0)
    stage.SetFramesPerSecond(60.0)
    stage.SetMetadata("comment", "Generated from simulation/config/romo_b_openarm_spec.json; do not hand-edit.")

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)

    ground = add_box(
        stage,
        "/World/Ground",
        dimensions=(10.0, 10.0, 0.02),
        color=(0.18, 0.18, 0.18),
        collision=True,
    )
    set_translate(ground.GetPrim(), (0.0, 0.0, -0.01))
    ground_ops = UsdGeom.Xformable(ground.GetPrim()).GetOrderedXformOps()
    UsdGeom.Xformable(ground.GetPrim()).SetXformOpOrder([ground_ops[1], ground_ops[0]])
    custom_attr(ground.GetPrim(), "romoB:role", Sdf.ValueTypeNames.String, "static_ground")

    robot = UsdGeom.Xform.Define(stage, "/World/ROMOBOpenArm")
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())
    solver = spec["simulation_proxy"]["articulation_solver"]
    collision_policy = spec["simulation_proxy"]["collision_policy"]
    physx_articulation = PhysxSchema.PhysxArticulationAPI.Apply(robot.GetPrim())
    physx_articulation.CreateEnabledSelfCollisionsAttr(bool(solver["enabled_self_collisions"]))
    physx_articulation.CreateSolverPositionIterationCountAttr(int(solver["position_iteration_count"]))
    physx_articulation.CreateSolverVelocityIterationCountAttr(int(solver["velocity_iteration_count"]))
    robot.GetPrim().SetCustomDataByKey("modelScope", spec["model_scope"])
    custom_attr(robot.GetPrim(), "romoB:coordinateFrame", Sdf.ValueTypeNames.String, "+X forward, +Y left, +Z up")
    custom_attr(robot.GetPrim(), "romoB:sourceSpec", Sdf.ValueTypeNames.Asset, Sdf.AssetPath("../config/romo_b_openarm_spec.json"))
    custom_attr(
        robot.GetPrim(),
        "romoB:allAuthoredCollidersEnabled",
        Sdf.ValueTypeNames.Bool,
        bool(collision_policy["all_authored_colliders_enabled"]),
    )
    custom_attr(
        robot.GetPrim(),
        "romoB:filteredPairsAllowed",
        Sdf.ValueTypeNames.Bool,
        bool(collision_policy["allow_filtered_pairs"]),
    )
    custom_attr(
        robot.GetPrim(),
        "romoB:allActiveRigidBodiesEnabled",
        Sdf.ValueTypeNames.Bool,
        bool(collision_policy["all_active_rigid_bodies_enabled"]),
    )
    custom_attr(
        robot.GetPrim(),
        "romoB:collisionRuntimeLimitation",
        Sdf.ValueTypeNames.String,
        collision_policy["runtime_limitation"],
    )
    custom_attr(
        robot.GetPrim(),
        "romoB:manualSteeringCommandToUsdMultiplier",
        Sdf.ValueTypeNames.Double,
        float(spec["romo_b"]["steering_command_convention"]["manual_to_usd_joint_multiplier"]),
    )
    custom_attr(
        robot.GetPrim(),
        "romoB:platformLinearSpeedLimit",
        Sdf.ValueTypeNames.Double,
        value(limits["platform_linear_speed_m_s"]),
    )
    custom_attr(
        robot.GetPrim(),
        "romoB:allowedPayload",
        Sdf.ValueTypeNames.Double,
        value(spec["romo_b"]["mass_kg"]["allowed_payload"]),
    )

    length = value(dims["overall_length"])
    overall_width = value(dims["overall_width"])
    height = value(dims["overall_height"])
    clearance = value(dims["ground_clearance"])
    chassis_height = height - clearance
    chassis_center_z = clearance + chassis_height / 2.0
    chassis_path = "/World/ROMOBOpenArm/Chassis"
    chassis = UsdGeom.Xform.Define(stage, chassis_path)
    set_translate(chassis.GetPrim(), (0.0, 0.0, chassis_center_z))
    add_mass(chassis.GetPrim(), float(masses["chassis"]))
    custom_attr(chassis.GetPrim(), "romoB:massAllocationQuality", Sdf.ValueTypeNames.String, masses["quality"])
    custom_attr(chassis.GetPrim(), "romoB:manualOverallLength", Sdf.ValueTypeNames.Double, length)
    custom_attr(chassis.GetPrim(), "romoB:manualOverallWidth", Sdf.ValueTypeNames.Double, overall_width)
    custom_attr(chassis.GetPrim(), "romoB:manualOverallHeight", Sdf.ValueTypeNames.Double, height)
    custom_attr(chassis.GetPrim(), "romoB:manualGroundClearance", Sdf.ValueTypeNames.Double, clearance)

    manual_geometry = proxy["manual_external_geometry"]
    body_width = float(proxy["chassis_visual_width_m"]["value"])
    outline_spec = manual_geometry["body_outline_normalized_xy"]
    outline_scale = tuple(float(item) for item in outline_spec["axis_scale_m"])
    outline = [
        (float(point[0]) * outline_scale[0], float(point[1]) * outline_scale[1])
        for point in outline_spec["value"]
    ]
    sections = manual_geometry["body_sections_world_z_m"]
    side_shell_z = tuple(float(item) - chassis_center_z for item in sections["side_shell"])
    rim_z = tuple(float(item) - chassis_center_z for item in sections["top_rim"])
    plate_z = tuple(
        float(item) - chassis_center_z for item in sections["recessed_top_plate"]
    )
    aluminium = (0.66, 0.69, 0.72)
    romo_blue = (0.02, 0.36, 0.72)
    dark_detail = (0.055, 0.065, 0.075)
    visual_root = f"{chassis_path}/Visuals"
    UsdGeom.Scope.Define(stage, visual_root)
    side_shell_top = add_xy_panel_mesh(
        stage,
        f"{visual_root}/SideShellTopPlate",
        outline,
        side_shell_z[1],
        aluminium,
    )
    custom_attr(
        side_shell_top.GetPrim(),
        "romoB:geometryQuality",
        Sdf.ValueTypeNames.String,
        manual_geometry["quality"],
    )
    plate_scale = tuple(float(item) for item in manual_geometry["top_plate_scale_xy"]["value"])
    plate_outline = [
        (point[0] * plate_scale[0], point[1] * plate_scale[1])
        for point in outline
    ]
    add_extruded_polygon_ring_mesh(
        stage,
        f"{visual_root}/TopRim",
        outline,
        plate_outline,
        rim_z[0],
        rim_z[1],
        romo_blue,
    )
    top_plate = add_extruded_polygon_mesh(
        stage,
        f"{visual_root}/RecessedTopPlate",
        plate_outline,
        plate_z[0],
        plate_z[1],
        aluminium,
    )

    grid = manual_geometry["mounting_hole_marker_grid"]
    columns = int(grid["columns"])
    rows = int(grid["rows"])
    spacing_x, spacing_y = (float(item) for item in grid["spacing_m"])
    corner_mask = grid["corner_mask"]
    outer_columns = int(corner_mask["outer_columns_each_side"])
    outer_rows = int(corner_mask["outer_rows_each_end"])
    expected_visible_markers = int(corner_mask["visible_marker_count"])
    custom_attr(
        top_plate.GetPrim(),
        "romoB:topHoleGridColumns",
        Sdf.ValueTypeNames.Int,
        columns,
    )
    custom_attr(
        top_plate.GetPrim(),
        "romoB:topHoleGridRows",
        Sdf.ValueTypeNames.Int,
        rows,
    )
    custom_attr(
        top_plate.GetPrim(),
        "romoB:topHolePitchM",
        Sdf.ValueTypeNames.Double,
        spacing_x,
    )
    custom_attr(
        top_plate.GetPrim(),
        "romoB:visibleTopHoleCount",
        Sdf.ValueTypeNames.Int,
        expected_visible_markers,
    )
    marker_z = plate_z[1] + 0.0006
    visible_marker_count = 0
    for row in range(rows):
        y = (row - (rows - 1) / 2.0) * spacing_y
        for column in range(columns):
            in_outer_rows = row < outer_rows or row >= rows - outer_rows
            in_outer_columns = (
                column < outer_columns or column >= columns - outer_columns
            )
            if in_outer_rows and in_outer_columns:
                continue
            x = (column - (columns - 1) / 2.0) * spacing_x
            add_cylinder(
                stage,
                f"{visual_root}/TopHoleMarker_r{row:02d}_c{column:02d}",
                UsdGeom.Tokens.z,
                float(grid["marker_radius_m"]),
                0.0012,
                (x, y, marker_z),
                dark_detail,
            )
            visible_marker_count += 1
    if visible_marker_count != expected_visible_markers:
        raise ValueError(
            "Top-hole visibility mask produced "
            f"{visible_marker_count} markers, expected {expected_visible_markers}"
        )

    service_covers = manual_geometry["top_service_covers"]
    for index, centre_xy in enumerate(service_covers["centres_xy_m"]):
        add_cylinder(
            stage,
            f"{visual_root}/TopServiceCover_{index + 1}",
            UsdGeom.Tokens.z,
            float(service_covers["radius_m"]),
            0.0016,
            (float(centre_xy[0]), float(centre_xy[1]), plate_z[1] + 0.0008),
            romo_blue,
        )

    steering_limit = value(limits["individual_steering_deg"])
    wheel_radius = float(proxy["wheel_radius_m"]["value"])
    wheel_width = float(proxy["wheel_width_m"]["value"])
    wheel_positions = proxy["wheel_center_positions_m"]
    steering_positions = proxy["steering_axis_positions_m"]
    wheel_house = manual_geometry["wheel_house_arch"]
    wheel_house_radius = float(wheel_house["opening_radius_m"])
    wheel_house_segments = int(wheel_house["arc_segments_per_wheel"])
    if int(wheel_house["arches_per_wheel"]) != 1:
        raise ValueError("The manual exterior requires one wheel-house arch per wheel")
    if bool(wheel_house["separate_fender_ribbon"]):
        raise ValueError("The manual exterior must not add a second fender arch")

    collision_sweep = manual_geometry["collision_steering_sweep_audit"]
    collision_sweep_limit = float(collision_sweep["steering_limit_deg"])
    if not math.isclose(collision_sweep_limit, steering_limit, abs_tol=1e-12):
        raise ValueError("Collision sweep and steering-joint limits must match")

    body_outline_x_values = sorted({float(point[0]) for point in outline})

    for side_key, side_sign, display_name in (
        ("left", 1.0, "Left"),
        ("right", -1.0, "Right"),
    ):
        side_wheel_centres = [
            (
                float(wheel_positions[corner][0]),
                float(wheel_positions[corner][2]),
            )
            for corner in (
                f"front_{side_key}",
                f"rear_{side_key}",
            )
        ]
        wheel_house_profile = single_circular_wheel_house_profile(
            body_x_values=body_outline_x_values,
            wheel_centres_xz=side_wheel_centres,
            opening_radius=wheel_house_radius,
            body_bottom_z=float(sections["side_shell"][0]),
            arc_segments=wheel_house_segments,
        )
        body_x_values = [point[0] for point in wheel_house_profile]
        side_y_values = [
            side_sign * positive_body_half_width_at_x(outline, x)
            for x in body_x_values
        ]
        lower_profile_local = [
            point[1] - chassis_center_z for point in wheel_house_profile
        ]
        side_panel = add_profiled_side_panel_mesh(
            stage,
            f"{visual_root}/{display_name}SidePanelWithWheelWells",
            body_x_values,
            side_y_values,
            lower_profile_local,
            side_shell_z[1],
            aluminium,
        )
        custom_attr(
            side_panel.GetPrim(),
            "romoB:geometryQuality",
            Sdf.ValueTypeNames.String,
            wheel_house["quality"],
        )
        custom_attr(
            side_panel.GetPrim(),
            "romoB:wheelHouseShape",
            Sdf.ValueTypeNames.String,
            "single_circular_arch_per_wheel",
        )
        custom_attr(
            side_panel.GetPrim(),
            "romoB:wheelHouseOpeningRadiusM",
            Sdf.ValueTypeNames.Double,
            wheel_house_radius,
        )
        custom_attr(
            side_panel.GetPrim(),
            "romoB:wheelHouseArcSegmentsPerWheel",
            Sdf.ValueTypeNames.Int,
            wheel_house_segments,
        )
        custom_attr(
            side_panel.GetPrim(),
            "romoB:archesPerWheel",
            Sdf.ValueTypeNames.Int,
            1,
        )
        custom_attr(
            side_panel.GetPrim(),
            "romoB:visibleProfileUsesSteeringSweep",
            Sdf.ValueTypeNames.Bool,
            False,
        )

    flat_end_half_width = outline_scale[1] * 0.48
    bumper_half_width = 0.100
    bumper_top_z = 0.127 - chassis_center_z
    for end_name, end_sign in (("Front", 1.0), ("Rear", -1.0)):
        end_x = end_sign * length / 2.0
        add_yz_panel_mesh(
            stage,
            f"{visual_root}/{end_name}UpperPanel",
            x=end_x,
            y_min=-flat_end_half_width,
            y_max=flat_end_half_width,
            z_min=bumper_top_z,
            z_max=side_shell_z[1],
            color=aluminium,
        )
        add_yz_panel_mesh(
            stage,
            f"{visual_root}/{end_name}LowerBumper",
            x=end_x,
            y_min=-bumper_half_width,
            y_max=bumper_half_width,
            z_min=side_shell_z[0],
            z_max=bumper_top_z,
            color=romo_blue,
        )
        for side_name, y_min, y_max in (
            ("Left", bumper_half_width, flat_end_half_width),
            ("Right", -flat_end_half_width, -bumper_half_width),
        ):
            add_yz_panel_mesh(
                stage,
                f"{visual_root}/{end_name}Lower{side_name}Panel",
                x=end_x,
                y_min=y_min,
                y_max=y_max,
                z_min=side_shell_z[0],
                z_max=bumper_top_z,
                color=aluminium,
            )
    add_translated_box(
        stage,
        f"{visual_root}/RightControlPanel",
        dimensions=(0.105, 0.004, 0.043),
        centre=(0.0, -body_width / 2.0 + 0.002, 0.268 - chassis_center_z),
        color=dark_detail,
    )

    collision_root = f"{chassis_path}/Collision"
    UsdGeom.Scope.Define(stage, collision_root)
    compound_collision = proxy["chassis_compound_collision"]
    for part in compound_collision["parts"]:
        centre_world = tuple(float(item) for item in part["centre_world_m"])
        centre_local = (
            centre_world[0],
            centre_world[1],
            centre_world[2] - chassis_center_z,
        )
        collider = add_translated_box(
            stage,
            f"{collision_root}/{part['name']}",
            dimensions=tuple(float(item) for item in part["dimensions_m"]),
            centre=centre_local,
            color=(0.18, 0.32, 0.52),
            collision=True,
        )
        collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
        custom_attr(
            collider.GetPrim(),
            "romoB:geometryQuality",
            Sdf.ValueTypeNames.String,
            compound_collision["quality"],
        )

    joints_root = UsdGeom.Scope.Define(stage, "/World/ROMOBOpenArm/Joints")
    custom_attr(joints_root.GetPrim(), "romoB:jointOrder", Sdf.ValueTypeNames.String, "front_left, front_right, rear_left, rear_right")

    knuckle_size = tuple(float(item) for item in proxy["knuckle_body_size_m"]["value"])
    steering_command_multiplier = float(
        spec["romo_b"]["steering_command_convention"]["manual_to_usd_joint_multiplier"]
    )
    wheel_max_torque = value(limits["wheel_motor_max_torque_nm"])
    steering_max_torque_nm = value(limits["steering_servo_max_torque_kg_cm"]) * 0.0980665
    steering_speed_deg_s = 60.0 / value(limits["steering_servo_no_load_s_per_60_deg"])
    steering_speed_rad_s = math.radians(steering_speed_deg_s)
    wheel_velocity_limit_deg_s = math.degrees(
        value(limits["platform_linear_speed_m_s"]) / wheel_radius
    )
    drive_gains = proxy["drive_gains"]

    for corner in ("front_left", "front_right", "rear_left", "rear_right"):
        steering_xyz = tuple(float(item) for item in steering_positions[corner])
        wheel_xyz = tuple(float(item) for item in wheel_positions[corner])
        corner_path = "/World/ROMOBOpenArm/" + "".join(part.title() for part in corner.split("_"))
        knuckle_path = f"{corner_path}/Knuckle"
        wheel_path = f"{corner_path}/Wheel"

        UsdGeom.Xform.Define(stage, corner_path)
        knuckle = UsdGeom.Xform.Define(stage, knuckle_path)
        set_translate(knuckle.GetPrim(), steering_xyz)
        add_mass(knuckle.GetPrim(), float(masses["each_knuckle"]))
        custom_attr(knuckle.GetPrim(), "romoB:massQuality", Sdf.ValueTypeNames.String, "manual_approximate_allocation")
        knuckle_visual = add_box(
            stage,
            f"{knuckle_path}/Visual",
            dimensions=knuckle_size,
            color=(0.36, 0.38, 0.41),
            collision=True,
        )
        knuckle_visual.CreatePurposeAttr(UsdGeom.Tokens.guide)
        custom_attr(
            stage.GetPrimAtPath(f"{knuckle_path}/Visual"),
            "romoB:geometryQuality",
            Sdf.ValueTypeNames.String,
            proxy["knuckle_body_size_m"]["quality"],
        )

        wheel = UsdGeom.Xform.Define(stage, wheel_path)
        set_translate(wheel.GetPrim(), wheel_xyz)
        add_mass(wheel.GetPrim(), float(masses["each_wheel"]))
        custom_attr(wheel.GetPrim(), "romoB:massQuality", Sdf.ValueTypeNames.String, "manual_approximate_allocation")
        wheel_geom = UsdGeom.Cylinder.Define(stage, f"{wheel_path}/Collision")
        wheel_geom.CreateAxisAttr(UsdGeom.Tokens.y)
        wheel_geom.CreateRadiusAttr(wheel_radius)
        wheel_geom.CreateHeightAttr(wheel_width)
        wheel_geom.CreateDisplayColorAttr([Gf.Vec3f(0.025, 0.025, 0.025)])
        UsdPhysics.CollisionAPI.Apply(wheel_geom.GetPrim()).CreateCollisionEnabledAttr(True)
        wheel_geom.CreatePurposeAttr(UsdGeom.Tokens.guide)

        wheel_visual_root = f"{wheel_path}/Visuals"
        UsdGeom.Scope.Define(stage, wheel_visual_root)
        tire_visual = UsdGeom.Cylinder.Define(stage, f"{wheel_visual_root}/Tire")
        tire_visual.CreateAxisAttr(UsdGeom.Tokens.y)
        tire_visual.CreateRadiusAttr(wheel_radius)
        tire_visual.CreateHeightAttr(wheel_width - 0.002)
        tire_visual.CreateDisplayColorAttr([Gf.Vec3f(0.025, 0.025, 0.025)])
        side_sign = 1.0 if wheel_xyz[1] > 0.0 else -1.0
        hub_spec = manual_geometry["wheel_hub_visual"]
        hub_depth = float(hub_spec["depth_m"])
        add_cylinder(
            stage,
            f"{wheel_visual_root}/Hub",
            UsdGeom.Tokens.y,
            float(hub_spec["hub_radius_m"]),
            hub_depth,
            (0.0, side_sign * (wheel_width / 2.0 - hub_depth / 2.0 - 0.0005), 0.0),
            (0.48, 0.51, 0.54),
        )
        cap_depth = 0.002
        add_cylinder(
            stage,
            f"{wheel_visual_root}/HubCap",
            UsdGeom.Tokens.y,
            float(hub_spec["cap_radius_m"]),
            cap_depth,
            (0.0, side_sign * (wheel_width / 2.0 - cap_depth / 2.0), 0.0),
            romo_blue,
        )

        steer_joint = add_revolute_joint(
            stage,
            f"/World/ROMOBOpenArm/Joints/{corner}_steering_joint",
            body0=chassis_path,
            body1=knuckle_path,
            axis="Z",
            local_pos0=(steering_xyz[0], steering_xyz[1], steering_xyz[2] - chassis_center_z),
            local_pos1=(0.0, 0.0, 0.0),
            lower_deg=-steering_limit,
            upper_deg=steering_limit,
        )
        steer_drive = UsdPhysics.DriveAPI.Apply(steer_joint.GetPrim(), "angular")
        steer_drive.CreateTypeAttr("force")
        steer_drive.CreateTargetPositionAttr(0.0)
        steer_drive.CreateMaxForceAttr(steering_max_torque_nm)
        steer_drive.CreateStiffnessAttr(float(drive_gains["steering_stiffness"]))
        steer_drive.CreateDampingAttr(float(drive_gains["steering_damping"]))
        PhysxSchema.PhysxJointAPI.Apply(steer_joint.GetPrim()).CreateMaxJointVelocityAttr(
            steering_speed_deg_s
        )
        custom_attr(
            steer_joint.GetPrim(),
            "romoB:approxNoLoadSpeedRadS",
            Sdf.ValueTypeNames.Double,
            steering_speed_rad_s,
        )
        manual_pivot_deg = float(spec["romo_b"]["pivot_steering_deg"][corner])
        custom_attr(
            steer_joint.GetPrim(),
            "romoB:manualCommandToUsdMultiplier",
            Sdf.ValueTypeNames.Double,
            steering_command_multiplier,
        )
        custom_attr(
            steer_joint.GetPrim(),
            "romoB:manualPivotPresetDeg",
            Sdf.ValueTypeNames.Double,
            manual_pivot_deg,
        )
        custom_attr(
            steer_joint.GetPrim(),
            "romoB:usdPivotTargetDeg",
            Sdf.ValueTypeNames.Double,
            manual_pivot_deg * steering_command_multiplier,
        )

        wheel_offset_y = wheel_xyz[1] - steering_xyz[1]
        drive_joint = add_revolute_joint(
            stage,
            f"/World/ROMOBOpenArm/Joints/{corner}_wheel_joint",
            body0=knuckle_path,
            body1=wheel_path,
            axis="Y",
            local_pos0=(0.0, wheel_offset_y, 0.0),
            local_pos1=(0.0, 0.0, 0.0),
        )
        wheel_drive = UsdPhysics.DriveAPI.Apply(drive_joint.GetPrim(), "angular")
        wheel_drive.CreateTypeAttr("force")
        wheel_drive.CreateTargetVelocityAttr(0.0)
        wheel_drive.CreateMaxForceAttr(wheel_max_torque)
        wheel_drive.CreateStiffnessAttr(0.0)
        wheel_drive.CreateDampingAttr(float(drive_gains["wheel_velocity_damping"]))
        PhysxSchema.PhysxJointAPI.Apply(drive_joint.GetPrim()).CreateMaxJointVelocityAttr(
            wheel_velocity_limit_deg_s
        )
        custom_attr(
            drive_joint.GetPrim(),
            "romoB:manualMotorRatedSpeedRpm",
            Sdf.ValueTypeNames.Double,
            value(limits["wheel_motor_rated_speed_rpm"]),
        )
        custom_attr(
            drive_joint.GetPrim(),
            "romoB:manualMotorMaxSpeedRpm",
            Sdf.ValueTypeNames.Double,
            value(limits["wheel_motor_max_speed_rpm"]),
        )
        custom_attr(
            drive_joint.GetPrim(),
            "romoB:nominalNoSlipLinearSpeedLimit",
            Sdf.ValueTypeNames.Double,
            value(limits["platform_linear_speed_m_s"]),
        )

    mass_total = float(masses["chassis"]) + 4.0 * (
        float(masses["each_wheel"]) + float(masses["each_knuckle"])
    )
    custom_attr(robot.GetPrim(), "romoB:allocatedPlatformMass", Sdf.ValueTypeNames.Double, mass_total)
    custom_attr(
        robot.GetPrim(),
        "romoB:manualPlatformMass",
        Sdf.ValueTypeNames.Double,
        value(spec["romo_b"]["mass_kg"]["platform_total"]),
    )

    arm_path = "/World/ROMOBOpenArm/OpenArm"
    arm = UsdGeom.Xform.Define(stage, arm_path)
    arm.GetPrim().GetPayloads().AddPayload(spec["sources"]["openarm_asset"]["url"])
    arm.GetPrim().GetVariantSet("Physics").SetVariantSelection("PhysX")
    arm.GetPrim().GetVariantSet("Sensor").SetVariantSelection("None")
    set_translate(arm.GetPrim(), spec["openarm_mount"]["position_m"])
    set_orient_wxyz(arm.GetPrim(), spec["openarm_mount"]["orientation_wxyz"])
    custom_attr(arm.GetPrim(), "romoB:mountReference", Sdf.ValueTypeNames.String, spec["openarm_mount"]["reference"])
    custom_attr(arm.GetPrim(), "romoB:mountQuality", Sdf.ValueTypeNames.String, spec["openarm_mount"]["quality"])
    custom_attr(
        arm.GetPrim(),
        "romoB:alternateFrontEdgeSetbackX",
        Sdf.ValueTypeNames.Double,
        float(spec["openarm_mount"]["alternate_x_if_17_cm_means_setback_from_front_edge_m"]),
    )

    # Pin every collision-relevant enable switch from the Isaac Sim 5.1 OpenArm
    # payload in this stronger layer. The relative paths are audited in the JSON
    # spec so the generated stage remains deterministic while the payload stays
    # unloaded/offline during authoring.
    openarm_source = spec["sources"]["openarm_asset"]
    for relative_path in openarm_source["active_collider_paths"]:
        stage.OverridePrim(f"{arm_path}/{relative_path}").CreateAttribute(
            "physics:collisionEnabled", Sdf.ValueTypeNames.Bool, custom=False
        ).Set(True)
    for relative_path in openarm_source["active_rigid_body_paths"]:
        stage.OverridePrim(f"{arm_path}/{relative_path}").CreateAttribute(
            "physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool, custom=False
        ).Set(True)

    # The payload's all-zero pose contains non-adjacent collider penetration once
    # articulation self-collision is enabled. Seed and hold the audited symmetric
    # shoulder/open-gripper pose without changing NVIDIA's collision geometry.
    initial_state = openarm_source["collision_safe_initial_joint_state"]
    for joint_state in initial_state["positions"]:
        joint_override = stage.OverridePrim(f"{arm_path}/{joint_state['path']}")
        instance = joint_state["instance"]
        position = float(joint_state["value"])
        joint_override.CreateAttribute(
            f"state:{instance}:physics:position",
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(position)
        joint_override.CreateAttribute(
            f"state:{instance}:physics:velocity",
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(0.0)
        if joint_state["drive_target"]:
            joint_override.CreateAttribute(
                f"drive:{instance}:physics:targetPosition",
                Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(position)
            joint_override.CreateAttribute(
                f"drive:{instance}:physics:targetVelocity",
                Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(0.0)

    # The NVIDIA asset is a fixed-base articulation. Disable its world root joint,
    # then connect the physical base body to the mobile chassis. This makes the arm
    # links part of the ROMO-B articulation instead of leaving a nested fixed robot.
    arm_root_joint = stage.OverridePrim(f"{arm_path}/{spec['sources']['openarm_asset']['world_root_joint']}")
    arm_root_joint.SetActive(False)

    mount_position = tuple(float(item) for item in spec["openarm_mount"]["position_m"])
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/World/ROMOBOpenArm/Joints/openarm_mount_joint")
    fixed_joint.CreateBody0Rel().SetTargets([Sdf.Path(chassis_path)])
    fixed_joint.CreateBody1Rel().SetTargets(
        [Sdf.Path(f"{arm_path}/{spec['sources']['openarm_asset']['base_body']}")]
    )
    fixed_joint.CreateLocalPos0Attr(
        Gf.Vec3f(mount_position[0], mount_position[1], mount_position[2] - chassis_center_z)
    )
    fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    mount_orientation = spec["openarm_mount"]["orientation_wxyz"]
    fixed_joint.CreateLocalRot0Attr(
        Gf.Quatf(
            mount_orientation[0],
            mount_orientation[1],
            mount_orientation[2],
            mount_orientation[3],
        )
    )
    fixed_joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    fixed_joint.CreateExcludeFromArticulationAttr(False)
    custom_attr(fixed_joint.GetPrim(), "romoB:mountOffsetSource", Sdf.ValueTypeNames.String, "user_supplied_approximately_17_cm")

    stage.GetRootLayer().Save()
    generated_text = Path(temporary_name).read_text(encoding="utf-8").rstrip() + "\n"
    Path(temporary_name).write_text(generated_text, encoding="utf-8")
    os.replace(temporary_name, output_path)
    atexit.unregister(remove_incomplete_output)
    print(f"wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.spec.resolve(), args.output.resolve())
