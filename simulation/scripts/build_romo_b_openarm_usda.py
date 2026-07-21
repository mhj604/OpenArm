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
    height = value(dims["overall_height"])
    clearance = value(dims["ground_clearance"])
    chassis_height = height - clearance
    chassis_center_z = clearance + chassis_height / 2.0
    chassis_path = "/World/ROMOBOpenArm/Chassis"
    chassis = UsdGeom.Xform.Define(stage, chassis_path)
    set_translate(chassis.GetPrim(), (0.0, 0.0, chassis_center_z))
    add_mass(chassis.GetPrim(), float(masses["chassis"]))
    custom_attr(chassis.GetPrim(), "romoB:massAllocationQuality", Sdf.ValueTypeNames.String, masses["quality"])
    add_box(
        stage,
        f"{chassis_path}/Body",
        dimensions=(length, float(proxy["chassis_visual_width_m"]["value"]), chassis_height),
        color=(0.08, 0.24, 0.52),
    )
    chassis_collision = add_box(
        stage,
        f"{chassis_path}/Collision",
        dimensions=(length, float(proxy["chassis_collision_width_m"]["value"]), chassis_height),
        color=(0.08, 0.24, 0.52),
        collision=True,
    )
    chassis_collision.CreatePurposeAttr(UsdGeom.Tokens.guide)
    custom_attr(
        chassis_collision.GetPrim(),
        "romoB:geometryQuality",
        Sdf.ValueTypeNames.String,
        proxy["chassis_collision_width_m"]["quality"],
    )

    joints_root = UsdGeom.Scope.Define(stage, "/World/ROMOBOpenArm/Joints")
    custom_attr(joints_root.GetPrim(), "romoB:jointOrder", Sdf.ValueTypeNames.String, "front_left, front_right, rear_left, rear_right")

    steering_limit = value(limits["individual_steering_deg"])
    wheel_radius = float(proxy["wheel_radius_m"]["value"])
    wheel_width = float(proxy["wheel_width_m"]["value"])
    wheel_positions = proxy["wheel_center_positions_m"]
    steering_positions = proxy["steering_axis_positions_m"]
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
        add_box(
            stage,
            f"{knuckle_path}/Visual",
            dimensions=knuckle_size,
            color=(0.75, 0.58, 0.12),
            collision=True,
        )
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
