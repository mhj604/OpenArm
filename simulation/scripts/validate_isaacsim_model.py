#!/usr/bin/env python3
"""Validate the ROMO-B/OpenArm USD contract without starting the renderer."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


SIMULATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = SIMULATION_ROOT / "config" / "romo_b_openarm_spec.json"
DEFAULT_ASSET = SIMULATION_ROOT / "assets" / "romo_b_openarm_manual_proxy.usda"
ROBOT = "/World/ROMOBOpenArm"
CHASSIS = f"{ROBOT}/Chassis"
ARM = f"{ROBOT}/OpenArm"
JOINTS = f"{ROBOT}/Joints"
CORNERS = ("front_left", "front_right", "rear_left", "rear_right")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(actual: float, expected: float, label: str, tolerance: float = 1e-5) -> None:
    require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance), f"{label}: {actual} != {expected}")


def relation_targets(joint: UsdPhysics.Joint, which: int) -> list[str]:
    relationship = joint.GetBody0Rel() if which == 0 else joint.GetBody1Rel()
    return [target.pathString for target in relationship.GetTargets()]


def local_translation(stage: Usd.Stage, path: str) -> tuple[float, float, float]:
    prim = stage.GetPrimAtPath(path)
    require(bool(prim), f"Missing prim: {path}")
    matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
    translation = matrix.ExtractTranslation()
    return tuple(float(translation[index]) for index in range(3))


def mass(stage: Usd.Stage, path: str) -> float:
    prim = stage.GetPrimAtPath(path)
    require(bool(prim), f"Missing mass prim: {path}")
    mass_value = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
    require(mass_value is not None, f"Missing physics:mass on {path}")
    return float(mass_value)


def validate_mass_shape(stage: Usd.Stage, path: str) -> None:
    """Require enough shape data for PhysX to derive a rigid body's inertia."""

    prim = stage.GetPrimAtPath(path)
    require(bool(prim), f"Missing rigid body: {path}")
    require(prim.HasAPI(UsdPhysics.RigidBodyAPI), f"Missing RigidBodyAPI on {path}")
    has_collider = any(
        descendant.IsActive()
        and descendant.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(descendant).GetCollisionEnabledAttr().Get() is not False
        for descendant in Usd.PrimRange(prim)
    )
    inertia_attr = UsdPhysics.MassAPI(prim).GetDiagonalInertiaAttr()
    has_authored_inertia = bool(inertia_attr) and inertia_attr.HasAuthoredValueOpinion()
    require(
        has_collider or has_authored_inertia,
        f"Rigid body needs a collider or authored diagonal inertia: {path}",
    )


def validate_collision_policy(stage: Usd.Stage, spec: dict, openarm_resolved: bool) -> None:
    """Require every authored collider and inter-body collision path to remain enabled."""

    policy = spec["simulation_proxy"]["collision_policy"]
    robot_prim = stage.GetPrimAtPath(ROBOT)
    require(
        bool(robot_prim.GetAttribute("romoB:allAuthoredCollidersEnabled").Get())
        == bool(policy["all_authored_colliders_enabled"]),
        "Root collider policy metadata mismatch",
    )
    require(
        bool(robot_prim.GetAttribute("romoB:filteredPairsAllowed").Get())
        == bool(policy["allow_filtered_pairs"]),
        "Root filtered-pair policy metadata mismatch",
    )
    require(
        bool(robot_prim.GetAttribute("romoB:allActiveRigidBodiesEnabled").Get())
        == bool(policy["all_active_rigid_bodies_enabled"]),
        "Root rigid-body policy metadata mismatch",
    )
    require(
        robot_prim.GetAttribute("romoB:collisionRuntimeLimitation").Get()
        == policy["runtime_limitation"],
        "Root collision runtime-limitation metadata mismatch",
    )

    active_prims = [prim for prim in stage.TraverseAll() if prim.IsActive()]
    colliders = [prim for prim in active_prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
    rigid_bodies = [prim for prim in active_prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [prim for prim in active_prims if bool(UsdPhysics.Joint(prim))]
    expected = (
        policy["expected_active_counts_with_openarm"]
        if openarm_resolved
        else policy["expected_active_counts_without_openarm_payload"]
    )
    minimum_colliders = int(expected["colliders"])
    minimum_rigid_bodies = int(expected["rigid_bodies"])
    minimum_joints = int(expected["joints"])
    require(
        len(colliders) >= minimum_colliders,
        f"Expected at least {minimum_colliders} active colliders, got {len(colliders)}",
    )
    require(
        len(rigid_bodies) >= minimum_rigid_bodies,
        f"Expected at least {minimum_rigid_bodies} active rigid bodies, got {len(rigid_bodies)}",
    )
    require(
        len(joints) >= minimum_joints,
        f"Expected at least {minimum_joints} active joints, got {len(joints)}",
    )
    ignored_true_joint_flags = [
        prim.GetPath().pathString
        for prim in joints
        if UsdPhysics.Joint(prim).GetCollisionEnabledAttr().Get() is True
    ]
    require(
        not ignored_true_joint_flags,
        "PhysX ignores per-joint collisionEnabled inside articulations; remove warning-only true flags: "
        f"{ignored_true_joint_flags}",
    )
    if bool(policy["all_authored_colliders_enabled"]):
        disabled_colliders = [
            prim.GetPath().pathString
            for prim in colliders
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True
        ]
        require(not disabled_colliders, f"Disabled colliders found: {disabled_colliders}")
    if bool(policy["all_active_rigid_bodies_enabled"]):
        disabled_rigid_bodies = [
            prim.GetPath().pathString
            for prim in rigid_bodies
            if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not True
        ]
        require(
            not disabled_rigid_bodies,
            f"Disabled rigid bodies found: {disabled_rigid_bodies}",
        )
    if openarm_resolved:
        openarm_source = spec["sources"]["openarm_asset"]
        for relative_path in openarm_source["active_collider_paths"]:
            path = f"{ARM}/{relative_path}"
            prim = stage.GetPrimAtPath(path)
            require(
                bool(prim) and prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI),
                f"Missing configured OpenArm collider: {path}",
            )
        for relative_path in openarm_source["active_rigid_body_paths"]:
            path = f"{ARM}/{relative_path}"
            prim = stage.GetPrimAtPath(path)
            require(
                bool(prim) and prim.IsActive() and prim.HasAPI(UsdPhysics.RigidBodyAPI),
                f"Missing configured OpenArm rigid body: {path}",
            )
    if not bool(policy["allow_filtered_pairs"]):
        filtered_pairs = []
        filtered_groups = []
        for prim in active_prims:
            if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                targets = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
                if targets:
                    filtered_pairs.append(
                        (prim.GetPath().pathString, [target.pathString for target in targets])
                    )
            group = UsdPhysics.CollisionGroup(prim)
            if group:
                targets = group.GetFilteredGroupsRel().GetTargets()
                inverted = bool(group.GetInvertFilteredGroupsAttr().Get())
                if targets or inverted:
                    filtered_groups.append(
                        (
                            prim.GetPath().pathString,
                            inverted,
                            [target.pathString for target in targets],
                        )
                    )
        require(not filtered_pairs, f"Filtered collision pairs found: {filtered_pairs}")
        require(not filtered_groups, f"Filtered collision groups found: {filtered_groups}")


def validate_romob_external_geometry(stage: Usd.Stage, spec: dict) -> None:
    """Check the manual-constrained external envelope and compound chassis shapes."""

    dims = spec["romo_b"]["dimensions_m"]
    proxy = spec["simulation_proxy"]
    geometry = proxy["manual_external_geometry"]
    chassis_height = (
        float(dims["overall_height"]["value"])
        - float(dims["ground_clearance"]["value"])
    )
    chassis_center_z = (
        float(dims["ground_clearance"]["value"]) + chassis_height / 2.0
    )

    def mesh_range(path: str) -> tuple[list[float], list[float]]:
        mesh = UsdGeom.Mesh.Get(stage, path)
        require(bool(mesh), f"Missing manual external mesh: {path}")
        points = mesh.GetPointsAttr().Get()
        require(bool(points), f"Manual external mesh has no points: {path}")
        minimum = [min(float(point[axis]) for point in points) for axis in range(3)]
        maximum = [max(float(point[axis]) for point in points) for axis in range(3)]
        return minimum, maximum

    visual_root = f"{CHASSIS}/Visuals"
    shell_minimum, shell_maximum = mesh_range(f"{visual_root}/SideShell")
    close(
        shell_maximum[0] - shell_minimum[0],
        float(dims["overall_length"]["value"]),
        "manual shell length",
    )
    close(
        shell_maximum[1] - shell_minimum[1],
        float(proxy["chassis_visual_width_m"]["value"]),
        "manual shell width",
    )
    shell_section = geometry["body_sections_world_z_m"]["side_shell"]
    close(
        shell_minimum[2] + chassis_center_z,
        float(shell_section[0]),
        "manual shell lower Z",
    )
    close(
        shell_maximum[2] + chassis_center_z,
        float(shell_section[1]),
        "manual shell upper Z",
    )
    rim_minimum, rim_maximum = mesh_range(f"{visual_root}/TopRim")
    rim_mesh = UsdGeom.Mesh.Get(stage, f"{visual_root}/TopRim")
    rim_points = rim_mesh.GetPointsAttr().Get()
    expected_rim_points = 4 * len(
        geometry["body_outline_normalized_xy"]["value"]
    )
    require(
        len(rim_points) == expected_rim_points,
        "Top rim must contain separate outer/inner contours at both Z levels: "
        f"{len(rim_points)} != {expected_rim_points}",
    )
    close(
        rim_maximum[2] + chassis_center_z,
        float(dims["overall_height"]["value"]),
        "manual rim overall height",
    )
    close(
        rim_maximum[0] - rim_minimum[0],
        float(dims["overall_length"]["value"]),
        "manual rim length",
    )

    fender_outer_y = float(geometry["wheel_fender_arch"]["outer_side_face_y_m"])
    for corner in CORNERS:
        camel = "".join(part.title() for part in corner.split("_"))
        fender_minimum, fender_maximum = mesh_range(
            f"{visual_root}/{camel}FenderArch"
        )
        actual_outer_y = (
            fender_maximum[1]
            if proxy["wheel_center_positions_m"][corner][1] > 0.0
            else fender_minimum[1]
        )
        close(
            abs(actual_outer_y),
            fender_outer_y,
            f"{corner} fender outer side",
        )
        require(
            bool(stage.GetPrimAtPath(f"{ROBOT}/{camel}/Wheel/Visuals/Tire")),
            f"Missing {corner} manual tire visual",
        )
        require(
            bool(stage.GetPrimAtPath(f"{ROBOT}/{camel}/Wheel/Visuals/Hub")),
            f"Missing {corner} manual hub visual",
        )

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.guide,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.render,
        ],
    )
    chassis_prim = stage.GetPrimAtPath(CHASSIS)
    for part in proxy["chassis_compound_collision"]["parts"]:
        path = f"{CHASSIS}/Collision/{part['name']}"
        prim = stage.GetPrimAtPath(path)
        require(bool(prim), f"Missing compound chassis collider: {path}")
        require(prim.HasAPI(UsdPhysics.CollisionAPI), f"Missing CollisionAPI: {path}")
        require(
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True,
            f"Disabled compound chassis collider: {path}",
        )
        extent = cache.ComputeRelativeBound(prim, chassis_prim).ComputeAlignedRange()
        actual_minimum = extent.GetMin()
        actual_maximum = extent.GetMax()
        actual_dimensions = actual_maximum - actual_minimum
        actual_centre = (actual_minimum + actual_maximum) / 2.0
        expected_centre = [
            float(part["centre_world_m"][0]),
            float(part["centre_world_m"][1]),
            float(part["centre_world_m"][2]) - chassis_center_z,
        ]
        for axis, expected_dimension in enumerate(part["dimensions_m"]):
            close(
                float(actual_dimensions[axis]),
                float(expected_dimension),
                f"{part['name']} collider dimension {'xyz'[axis]}",
            )
            close(
                float(actual_centre[axis]),
                expected_centre[axis],
                f"{part['name']} collider centre {'xyz'[axis]}",
            )


def validate_openarm_initial_state(stage: Usd.Stage, spec: dict, openarm_resolved: bool) -> None:
    """Validate the self-collision-safe OpenArm start state and matching drive targets."""

    initial_state = spec["sources"]["openarm_asset"]["collision_safe_initial_joint_state"]
    for joint_state in initial_state["positions"]:
        path = f"{ARM}/{joint_state['path']}"
        prim = stage.GetPrimAtPath(path)
        require(bool(prim), f"Missing OpenArm initial-state override prim: {path}")
        instance = joint_state["instance"]
        position = float(joint_state["value"])
        expected_unit = "degree" if instance == "angular" else "metre"
        require(joint_state["unit"] == expected_unit, f"Initial-state unit mismatch: {path}")
        position_attr = prim.GetAttribute(f"state:{instance}:physics:position")
        velocity_attr = prim.GetAttribute(f"state:{instance}:physics:velocity")
        require(bool(position_attr), f"Missing initial joint position: {path}")
        require(bool(velocity_attr), f"Missing initial joint velocity: {path}")
        close(float(position_attr.Get()), position, f"initial joint position {path}")
        close(float(velocity_attr.Get()), 0.0, f"initial joint velocity {path}")
        if joint_state["drive_target"]:
            target_position = prim.GetAttribute(f"drive:{instance}:physics:targetPosition")
            target_velocity = prim.GetAttribute(f"drive:{instance}:physics:targetVelocity")
            require(bool(target_position), f"Missing initial drive target: {path}")
            require(bool(target_velocity), f"Missing initial drive velocity target: {path}")
            close(float(target_position.Get()), position, f"initial drive target {path}")
            close(float(target_velocity.Get()), 0.0, f"initial drive velocity {path}")

        if openarm_resolved:
            require(prim.IsActive() and bool(UsdPhysics.Joint(prim)), f"Invalid OpenArm joint: {path}")
            if instance == "angular":
                typed_joint = UsdPhysics.RevoluteJoint(prim)
            else:
                typed_joint = UsdPhysics.PrismaticJoint(prim)
            require(bool(typed_joint), f"OpenArm joint/state type mismatch: {path}")
            lower = float(typed_joint.GetLowerLimitAttr().Get())
            upper = float(typed_joint.GetUpperLimitAttr().Get())
            require(
                lower - 1e-6 <= position <= upper + 1e-6,
                f"Initial position outside joint limits: {path}",
            )


def validate_articulation_graph(stage: Usd.Stage) -> None:
    active_prims = [prim for prim in stage.TraverseAll() if prim.IsActive()]
    articulation_roots = [
        prim.GetPath().pathString
        for prim in active_prims
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    require(articulation_roots == [ROBOT], f"Expected exactly one active articulation root at {ROBOT}: {articulation_roots}")

    graph: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = defaultdict(int)
    anchor_errors: list[tuple[str, float]] = []
    rotation_errors: list[tuple[str, float]] = []

    for prim in active_prims:
        joint = UsdPhysics.Joint(prim)
        if not joint:
            continue
        body0 = relation_targets(joint, 0)
        body1 = relation_targets(joint, 1)
        if len(body0) != 1 or len(body1) != 1:
            continue
        parent, child = body0[0], body1[0]
        graph[parent].append(child)
        indegree[child] += 1
        require(indegree[child] == 1, f"Rigid body has more than one parent joint: {child}")

        parent_prim = stage.GetPrimAtPath(parent)
        child_prim = stage.GetPrimAtPath(child)
        if not parent_prim or not child_prim:
            continue
        parent_matrix = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        child_matrix = UsdGeom.Xformable(child_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        local_pos0 = joint.GetLocalPos0Attr().Get() or Gf.Vec3f(0.0)
        local_pos1 = joint.GetLocalPos1Attr().Get() or Gf.Vec3f(0.0)
        world_pos0 = parent_matrix.Transform(Gf.Vec3d(local_pos0))
        world_pos1 = child_matrix.Transform(Gf.Vec3d(local_pos1))
        anchor_errors.append((prim.GetPath().pathString, float((world_pos0 - world_pos1).GetLength())))

        local_rot0 = joint.GetLocalRot0Attr().Get() or Gf.Quatf(1.0)
        local_rot1 = joint.GetLocalRot1Attr().Get() or Gf.Quatf(1.0)
        world_rot0 = parent_matrix.ExtractRotationQuat() * Gf.Quatd(
            float(local_rot0.GetReal()), Gf.Vec3d(local_rot0.GetImaginary())
        )
        world_rot1 = child_matrix.ExtractRotationQuat() * Gf.Quatd(
            float(local_rot1.GetReal()), Gf.Vec3d(local_rot1.GetImaginary())
        )
        dot = abs(
            world_rot0.GetReal() * world_rot1.GetReal()
            + Gf.Dot(world_rot0.GetImaginary(), world_rot1.GetImaginary())
        )
        rotation_errors.append((prim.GetPath().pathString, abs(1.0 - float(dot))))

    state: dict[str, int] = {}

    def visit(node: str) -> None:
        require(state.get(node, 0) != 1, f"Joint graph contains a cycle at {node}")
        if state.get(node, 0) == 2:
            return
        state[node] = 1
        for child in graph.get(node, []):
            visit(child)
        state[node] = 2

    for node in set(graph) | set(indegree):
        visit(node)

    require(anchor_errors, "No two-body joints were found")
    worst_anchor = max(anchor_errors, key=lambda item: item[1])
    worst_rotation = max(rotation_errors, key=lambda item: item[1])
    require(worst_anchor[1] <= 1e-5, f"Joint anchor position mismatch: {worst_anchor}")
    require(worst_rotation[1] <= 1e-5, f"Joint anchor rotation mismatch: {worst_rotation}")


def validate(
    spec_path: Path,
    asset_path: Path,
    resolve_openarm: bool,
    openarm_asset_override: str | None = None,
) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    stage = Usd.Stage.Open(str(asset_path), load=Usd.Stage.LoadNone)
    require(stage is not None, f"Could not open stage: {asset_path}")

    close(UsdGeom.GetStageMetersPerUnit(stage), 1.0, "metersPerUnit")
    require(UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z, "Stage must be Z-up")
    require(stage.GetDefaultPrim().GetPath() == Sdf.Path("/World"), "Default prim must be /World")
    require(bool(UsdPhysics.Scene.Get(stage, "/World/PhysicsScene")), "Missing PhysicsScene")

    robot_prim = stage.GetPrimAtPath(ROBOT)
    require(bool(robot_prim), f"Missing robot prim: {ROBOT}")
    require(robot_prim.HasAPI(UsdPhysics.ArticulationRootAPI), "Robot root lacks ArticulationRootAPI")
    solver = spec["simulation_proxy"]["articulation_solver"]
    physx_articulation = PhysxSchema.PhysxArticulationAPI(robot_prim)
    require(bool(physx_articulation), "Robot root lacks PhysxArticulationAPI")
    require(
        bool(physx_articulation.GetEnabledSelfCollisionsAttr().Get())
        == bool(solver["enabled_self_collisions"]),
        "PhysX self-collision setting mismatch",
    )
    require(
        int(physx_articulation.GetSolverPositionIterationCountAttr().Get())
        == int(solver["position_iteration_count"]),
        "PhysX position iteration count mismatch",
    )
    require(
        int(physx_articulation.GetSolverVelocityIterationCountAttr().Get())
        == int(solver["velocity_iteration_count"]),
        "PhysX velocity iteration count mismatch",
    )

    dims = spec["romo_b"]["dimensions_m"]
    proxy = spec["simulation_proxy"]
    limits = spec["romo_b"]["limits"]
    masses = proxy["mass_allocation_kg"]
    steering_command_multiplier = float(
        spec["romo_b"]["steering_command_convention"]["manual_to_usd_joint_multiplier"]
    )
    expected_steering_velocity_deg_s = 60.0 / float(
        limits["steering_servo_no_load_s_per_60_deg"]["value"]
    )
    expected_wheel_velocity_deg_s = math.degrees(
        float(limits["platform_linear_speed_m_s"]["value"])
        / float(proxy["wheel_radius_m"]["value"])
    )
    close(
        float(robot_prim.GetAttribute("romoB:manualSteeringCommandToUsdMultiplier").Get()),
        steering_command_multiplier,
        "manual-to-USD steering command multiplier",
    )
    chassis_height = float(dims["overall_height"]["value"]) - float(dims["ground_clearance"]["value"])
    chassis_center_z = float(dims["ground_clearance"]["value"]) + chassis_height / 2.0
    close(local_translation(stage, CHASSIS)[2], chassis_center_z, "chassis centre Z")

    total_mass = mass(stage, CHASSIS)
    validate_mass_shape(stage, CHASSIS)
    for corner in CORNERS:
        camel = "".join(part.title() for part in corner.split("_"))
        corner_root = f"{ROBOT}/{camel}"
        knuckle_path = f"{corner_root}/Knuckle"
        wheel_path = f"{corner_root}/Wheel"
        expected_steering = tuple(float(v) for v in proxy["steering_axis_positions_m"][corner])
        expected_wheel = tuple(float(v) for v in proxy["wheel_center_positions_m"][corner])
        for actual, expected, axis in zip(local_translation(stage, knuckle_path), expected_steering, "xyz"):
            close(actual, expected, f"{corner} steering {axis}")
        for actual, expected, axis in zip(local_translation(stage, wheel_path), expected_wheel, "xyz"):
            close(actual, expected, f"{corner} wheel {axis}")
        total_mass += mass(stage, knuckle_path) + mass(stage, wheel_path)
        validate_mass_shape(stage, knuckle_path)
        validate_mass_shape(stage, wheel_path)

        steering = UsdPhysics.RevoluteJoint.Get(stage, f"{JOINTS}/{corner}_steering_joint")
        require(bool(steering), f"Missing {corner} steering joint")
        require(steering.GetAxisAttr().Get() == "Z", f"{corner} steering axis must be Z")
        close(float(steering.GetLowerLimitAttr().Get()), -float(limits["individual_steering_deg"]["value"]), f"{corner} lower steering limit")
        close(float(steering.GetUpperLimitAttr().Get()), float(limits["individual_steering_deg"]["value"]), f"{corner} upper steering limit")
        require(relation_targets(steering, 0) == [CHASSIS], f"{corner} steering body0 mismatch")
        require(relation_targets(steering, 1) == [knuckle_path], f"{corner} steering body1 mismatch")
        steering_physx = PhysxSchema.PhysxJointAPI(steering.GetPrim())
        require(bool(steering_physx), f"Missing PhysxJointAPI on {corner} steering")
        close(
            float(steering_physx.GetMaxJointVelocityAttr().Get()),
            expected_steering_velocity_deg_s,
            f"{corner} steering max velocity",
            tolerance=1e-4,
        )
        close(
            float(steering.GetPrim().GetAttribute("romoB:manualCommandToUsdMultiplier").Get()),
            steering_command_multiplier,
            f"{corner} steering command multiplier",
        )
        manual_pivot_deg = float(spec["romo_b"]["pivot_steering_deg"][corner])
        close(
            float(steering.GetPrim().GetAttribute("romoB:manualPivotPresetDeg").Get()),
            manual_pivot_deg,
            f"{corner} manual pivot preset",
        )
        close(
            float(steering.GetPrim().GetAttribute("romoB:usdPivotTargetDeg").Get()),
            manual_pivot_deg * steering_command_multiplier,
            f"{corner} USD pivot target",
        )

        wheel_joint = UsdPhysics.RevoluteJoint.Get(stage, f"{JOINTS}/{corner}_wheel_joint")
        require(bool(wheel_joint), f"Missing {corner} wheel joint")
        require(wheel_joint.GetAxisAttr().Get() == "Y", f"{corner} wheel axis must be Y")
        require(relation_targets(wheel_joint, 0) == [knuckle_path], f"{corner} wheel body0 mismatch")
        require(relation_targets(wheel_joint, 1) == [wheel_path], f"{corner} wheel body1 mismatch")
        wheel_physx = PhysxSchema.PhysxJointAPI(wheel_joint.GetPrim())
        require(bool(wheel_physx), f"Missing PhysxJointAPI on {corner} wheel")
        close(
            float(wheel_physx.GetMaxJointVelocityAttr().Get()),
            expected_wheel_velocity_deg_s,
            f"{corner} wheel max velocity",
            tolerance=1e-4,
        )
        close(
            float(wheel_joint.GetPrim().GetAttribute("romoB:manualMotorRatedSpeedRpm").Get()),
            float(limits["wheel_motor_rated_speed_rpm"]["value"]),
            f"{corner} rated motor speed",
        )
        close(
            float(wheel_joint.GetPrim().GetAttribute("romoB:manualMotorMaxSpeedRpm").Get()),
            float(limits["wheel_motor_max_speed_rpm"]["value"]),
            f"{corner} max motor speed",
        )

    close(total_mass, float(spec["romo_b"]["mass_kg"]["platform_total"]["value"]), "allocated ROMO-B mass")
    close(total_mass, float(masses["aggregate"]), "declared proxy mass")
    validate_romob_external_geometry(stage, spec)

    arm_prim = stage.GetPrimAtPath(ARM)
    require(bool(arm_prim), f"Missing OpenArm prim: {ARM}")
    payload_metadata = str(arm_prim.GetMetadata("payload"))
    require(spec["sources"]["openarm_asset"]["url"] in payload_metadata, "OpenArm payload URL mismatch")
    actual_mount = local_translation(stage, ARM)
    for actual, expected, axis in zip(actual_mount, spec["openarm_mount"]["position_m"], "xyz"):
        close(actual, float(expected), f"OpenArm mount {axis}")

    root_joint_path = f"{ARM}/{spec['sources']['openarm_asset']['world_root_joint']}"
    root_joint = stage.GetPrimAtPath(root_joint_path)
    require(bool(root_joint), f"Missing authored OpenArm root-joint override: {root_joint_path}")
    require(not root_joint.IsActive(), "The OpenArm world root joint must be inactive")

    mount_joint = UsdPhysics.FixedJoint.Get(stage, f"{JOINTS}/openarm_mount_joint")
    require(bool(mount_joint), "Missing OpenArm chassis fixed joint")
    require(relation_targets(mount_joint, 0) == [CHASSIS], "OpenArm mount body0 mismatch")
    expected_arm_body = f"{ARM}/{spec['sources']['openarm_asset']['base_body']}"
    require(relation_targets(mount_joint, 1) == [expected_arm_body], "OpenArm mount body1 mismatch")
    require(mount_joint.GetExcludeFromArticulationAttr().Get() is False, "OpenArm mount must remain inside articulation")
    if resolve_openarm:
        if openarm_asset_override:
            stage.SetEditTarget(stage.GetSessionLayer())
            arm_prim.GetPayloads().ClearPayloads()
            arm_prim.GetPayloads().AddPayload(openarm_asset_override)
        stage.Load(ARM)
        require(stage.GetPrimAtPath(expected_arm_body).IsValid(), f"OpenArm payload did not resolve: {expected_arm_body}")
        arm_joints = [
            prim
            for prim in Usd.PrimRange(stage.GetPrimAtPath(ARM))
            if prim.IsActive() and bool(UsdPhysics.RevoluteJoint(prim))
        ]
        require(len(arm_joints) >= 14, f"Expected at least 14 active OpenArm revolute joints, got {len(arm_joints)}")

    validate_openarm_initial_state(stage, spec, resolve_openarm)
    validate_collision_policy(stage, spec, resolve_openarm)
    validate_articulation_graph(stage)

    print("Isaac Sim model validation: PASS")
    print(f"asset: {asset_path}")
    print(f"ROMO-B allocated mass: {total_mass:.3f} kg")
    print(f"OpenArm mount: {actual_mount} m")
    print(f"OpenArm payload resolved: {resolve_openarm}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument(
        "--resolve-openarm",
        action="store_true",
        help="Also resolve the remote NVIDIA payload and inspect its arm joints.",
    )
    parser.add_argument(
        "--openarm-asset",
        help="Resolve against this local/remote OpenArm asset instead of the committed URL (implies --resolve-openarm).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate(
        args.spec.resolve(),
        args.asset.resolve(),
        args.resolve_openarm or bool(args.openarm_asset),
        args.openarm_asset,
    )
