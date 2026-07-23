#!/usr/bin/env python3
"""Load and step the ROMO-B/OpenArm stage in Isaac Sim 5.1.

This is an integration smoke test, not a calibrated dynamics experiment. The
NVIDIA OpenArm payload must be reachable through the Isaac asset cache or the
network.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


SIMULATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = SIMULATION_ROOT / "assets" / "romo_b_openarm_manual_proxy.usda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--gui", action="store_true", help="Show the Isaac Sim UI instead of running headless.")
    parser.add_argument(
        "--steering-sweep",
        action="store_true",
        help="Exercise all ROMO-B steering joints at +30, -30, then 0 degrees while stepping.",
    )
    parser.add_argument(
        "--openarm-asset",
        help="Use this local/remote OpenArm USD instead of the committed payload URL for this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset = args.asset.resolve()
    if not asset.is_file():
        raise FileNotFoundError(asset)
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.steering_sweep and args.steps < 4:
        raise ValueError(
            "--steering-sweep requires --steps of at least 4 "
            "(one neutral baseline plus three target segments)"
        )

    # Omniverse and pxr imports must occur after SimulationApp is instantiated.
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": not args.gui,
            "create_new_stage": False,
            "sync_loads": True,
            "anti_aliasing": 0,
            "disable_viewport_updates": not args.gui,
            # All result/error lines below are explicitly flushed before Kit's
            # fast standalone-process teardown.
            "fast_shutdown": True,
        }
    )
    try:
        import carb
        import omni.usd
        from omni.physx.bindings._physx import SETTING_UPDATE_TO_USD
        from pxr import (
            Gf,
            PhysicsSchemaTools,
            PhysxSchema,
            Usd,
            UsdGeom,
            UsdPhysics,
            UsdUtils,
        )

        context = omni.usd.get_context()
        initial_load = (
            omni.usd.UsdContextInitialLoadSet.LOAD_NONE
            if args.openarm_asset
            else omni.usd.UsdContextInitialLoadSet.LOAD_ALL
        )
        if not context.open_stage(str(asset), None, initial_load):
            raise RuntimeError(f"Isaac Sim refused to open {asset}")
        while context.get_stage_loading_status()[2] > 0:
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError(f"Isaac Sim did not open {asset}")

        if args.openarm_asset:
            arm_prim = stage.GetPrimAtPath("/World/ROMOBOpenArm/OpenArm")
            stage.SetEditTarget(stage.GetSessionLayer())
            arm_prim.GetPayloads().ClearPayloads()
            arm_prim.GetPayloads().AddPayload(args.openarm_asset)
            stage.Load("/World/ROMOBOpenArm/OpenArm")
            while context.get_stage_loading_status()[2] > 0:
                app.update()

        robot = stage.GetPrimAtPath("/World/ROMOBOpenArm")
        arm_body = stage.GetPrimAtPath("/World/ROMOBOpenArm/OpenArm/openarm_body_link")
        if not robot or not robot.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise RuntimeError("ROMO-B articulation root is missing")
        if not arm_body or not arm_body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                "The NVIDIA OpenArm payload did not resolve. Connect the Isaac asset cache/network and retry."
            )

        articulation = PhysxSchema.PhysxArticulationAPI(robot)
        if not articulation or not bool(articulation.GetEnabledSelfCollisionsAttr().Get()):
            raise RuntimeError("Combined ROMO-B/OpenArm articulation self-collision is not enabled")
        active_prims = [prim for prim in stage.TraverseAll() if prim.IsActive()]
        colliders = [prim for prim in active_prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
        disabled_colliders = [
            prim.GetPath().pathString
            for prim in colliders
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False
        ]
        if len(colliders) < 32 or disabled_colliders:
            raise RuntimeError(
                f"Full collision policy failed: colliders={len(colliders)}, disabled={disabled_colliders}"
            )
        filtered_pairs = [
            prim.GetPath().pathString
            for prim in active_prims
            if prim.HasAPI(UsdPhysics.FilteredPairsAPI)
            and UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
        ]
        filtered_groups = [
            prim.GetPath().pathString
            for prim in active_prims
            if UsdPhysics.CollisionGroup(prim)
            and (
                UsdPhysics.CollisionGroup(prim).GetFilteredGroupsRel().GetTargets()
                or bool(UsdPhysics.CollisionGroup(prim).GetInvertFilteredGroupsAttr().Get())
            )
        ]
        if filtered_pairs or filtered_groups:
            raise RuntimeError(
                f"Collision filters remain: pairs={filtered_pairs}, groups={filtered_groups}"
            )

        rigid_bodies = [prim for prim in active_prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        disabled_rigid_bodies = [
            prim.GetPath().pathString
            for prim in rigid_bodies
            if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not True
        ]
        if len(rigid_bodies) < 32 or disabled_rigid_bodies:
            raise RuntimeError(
                "Full rigid-body policy failed: "
                f"rigid_bodies={len(rigid_bodies)}, disabled={disabled_rigid_bodies}"
            )
        joints = [prim for prim in active_prims if bool(UsdPhysics.Joint(prim))]
        ignored_true_joint_flags = [
            prim.GetPath().pathString
            for prim in joints
            if UsdPhysics.Joint(prim).GetCollisionEnabledAttr().Get() is True
        ]
        if len(joints) < 31 or ignored_true_joint_flags:
            raise RuntimeError(
                "Articulation joint policy failed: "
                f"joints={len(joints)}, ignored_true_flags={ignored_true_joint_flags}"
            )

        from omni.physx import get_physx_simulation_interface

        # Contact reporting is a diagnostic session override, not an asset
        # dependency. Keep it active for every neutral/sweep step so transient
        # self-collisions cannot hide behind an initial-only contact check.
        previous_edit_target = stage.GetEditTarget()
        stage.SetEditTarget(stage.GetSessionLayer())
        for prim in rigid_bodies:
            contact_report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            contact_report.CreateThresholdAttr(0.0)
        stage.SetEditTarget(previous_edit_target)

        # Force live PhysX transforms back into USD for the chassis-frame AABB
        # and displacement checks. Fabric otherwise keeps the authoritative
        # poses outside the USD xforms queried below.
        settings = carb.settings.get_settings()
        previous_update_to_usd = settings.get_as_bool(SETTING_UPDATE_TO_USD)
        previous_fabric_enabled = settings.get_as_bool("/physics/fabricEnabled")
        settings.set(SETTING_UPDATE_TO_USD, True)
        settings.set("/physics/fabricEnabled", False)

        simulation_interface = get_physx_simulation_interface()
        stage_cache = UsdUtils.StageCache.Get()
        stage_id = stage_cache.Insert(stage).ToLongInt()
        attached_here = simulation_interface.get_attached_stage() != stage_id
        if attached_here:
            simulation_interface.attach_stage(stage_id)

        def current_contact_metrics() -> dict[tuple[str, str], tuple[float, float]]:
            contact_headers, contact_data = simulation_interface.get_contact_report()
            metrics: dict[tuple[str, str], tuple[float, float]] = {}
            for header in contact_headers:
                start = int(header.contact_data_offset)
                count = int(header.num_contact_data)
                points = contact_data[start : start + count]
                if not points:
                    continue
                path0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
                path1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
                pair = tuple(sorted((path0, path1)))
                minimum_separation = min(float(point.separation) for point in points)
                maximum_impulse = max(
                    math.sqrt(sum(float(component) ** 2 for component in point.impulse))
                    for point in points
                )
                previous = metrics.get(pair)
                if previous:
                    minimum_separation = min(minimum_separation, previous[0])
                    maximum_impulse = max(maximum_impulse, previous[1])
                metrics[pair] = (minimum_separation, maximum_impulse)
            return metrics

        def audit_contacts(
            phase: str,
        ) -> tuple[set[tuple[str, str]], float, float]:
            metrics = current_contact_metrics()
            unexpected_ground_contacts = []
            penetrating_self_contacts = []
            impulsive_self_contacts = []
            non_ground_pairs: set[tuple[str, str]] = set()
            minimum_non_ground_separation = math.inf
            maximum_non_ground_impulse = 0.0
            for pair, (minimum_separation, maximum_impulse) in metrics.items():
                ground_path = "/World/Ground"
                if ground_path in pair:
                    other = pair[0] if pair[1] == ground_path else pair[1]
                    if not other.endswith("/Wheel/Collision"):
                        unexpected_ground_contacts.append(pair)
                    continue
                non_ground_pairs.add(pair)
                minimum_non_ground_separation = min(
                    minimum_non_ground_separation, minimum_separation
                )
                maximum_non_ground_impulse = max(
                    maximum_non_ground_impulse, maximum_impulse
                )
                if not math.isfinite(minimum_separation) or minimum_separation < -1e-5:
                    penetrating_self_contacts.append((pair, minimum_separation))
                if not math.isfinite(maximum_impulse) or maximum_impulse > 1e-4:
                    impulsive_self_contacts.append((pair, maximum_impulse))
            if unexpected_ground_contacts or penetrating_self_contacts or impulsive_self_contacts:
                raise RuntimeError(
                    f"{phase} contact audit failed: "
                    f"unexpected_ground={unexpected_ground_contacts}, "
                    f"penetrating_self={penetrating_self_contacts}, "
                    f"impulsive_self={impulsive_self_contacts}"
                )
            return (
                non_ground_pairs,
                minimum_non_ground_separation,
                maximum_non_ground_impulse,
            )

        def wheel_chassis_aabb_gap() -> float:
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [
                    UsdGeom.Tokens.default_,
                    UsdGeom.Tokens.guide,
                    UsdGeom.Tokens.proxy,
                    UsdGeom.Tokens.render,
                ],
            )
            chassis_prim = stage.GetPrimAtPath("/World/ROMOBOpenArm/Chassis")
            if not chassis_prim:
                raise RuntimeError("ROMO-B chassis is missing")
            chassis_colliders = [
                prim
                for prim in Usd.PrimRange(chassis_prim)
                if prim.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                is True
            ]
            if len(chassis_colliders) != 4:
                raise RuntimeError(
                    "ROMO-B compound chassis collision is incomplete: "
                    f"{[prim.GetPath().pathString for prim in chassis_colliders]}"
                )
            # Compare in the chassis frame. World-axis AABBs widen when the whole
            # robot yaws or rolls and can report a false wheel/chassis overlap.
            chassis_ranges = [
                (
                    collider.GetPath().pathString,
                    cache.ComputeRelativeBound(
                        collider, chassis_prim
                    ).ComputeAlignedRange(),
                )
                for collider in chassis_colliders
            ]
            minimum_gap = math.inf
            for corner in ("FrontLeft", "FrontRight", "RearLeft", "RearRight"):
                wheel_collider = stage.GetPrimAtPath(
                    f"/World/ROMOBOpenArm/{corner}/Wheel/Collision"
                )
                wheel_range = cache.ComputeRelativeBound(
                    wheel_collider, chassis_prim
                ).ComputeAlignedRange()
                for chassis_path, chassis_range in chassis_ranges:
                    # Two three-dimensional AABBs are separated if at least one
                    # axis has a positive face-to-face gap. Taking the maximum
                    # axis gap therefore accepts the full-width deck above the
                    # wheel and the end members ahead/behind it, while still
                    # rejecting overlap on all three axes.
                    axis_gaps = [
                        max(
                            float(
                                chassis_range.GetMin()[axis]
                                - wheel_range.GetMax()[axis]
                            ),
                            float(
                                wheel_range.GetMin()[axis]
                                - chassis_range.GetMax()[axis]
                            ),
                        )
                        for axis in range(3)
                    ]
                    gap = max(axis_gaps)
                    if gap < -1e-5:
                        raise RuntimeError(
                            "Steering sweep overlaps compound chassis collider "
                            f"{chassis_path} and {corner} wheel chassis-frame "
                            f"AABBs by {-gap:.6f} m; axis_gaps={axis_gaps}"
                        )
                    minimum_gap = min(minimum_gap, gap)
            return minimum_gap

        steering_corners = (
            ("front_left", "FrontLeft", 1.0),
            ("front_right", "FrontRight", -1.0),
            ("rear_left", "RearLeft", 1.0),
            ("rear_right", "RearRight", -1.0),
        )

        def current_steering_angles() -> dict[str, float]:
            chassis_prim = stage.GetPrimAtPath("/World/ROMOBOpenArm/Chassis")
            chassis_world_inverse = (
                UsdGeom.Xformable(chassis_prim)
                .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                .GetInverse()
            )
            angles = {}
            for joint_name, prim_name, side in steering_corners:
                wheel_prim = stage.GetPrimAtPath(
                    f"/World/ROMOBOpenArm/{prim_name}/Wheel"
                )
                wheel_world_position = (
                    UsdGeom.Xformable(wheel_prim)
                    .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    .ExtractTranslation()
                )
                wheel_chassis_position = chassis_world_inverse.Transform(
                    Gf.Vec3d(wheel_world_position)
                )
                steering_joint = UsdPhysics.Joint(
                    stage.GetPrimAtPath(
                        f"/World/ROMOBOpenArm/Joints/{joint_name}_steering_joint"
                    )
                )
                pivot = steering_joint.GetLocalPos0Attr().Get()
                radial_x = float(wheel_chassis_position[0] - pivot[0])
                radial_y = float(wheel_chassis_position[1] - pivot[1])
                angles[joint_name] = math.degrees(
                    math.atan2(-side * radial_x, side * radial_y)
                )
            return angles

        minimum_wheel_chassis_gap = math.inf
        maximum_steering_target_error = 0.0
        runtime_non_ground_pairs: set[tuple[str, str]] = set()
        runtime_minimum_non_ground_separation = math.inf
        runtime_maximum_non_ground_impulse = 0.0
        physics_dt = 1.0 / 60.0
        physics_time = 0.0

        def step_physics() -> None:
            nonlocal physics_time
            simulation_interface.simulate(physics_dt, physics_time)
            simulation_interface.fetch_results()
            physics_time += physics_dt
            if args.gui:
                app.update()

        def sample_runtime_contacts(
            phase: str,
        ) -> tuple[set[tuple[str, str]], float, float]:
            nonlocal runtime_minimum_non_ground_separation
            nonlocal runtime_maximum_non_ground_impulse
            pairs, minimum_separation, maximum_impulse = audit_contacts(phase)
            runtime_non_ground_pairs.update(pairs)
            runtime_minimum_non_ground_separation = min(
                runtime_minimum_non_ground_separation, minimum_separation
            )
            runtime_maximum_non_ground_impulse = max(
                runtime_maximum_non_ground_impulse, maximum_impulse
            )
            return pairs, minimum_separation, maximum_impulse

        try:
            # Establish the physics baseline after one neutral step, then audit
            # every requested fixed 1/60 s step. Direct PhysX stepping avoids
            # tying simulated time to the wall-clock duration of app.update().
            step_physics()
            (
                initial_non_ground_pairs,
                initial_minimum_non_ground_separation,
                initial_maximum_non_ground_impulse,
            ) = sample_runtime_contacts("Initial neutral step")
            initial_positions = {
                prim.GetPath().pathString: UsdGeom.Xformable(prim)
                .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                .ExtractTranslation()
                for prim in rigid_bodies
            }

            if args.steering_sweep:
                steering_drives = [
                    UsdPhysics.DriveAPI(
                        stage.GetPrimAtPath(
                            f"/World/ROMOBOpenArm/Joints/{joint_name}_steering_joint"
                        ),
                        "angular",
                    )
                    for joint_name, _, _ in steering_corners
                ]
                sweep_steps = args.steps - 1
                segment_lengths = [sweep_steps // 3] * 3
                for index in range(sweep_steps % 3):
                    segment_lengths[index] += 1
                for target, segment_steps in zip((30.0, -30.0, 0.0), segment_lengths):
                    for drive in steering_drives:
                        target_attr = drive.GetTargetPositionAttr()
                        if not target_attr:
                            raise RuntimeError(
                                "A ROMO-B steering drive is missing its target position"
                            )
                        target_attr.Set(target)
                    for step_index in range(segment_steps):
                        step_physics()
                        sample_runtime_contacts(
                            f"Steering {target:+.0f} degree step {step_index + 1}/{segment_steps}"
                        )
                    actual_angles = current_steering_angles()
                    target_errors = {
                        name: abs(angle - target)
                        for name, angle in actual_angles.items()
                    }
                    segment_maximum_error = max(target_errors.values())
                    maximum_steering_target_error = max(
                        maximum_steering_target_error, segment_maximum_error
                    )
                    if segment_maximum_error > 1.0:
                        raise RuntimeError(
                            f"Steering target {target:+.0f} degrees was not reached: "
                            f"actual={actual_angles}"
                        )
                    minimum_wheel_chassis_gap = min(
                        minimum_wheel_chassis_gap,
                        wheel_chassis_aabb_gap(),
                    )
            else:
                hold_steps = args.steps - 1
                for step_index in range(hold_steps):
                    step_physics()
                    sample_runtime_contacts(f"Hold step {step_index + 1}/{hold_steps}")
                minimum_wheel_chassis_gap = wheel_chassis_aabb_gap()

            chassis = stage.GetPrimAtPath("/World/ROMOBOpenArm/Chassis")
            matrix = UsdGeom.Xformable(chassis).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            xyz = matrix.ExtractTranslation()
            if not all(math.isfinite(float(xyz[index])) for index in range(3)):
                raise RuntimeError(f"Non-finite chassis pose after stepping: {xyz}")

            largest_displacement = ("none", 0.0)
            for prim in rigid_bodies:
                path = prim.GetPath().pathString
                current = (
                    UsdGeom.Xformable(prim)
                    .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    .ExtractTranslation()
                )
                if not all(math.isfinite(float(current[index])) for index in range(3)):
                    raise RuntimeError(
                        f"Non-finite rigid-body pose after stepping: {path} -> {current}"
                    )
                displacement = float((current - initial_positions[path]).GetLength())
                if displacement > largest_displacement[1]:
                    largest_displacement = (path, displacement)
            if largest_displacement[1] > 2.0:
                raise RuntimeError(
                    "Collision instability moved a rigid body more than 2 m: "
                    f"{largest_displacement[0]} -> {largest_displacement[1]:.6f} m"
                )
        finally:
            if attached_here:
                simulation_interface.detach_stage()
            settings.set(SETTING_UPDATE_TO_USD, previous_update_to_usd)
            settings.set("/physics/fabricEnabled", previous_fabric_enabled)

        print("Isaac Sim smoke test: PASS", flush=True)
        print(f"asset: {asset}", flush=True)
        print(f"steps: {args.steps}", flush=True)
        print(
            f"active collision state: {len(colliders)} colliders, {len(rigid_bodies)} rigid bodies, "
            f"{len(joints)} articulation joints (self-collision enabled, no filtered pairs)",
            flush=True,
        )
        print(f"steering sweep exercised: {args.steering_sweep}", flush=True)
        if args.steering_sweep:
            print(
                f"maximum steering target error: {maximum_steering_target_error:.6f} degrees",
                flush=True,
            )
        print(
            f"fixed physics duration: {physics_time:.6f} s at {physics_dt:.6f} s/step",
            flush=True,
        )
        print(
            "minimum wheel-to-compound-chassis AABB gap: "
            f"{minimum_wheel_chassis_gap:.6f} m",
            flush=True,
        )
        print(
            "initial non-ground contact-offset pairs: "
            f"{len(initial_non_ground_pairs)} (minimum separation "
            f"{initial_minimum_non_ground_separation:.6f} m, maximum impulse "
            f"{initial_maximum_non_ground_impulse:.6f})",
            flush=True,
        )
        print(
            "full-run non-ground contact-offset pairs: "
            f"{len(runtime_non_ground_pairs)} (minimum separation "
            f"{runtime_minimum_non_ground_separation:.6f} m, maximum impulse "
            f"{runtime_maximum_non_ground_impulse:.6f})",
            flush=True,
        )
        print(
            f"max rigid-body displacement: {largest_displacement[1]:.6f} m "
            f"({largest_displacement[0]})",
            flush=True,
        )
        print(
            f"chassis translation: ({float(xyz[0]):.6f}, {float(xyz[1]):.6f}, {float(xyz[2]):.6f})",
            flush=True,
        )
    except Exception as exc:
        # Emit and preserve a nonzero exit before Kit fast teardown, which exits
        # the interpreter with status 0 before a pending Python exception can be
        # re-raised. The OS releases this standalone process's GPU resources.
        print(
            f"Isaac Sim smoke test: FAIL ({type(exc).__name__}): {exc}",
            flush=True,
        )
        os._exit(1)
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
