#!/usr/bin/env python3
"""Pure-Python contract tests for the audited ROMO-B/OpenArm model."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path


SIMULATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SIMULATION_ROOT.parent
SPEC_PATH = SIMULATION_ROOT / "config" / "romo_b_openarm_spec.json"
ASSET_PATH = SIMULATION_ROOT / "assets" / "romo_b_openarm_manual_proxy.usda"
MANUAL_PATH = REPOSITORY_ROOT / "source_files" / "hardware" / "ROMO-B_manual_verified_complete.md"
SCRIPTS_ROOT = SIMULATION_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from romo_b_geometry import (  # noqa: E402
    circular_arch_top_z,
    positive_body_half_width_at_x,
    single_circular_wheel_house_profile,
    steered_cylinder_section_top_z,
)
from run_isaacsim_smoke import (  # noqa: E402
    STEERING_JOINT_NAMES,
    steering_target_patterns,
)


class ModelContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        cls.asset_text = ASSET_PATH.read_text(encoding="utf-8")

    def assertClose(self, actual: float, expected: float, message: str = "") -> None:  # noqa: N802
        self.assertTrue(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), message or f"{actual} != {expected}")

    def test_source_manual_is_preserved(self) -> None:
        self.assertTrue(MANUAL_PATH.is_file())
        manual = MANUAL_PATH.read_text(encoding="utf-8")
        self.assertIn("756", manual)
        self.assertIn("597", manual)
        self.assertIn("4WID", manual)

    def test_manual_stated_dimensions_are_not_rounded_again(self) -> None:
        dims = self.spec["romo_b"]["dimensions_m"]
        expected = {
            "overall_length": 0.756,
            "overall_width": 0.597,
            "overall_height": 0.342,
            "ground_clearance": 0.062,
            "wheelbase": 0.323,
            "wheel_track": 0.529,
            "steering_knuckle_spacing": 0.390,
            "wheel_diameter": 0.206,
        }
        for name, expected_value in expected.items():
            self.assertClose(float(dims[name]["value"]), expected_value, name)
            self.assertIn("quality", dims[name])
            self.assertIn("manual_lines", dims[name])
        wheelbase = dims["wheelbase"]
        self.assertIs(wheelbase["default"], True)
        self.assertEqual(wheelbase["manual_pdf_pages"], [2, 5, 26, 35])
        self.assertEqual(int(wheelbase["occurrence_count"]), 6)
        self.assertClose(float(wheelbase["conflicting_value"]), 0.324)
        self.assertEqual(wheelbase["conflicting_pdf_pages"], [3])
        self.assertEqual(int(wheelbase["conflicting_occurrence_count"]), 1)

    def test_pivot_geometry_uses_active_wheelbase_default(self) -> None:
        dims = self.spec["romo_b"]["dimensions_m"]
        geometry = self.spec["romo_b"]["pivot_geometry"]
        wheelbase = float(dims["wheelbase"]["value"])
        knuckle_spacing = float(dims["steering_knuckle_spacing"]["value"])
        fixed_angle_deg = abs(float(self.spec["romo_b"]["pivot_steering_deg"]["front_left"]))
        ideal_angle_deg = math.degrees(math.atan(wheelbase / knuckle_spacing))
        deficit_deg = ideal_angle_deg - fixed_angle_deg
        virtual_icr_separation = wheelbase - knuckle_spacing * math.tan(math.radians(fixed_angle_deg))
        knuckle_radius = math.hypot(wheelbase / 2.0, knuckle_spacing / 2.0)
        residual_coefficient = knuckle_radius * math.sin(math.radians(deficit_deg))
        self.assertClose(float(geometry["ideal_centered_zero_slip_angle_deg"]), ideal_angle_deg)
        self.assertClose(float(geometry["fixed_angle_deficit_deg"]), deficit_deg)
        self.assertClose(float(geometry["front_rear_virtual_icr_separation_m_at_30_deg"]), virtual_icr_separation)
        self.assertClose(float(geometry["knuckle_radius_m"]), knuckle_radius)
        self.assertClose(float(geometry["lateral_residual_coefficient_m_per_rad"]), residual_coefficient)

    def test_geometry_derivations(self) -> None:
        self.assertEqual(self.spec["schema_version"], "1.4.0")
        dims = self.spec["romo_b"]["dimensions_m"]
        proxy = self.spec["simulation_proxy"]
        self.assertClose(float(proxy["wheel_radius_m"]["value"]), float(dims["wheel_diameter"]["value"]) / 2.0)
        self.assertClose(
            float(proxy["wheel_width_m"]["value"]),
            float(dims["overall_width"]["value"]) - float(dims["wheel_track"]["value"]),
        )
        for corner, position in proxy["steering_axis_positions_m"].items():
            if not isinstance(position, list):
                continue
            self.assertClose(abs(float(position[0])), float(dims["wheelbase"]["value"]) / 2.0, corner)
            self.assertClose(abs(float(position[1])), float(dims["steering_knuckle_spacing"]["value"]) / 2.0, corner)
        for corner, position in proxy["wheel_center_positions_m"].items():
            if not isinstance(position, list):
                continue
            self.assertClose(abs(float(position[1])), float(dims["wheel_track"]["value"]) / 2.0, corner)

        knuckle_size = proxy["knuckle_body_size_m"]
        self.assertEqual(knuckle_size["value"], [0.045, 0.045, 0.070])
        self.assertEqual(knuckle_size["axis_order"], ["x", "y", "z"])
        self.assertIn("unverified", knuckle_size["quality"])
        self.assertIn("collision_and_visual", knuckle_size["quality"])
        self.assertIn("not provide", knuckle_size["source"])

        self.assertClose(
            float(proxy["chassis_visual_width_m"]["value"]),
            float(dims["overall_width"]["value"]),
        )
        self.assertIn(
            "manual_stated",
            proxy["chassis_visual_width_m"]["quality"],
        )
        wheel_outer_span = (
            float(dims["wheel_track"]["value"])
            + float(proxy["wheel_width_m"]["value"])
        )
        self.assertClose(wheel_outer_span, float(dims["overall_width"]["value"]))
        self.assertClose(
            wheel_outer_span,
            float(proxy["chassis_visual_width_m"]["value"]),
        )

        manual_geometry = proxy["manual_external_geometry"]
        outline = manual_geometry["body_outline_normalized_xy"]
        self.assertEqual(len(outline["value"]), 12)
        self.assertEqual(outline["axis_scale_m"], [0.378, 0.2985])
        scaled_x = [
            float(point[0]) * float(outline["axis_scale_m"][0])
            for point in outline["value"]
        ]
        scaled_y = [
            float(point[1]) * float(outline["axis_scale_m"][1])
            for point in outline["value"]
        ]
        self.assertClose(max(scaled_x) - min(scaled_x), float(dims["overall_length"]["value"]))
        self.assertClose(
            max(scaled_y) - min(scaled_y),
            float(proxy["chassis_visual_width_m"]["value"]),
        )
        sections = manual_geometry["body_sections_world_z_m"]
        self.assertClose(float(sections["side_shell"][0]), float(dims["ground_clearance"]["value"]))
        self.assertClose(float(sections["top_rim"][1]), float(dims["overall_height"]["value"]))
        self.assertEqual(
            manual_geometry["top_plate_scale_xy"]["value"],
            [0.86, 0.75],
        )
        grid = manual_geometry["mounting_hole_marker_grid"]
        self.assertEqual(grid["columns"], 13)
        self.assertEqual(grid["rows"], 9)
        self.assertEqual(grid["spacing_m"], [0.050, 0.050])
        self.assertIn("user_supplied", grid["spacing_quality"])
        corner_mask = grid["corner_mask"]
        masked = 4 * int(corner_mask["outer_columns_each_side"]) * int(
            corner_mask["outer_rows_each_end"]
        )
        self.assertEqual(masked, 24)
        self.assertEqual(masked, int(corner_mask["masked_marker_count"]))
        self.assertEqual(
            grid["columns"] * grid["rows"] - masked,
            int(corner_mask["visible_marker_count"]),
        )
        self.assertEqual(int(corner_mask["visible_marker_count"]), 93)
        wheel_house = manual_geometry["wheel_house_arch"]
        collision_sweep = manual_geometry["collision_steering_sweep_audit"]
        self.assertClose(float(wheel_house["opening_radius_m"]), 0.126)
        self.assertEqual(int(wheel_house["arc_segments_per_wheel"]), 96)
        self.assertEqual(int(wheel_house["arches_per_wheel"]), 1)
        self.assertIs(wheel_house["separate_fender_ribbon"], False)
        self.assertIn("single_circle", wheel_house["quality"])
        self.assertIn("Invisible collision", collision_sweep["applies_to"])
        self.assertClose(
            float(collision_sweep["steering_limit_deg"]),
            float(
                self.spec["romo_b"]["limits"]["individual_steering_deg"][
                    "value"
                ]
            ),
        )
        self.assertEqual(int(collision_sweep["steering_sample_count"]), 361)
        self.assertEqual(int(collision_sweep["steering_sample_count"]) % 2, 1)

        chassis_collision_width = float(proxy["chassis_collision_width_m"]["value"])
        self.assertClose(chassis_collision_width, 0.32)
        steering_limit_rad = math.radians(
            float(self.spec["romo_b"]["limits"]["individual_steering_deg"]["value"])
        )
        wheel_sweep_bound = 2.0 * (
            0.195
            + (0.0695 - float(proxy["wheel_width_m"]["value"]) / 2.0)
            * math.cos(steering_limit_rad)
            - float(proxy["wheel_radius_m"]["value"]) * math.sin(steering_limit_rad)
        )
        rotated_knuckle_bound = 2.0 * (
            0.195
            - 0.0225
            * (math.cos(steering_limit_rad) + math.sin(steering_limit_rad))
        )
        self.assertLess(chassis_collision_width, wheel_sweep_bound)
        self.assertLess(chassis_collision_width, rotated_knuckle_bound)
        self.assertIn("not_measured", proxy["chassis_collision_width_m"]["quality"])

        collision_parts = proxy["chassis_compound_collision"]["parts"]
        self.assertEqual([part["name"] for part in collision_parts], [
            "Core",
            "TopDeck",
            "FrontLower",
            "RearLower",
        ])
        top_deck = collision_parts[1]
        self.assertClose(
            float(top_deck["dimensions_m"][1]),
            float(dims["overall_width"]["value"]),
        )
        top_deck_bottom = (
            float(top_deck["centre_world_m"][2])
            - float(top_deck["dimensions_m"][2]) / 2.0
        )
        expected_top_deck_bottom = (
            float(proxy["wheel_center_positions_m"]["front_left"][2])
            + float(wheel_house["opening_radius_m"])
            + 0.001
        )
        self.assertClose(top_deck_bottom, expected_top_deck_bottom)
        self.assertGreater(
            top_deck_bottom,
            float(dims["wheel_diameter"]["value"]),
        )

    def test_visible_wheel_house_is_one_circle_per_wheel(self) -> None:
        proxy = self.spec["simulation_proxy"]
        geometry = proxy["manual_external_geometry"]
        outline_spec = geometry["body_outline_normalized_xy"]
        outline = [
            (
                float(point[0]) * float(outline_spec["axis_scale_m"][0]),
                float(point[1]) * float(outline_spec["axis_scale_m"][1]),
            )
            for point in outline_spec["value"]
        ]
        self.assertClose(positive_body_half_width_at_x(outline, 0.0), 0.2985)
        self.assertClose(
            positive_body_half_width_at_x(outline, 0.378),
            0.2985 * 0.48,
        )

        wheel_house = geometry["wheel_house_arch"]
        radius = float(wheel_house["opening_radius_m"])
        body_bottom = float(geometry["body_sections_world_z_m"]["side_shell"][0])
        left_wheels = [
            proxy["wheel_center_positions_m"]["front_left"],
            proxy["wheel_center_positions_m"]["rear_left"],
        ]
        profile = single_circular_wheel_house_profile(
            body_x_values=sorted({point[0] for point in outline}),
            wheel_centres_xz=[
                (float(wheel[0]), float(wheel[2])) for wheel in left_wheels
            ],
            opening_radius=radius,
            body_bottom_z=body_bottom,
            arc_segments=int(wheel_house["arc_segments_per_wheel"]),
        )
        for wheel in left_wheels:
            centre_x = float(wheel[0])
            centre_z = float(wheel[2])
            self.assertClose(
                float(
                    circular_arch_top_z(
                        x=centre_x,
                        centre_x=centre_x,
                        centre_z=centre_z,
                        radius=radius,
                    )
                ),
                centre_z + radius,
            )
            for boundary_x in (centre_x - radius, centre_x + radius):
                boundary_values = [
                    z
                    for x, z in profile
                    if math.isclose(
                        x,
                        boundary_x,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ]
                self.assertEqual(len(boundary_values), 2)
                self.assertTrue(
                    any(
                        math.isclose(
                            z,
                            body_bottom,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        for z in boundary_values
                    )
                )
                self.assertTrue(
                    any(
                        math.isclose(
                            z,
                            centre_z,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        for z in boundary_values
                    )
                )
            arc_points = [
                (x, z)
                for x, z in profile
                if (
                    centre_x - radius - 1e-12
                    <= x
                    <= centre_x + radius + 1e-12
                    and z >= centre_z - 1e-12
                )
            ]
            self.assertGreater(
                len(arc_points),
                int(wheel_house["arc_segments_per_wheel"]),
            )
            for x, z in arc_points:
                self.assertClose(
                    (x - centre_x) ** 2 + (z - centre_z) ** 2,
                    radius**2,
                )

        collision_sweep = geometry["collision_steering_sweep_audit"]
        wheel = proxy["wheel_center_positions_m"]["front_left"]
        pivot = proxy["steering_axis_positions_m"]["front_left"]
        body_y = positive_body_half_width_at_x(outline, float(wheel[0]))
        tire_top = steered_cylinder_section_top_z(
            x=float(wheel[0]),
            side_plane_y=body_y,
            steering_pivot_xyz=pivot,
            neutral_wheel_xyz=wheel,
            cylinder_radius=float(proxy["wheel_radius_m"]["value"]),
            cylinder_half_width=float(proxy["wheel_width_m"]["value"]) / 2.0,
            steering_limit_deg=float(collision_sweep["steering_limit_deg"]),
            steering_sample_count=int(collision_sweep["steering_sample_count"]),
        )
        audit_envelope_top = steered_cylinder_section_top_z(
            x=float(wheel[0]),
            side_plane_y=body_y,
            steering_pivot_xyz=pivot,
            neutral_wheel_xyz=wheel,
            cylinder_radius=(
                float(proxy["wheel_radius_m"]["value"])
                + float(collision_sweep["radial_clearance_m"])
            ),
            cylinder_half_width=(
                float(proxy["wheel_width_m"]["value"]) / 2.0
                + float(collision_sweep["axial_clearance_m"])
            ),
            steering_limit_deg=float(collision_sweep["steering_limit_deg"]),
            steering_sample_count=int(collision_sweep["steering_sample_count"]),
        )
        self.assertIsNotNone(tire_top)
        self.assertIsNotNone(audit_envelope_top)
        self.assertGreater(float(audit_envelope_top), float(tire_top))
        self.assertIn("Invisible collision", collision_sweep["applies_to"])

    def test_manual_steering_sign_is_mapped_to_usd_positive_z(self) -> None:
        convention = self.spec["romo_b"]["steering_command_convention"]
        self.assertEqual(convention["manual_negative"], "left steering")
        self.assertEqual(convention["manual_positive"], "right steering")
        self.assertEqual(convention["usd_joint_axis"], "+Z")
        self.assertIn("left steering", convention["usd_positive"])
        multiplier = float(convention["manual_to_usd_joint_multiplier"])
        self.assertClose(multiplier, -1.0)
        self.assertClose(-30.0 * multiplier, 30.0)
        self.assertClose(30.0 * multiplier, -30.0)
        self.assertIn("derived_coordinate_conversion", convention["quality"])

    def test_smoke_sweep_includes_manual_pivot_and_every_step_aabb(self) -> None:
        multiplier = float(
            self.spec["romo_b"]["steering_command_convention"][
                "manual_to_usd_joint_multiplier"
            ]
        )
        pivot_targets = {
            joint_name: (
                float(self.spec["romo_b"]["pivot_steering_deg"][joint_name])
                * multiplier
            )
            for joint_name in STEERING_JOINT_NAMES
        }
        patterns = steering_target_patterns(pivot_targets)
        self.assertEqual(
            [name for name, _ in patterns],
            [
                "all +30 degrees",
                "all -30 degrees",
                "manual Pivot",
                "neutral",
            ],
        )
        self.assertEqual(patterns[2][1], pivot_targets)
        self.assertEqual(
            pivot_targets,
            {
                "front_left": -30.0,
                "front_right": 30.0,
                "rear_left": 30.0,
                "rear_right": -30.0,
            },
        )
        smoke_source = (
            SIMULATION_ROOT / "scripts" / "run_isaacsim_smoke.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "if wheel_chassis_aabb_samples != args.steps:",
            smoke_source,
        )

    def test_mass_allocation_preserves_manual_total(self) -> None:
        allocation = self.spec["simulation_proxy"]["mass_allocation_kg"]
        total = float(allocation["chassis"]) + 4.0 * (
            float(allocation["each_wheel"]) + float(allocation["each_knuckle"])
        )
        self.assertClose(total, 55.0)
        self.assertClose(total, float(allocation["aggregate"]))
        self.assertClose(total, float(self.spec["romo_b"]["mass_kg"]["platform_total"]["value"]))
        self.assertIn("not_measured", allocation["quality"])

    def test_manual_wheel_motor_speed_values_are_preserved(self) -> None:
        limits = self.spec["romo_b"]["limits"]
        rated = limits["wheel_motor_rated_speed_rpm"]
        maximum = limits["wheel_motor_max_speed_rpm"]
        self.assertClose(float(rated["value"]), 200.0)
        self.assertClose(float(maximum["value"]), 300.0)
        self.assertEqual(rated["manual_lines"], [330])
        self.assertEqual(maximum["manual_lines"], [331])
        self.assertIn("manual_stated", rated["quality"])
        self.assertIn("manual_stated", maximum["quality"])

    def test_mount_uses_user_front_offset_from_geometric_center(self) -> None:
        mount = self.spec["openarm_mount"]
        self.assertEqual(mount["position_m"], [0.17, 0.0, 0.342])
        self.assertClose(float(mount["position_m"][0]), float(mount["user_front_offset_m"]))
        length = float(self.spec["romo_b"]["dimensions_m"]["overall_length"]["value"])
        alternate = length / 2.0 - float(mount["user_front_offset_m"])
        self.assertClose(alternate, float(mount["alternate_x_if_17_cm_means_setback_from_front_edge_m"]))
        self.assertIn("user_supplied_approximate", mount["quality"])

    def test_openarm_start_pose_avoids_unfiltered_self_penetration(self) -> None:
        initial_state = self.spec["sources"]["openarm_asset"][
            "collision_safe_initial_joint_state"
        ]
        positions = {item["path"]: item for item in initial_state["positions"]}
        self.assertEqual(positions["joints/openarm_left_joint2"]["value"], -90.0)
        self.assertEqual(positions["joints/openarm_right_joint2"]["value"], 90.0)
        for side in ("left", "right"):
            reference = positions[f"joints/openarm_{side}_finger_joint1"]
            follower = positions[f"joints/openarm_{side}_finger_joint2"]
            self.assertClose(float(reference["value"]), 0.044)
            self.assertClose(float(follower["value"]), 0.044)
            self.assertTrue(reference["drive_target"])
            self.assertFalse(follower["drive_target"])
        self.assertIn("not_hardware_home", initial_state["quality"])
        self.assertIn("no OpenArm self-penetration", initial_state["validation"])

    def test_manual_and_proxy_values_are_explicitly_separated(self) -> None:
        proxy = self.spec["simulation_proxy"]
        self.assertIn("unverified", proxy["chassis_collision_width_m"]["quality"])
        self.assertIn("unverified", proxy["knuckle_body_size_m"]["quality"])
        self.assertIn("derived", proxy["wheel_width_m"]["quality"])
        self.assertIn("unverified", proxy["drive_gains"]["quality"])
        self.assertEqual(proxy["articulation_solver"]["position_iteration_count"], 32)
        self.assertEqual(proxy["articulation_solver"]["velocity_iteration_count"], 1)
        self.assertTrue(proxy["articulation_solver"]["enabled_self_collisions"])
        self.assertTrue(proxy["collision_policy"]["all_authored_colliders_enabled"])
        self.assertTrue(proxy["collision_policy"]["all_active_rigid_bodies_enabled"])
        self.assertFalse(proxy["collision_policy"]["allow_filtered_pairs"])
        self.assertIn("user_requested", proxy["collision_policy"]["quality"])
        self.assertIn("ignores per-joint collision settings", proxy["collision_policy"]["runtime_limitation"])
        self.assertGreaterEqual(len(proxy["omitted_unmeasured_properties"]), 6)
        self.assertEqual(self.spec["model_scope"], "manual-constrained simulation proxy")

    def test_usda_contains_mobile_articulation_and_openarm_attachment(self) -> None:
        required_tokens = [
            'defaultPrim = "World"',
            'metersPerUnit = 1',
            'upAxis = "Z"',
            'def Xform "ROMOBOpenArm"',
            'PhysicsArticulationRootAPI',
            'PhysxArticulationAPI',
            'def Mesh "SideShellTopPlate"',
            'def Mesh "LeftSidePanelWithWheelWells"',
            'def Mesh "RightSidePanelWithWheelWells"',
            'def Mesh "TopRim"',
            'def Mesh "RecessedTopPlate"',
            'custom int romoB:archesPerWheel = 1',
            'custom string romoB:wheelHouseShape = "single_circular_arch_per_wheel"',
            'custom bool romoB:visibleProfileUsesSteeringSweep = 0',
            'def Scope "Collision"',
            'def Cube "Core"',
            'def Cube "TopDeck"',
            'def Cube "FrontLower"',
            'def Cube "RearLower"',
            'def Scope "Visuals"',
            'def Cylinder "Tire"',
            'def Cylinder "Hub"',
            'def Cylinder "HubCap"',
            'def PhysicsRevoluteJoint "front_left_steering_joint"',
            'def PhysicsRevoluteJoint "rear_right_wheel_joint"',
            'def PhysicsFixedJoint "openarm_mount_joint"',
            'over "root_joint" (',
            'active = false',
            'bool physxArticulation:enabledSelfCollisions = 1',
            'int physxArticulation:solverPositionIterationCount = 32',
            'custom bool romoB:allAuthoredCollidersEnabled = 1',
            'custom bool romoB:allActiveRigidBodiesEnabled = 1',
            'custom bool romoB:filteredPairsAllowed = 0',
            'custom string romoB:collisionRuntimeLimitation = "Isaac Sim 5.1 PhysX ignores per-joint collision settings',
            'float state:angular:physics:position = -90',
            'float state:angular:physics:position = 90',
            'float state:linear:physics:position = 0.044',
            self.spec["sources"]["openarm_asset"]["url"],
        ]
        for token in required_tokens:
            self.assertIn(token, self.asset_text)
        self.assertNotIn("FenderArch", self.asset_text)
        self.assertNotIn("steeringSweepLimitDeg", self.asset_text)
        self.assertEqual(
            self.asset_text.count(
                'custom string romoB:wheelHouseShape = '
                '"single_circular_arch_per_wheel"'
            ),
            2,
        )
        self.assertNotIn("PhysicsFilteredPairsAPI", self.asset_text)
        self.assertNotIn("physics:filteredPairs", self.asset_text)
        self.assertEqual(
            self.asset_text.count('def Cylinder "TopHoleMarker_'),
            93,
        )
        self.assertIn("custom double romoB:topHolePitchM = 0.05", self.asset_text)
        self.assertIn("custom int romoB:visibleTopHoleCount = 93", self.asset_text)
        self.assertEqual(self.asset_text.count("bool physics:collisionEnabled = 1"), 32)
        self.assertEqual(self.asset_text.count("bool physics:rigidBodyEnabled = 1"), 32)


if __name__ == "__main__":
    unittest.main()
