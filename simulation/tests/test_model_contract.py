#!/usr/bin/env python3
"""Pure-Python contract tests for the audited ROMO-B/OpenArm model."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path


SIMULATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SIMULATION_ROOT.parent
SPEC_PATH = SIMULATION_ROOT / "config" / "romo_b_openarm_spec.json"
ASSET_PATH = SIMULATION_ROOT / "assets" / "romo_b_openarm_manual_proxy.usda"
MANUAL_PATH = REPOSITORY_ROOT / "source_files" / "hardware" / "ROMO-B_manual_verified_complete.md"


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
        self.assertClose(float(dims["wheelbase"]["conflicting_value"]), 0.324)

    def test_geometry_derivations(self) -> None:
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

    def test_manual_and_proxy_values_are_explicitly_separated(self) -> None:
        proxy = self.spec["simulation_proxy"]
        self.assertIn("unverified", proxy["chassis_collision_width_m"]["quality"])
        self.assertIn("unverified", proxy["knuckle_body_size_m"]["quality"])
        self.assertIn("derived", proxy["wheel_width_m"]["quality"])
        self.assertIn("unverified", proxy["drive_gains"]["quality"])
        self.assertEqual(proxy["articulation_solver"]["position_iteration_count"], 32)
        self.assertEqual(proxy["articulation_solver"]["velocity_iteration_count"], 1)
        self.assertFalse(proxy["articulation_solver"]["enabled_self_collisions"])
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
            'def PhysicsRevoluteJoint "front_left_steering_joint"',
            'def PhysicsRevoluteJoint "rear_right_wheel_joint"',
            'def PhysicsFixedJoint "openarm_mount_joint"',
            'over "root_joint" (',
            'active = false',
            'int physxArticulation:solverPositionIterationCount = 32',
            self.spec["sources"]["openarm_asset"]["url"],
        ]
        for token in required_tokens:
            self.assertIn(token, self.asset_text)


if __name__ == "__main__":
    unittest.main()
