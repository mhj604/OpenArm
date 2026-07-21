#!/usr/bin/env python3
"""Load and step the ROMO-B/OpenArm stage in Isaac Sim 5.1.

This is an integration smoke test, not a calibrated dynamics experiment. The
NVIDIA OpenArm payload must be reachable through the Isaac asset cache or the
network.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


SIMULATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = SIMULATION_ROOT / "assets" / "romo_b_openarm_manual_proxy.usda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--gui", action="store_true", help="Show the Isaac Sim UI instead of running headless.")
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

    # Omniverse and pxr imports must occur after SimulationApp is instantiated.
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": not args.gui,
            "create_new_stage": False,
            "sync_loads": True,
            "anti_aliasing": 0,
            "disable_viewport_updates": not args.gui,
            "fast_shutdown": True,
        }
    )
    try:
        import omni.timeline
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics

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

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(args.steps):
            app.update()
        timeline.stop()
        app.update()

        chassis = stage.GetPrimAtPath("/World/ROMOBOpenArm/Chassis")
        matrix = UsdGeom.Xformable(chassis).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        xyz = matrix.ExtractTranslation()
        if not all(math.isfinite(float(xyz[index])) for index in range(3)):
            raise RuntimeError(f"Non-finite chassis pose after stepping: {xyz}")

        print("Isaac Sim smoke test: PASS", flush=True)
        print(f"asset: {asset}", flush=True)
        print(f"steps: {args.steps}", flush=True)
        print(
            f"chassis translation: ({float(xyz[0]):.6f}, {float(xyz[1]):.6f}, {float(xyz[2]):.6f})",
            flush=True,
        )
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
