"""Open a registered environment once and keep the GUI responsive without stepping physics."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_SOURCE = PROJECT_ROOT / "source" / "openarm_testusd_task"
sys.path.insert(0, str(TASK_SOURCE))

import openarm_testusd_task  # noqa: F401, E402
from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description="Static viewer for the original-TestUSD task.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Reach-OpenArm-TestUSD-Bi-v0",
    help="Name of the task.",
)
parser.add_argument("--fps", type=float, default=30.0, help="GUI update rate while the scene is paused.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    sleep_dt = 1.0 / args_cli.fps if args_cli.fps > 0 else 0.0
    try:
        while simulation_app.is_running():
            simulation_app.update()
            if sleep_dt > 0:
                time.sleep(sleep_dt)
    finally:
        env.close()


if __name__ == "__main__":
    main()
