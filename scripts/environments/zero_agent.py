"""Run the original-TestUSD task with zero actions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_SOURCE = PROJECT_ROOT / "source" / "openarm_testusd_task"
sys.path.insert(0, str(TASK_SOURCE))

import openarm_testusd_task  # noqa: F401, E402
from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description="Zero-action viewer for the original-TestUSD task.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Reach-OpenArm-TestUSD-Bi-v0",
    help="Name of the task.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    env.reset()
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)
    finally:
        env.close()


if __name__ == "__main__":
    main()
