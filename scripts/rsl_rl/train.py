"""Register the original-TestUSD task before running Isaac Lab RSL-RL training."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = Path("/home/mh/isaac-stack-2.3.2/IsaacLab-2.3.2")
TASK_SOURCE = PROJECT_ROOT / "source" / "openarm_testusd_task"
ISAAC_TRAIN_DIR = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"

sys.path.insert(0, str(TASK_SOURCE))
sys.path.insert(0, str(ISAAC_TRAIN_DIR))

import openarm_testusd_task  # noqa: F401, E402
from openarm_testusd_task.tasks.testusd_pick_place.agents.authority_masked_ppo import (  # noqa: E402
    register_openarm_rsl_rl_classes,
)

register_openarm_rsl_rl_classes()

try:
    from isaacsim.simulation_app import SimulationApp

    SimulationApp.close = lambda self, *args, **kwargs: None
except Exception:
    pass

runpy.run_path(str(ISAAC_TRAIN_DIR / "train.py"), run_name="__main__")
