"""Register the original-TestUSD task before running Isaac Lab RSL-RL play."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = Path("/home/mh/isaac-stack-2.3.2/IsaacLab-2.3.2")
TASK_SOURCE = PROJECT_ROOT / "source" / "openarm_testusd_task"
ISAAC_PLAY_DIR = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"

sys.path.insert(0, str(TASK_SOURCE))
sys.path.insert(0, str(ISAAC_PLAY_DIR))

import openarm_testusd_task  # noqa: F401, E402

try:
    from isaacsim.simulation_app import SimulationApp

    SimulationApp.close = lambda self, *args, **kwargs: None
except Exception:
    pass

runpy.run_path(str(ISAAC_PLAY_DIR / "play.py"), run_name="__main__")
