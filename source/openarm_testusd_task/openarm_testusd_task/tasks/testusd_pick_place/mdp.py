from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.manipulation.lift.mdp import *  # noqa: F401, F403

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_body_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Reward proximity between the object and a robot body using a tanh kernel."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_id = robot_body_cfg.body_ids[0]
    body_pos_w = robot.data.body_pos_w[:, body_id, :]
    distance = torch.norm(obj.data.root_pos_w - body_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / std)

