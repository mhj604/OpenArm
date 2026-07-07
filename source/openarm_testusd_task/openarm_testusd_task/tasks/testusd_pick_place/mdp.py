from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

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


def object_body_offset_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Reward proximity between a robot body and an offset point around the object."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_id = robot_body_cfg.body_ids[0]
    body_pos_w = robot.data.body_pos_w[:, body_id, :]
    offset = torch.tensor(object_offset, device=obj.data.root_pos_w.device, dtype=obj.data.root_pos_w.dtype)
    target_pos_w = obj.data.root_pos_w + offset
    distance = torch.norm(target_pos_w - body_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def gripper_closed_near_object(
    env: ManagerBasedRLEnv,
    std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward closing the gripper only when the right hand is close to the object."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_id = robot_body_cfg.body_ids[0]
    body_pos_w = robot.data.body_pos_w[:, body_id, :]
    offset = torch.tensor(object_offset, device=obj.data.root_pos_w.device, dtype=obj.data.root_pos_w.dtype)
    target_pos_w = obj.data.root_pos_w + offset
    distance = torch.norm(target_pos_w - body_pos_w, dim=1)
    proximity = 1.0 - torch.tanh(distance / std)

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    closed = torch.clamp((open_joint_pos - gripper_pos) / open_joint_pos, min=0.0, max=1.0)
    return proximity * torch.mean(closed, dim=1)


def gripper_closed_near_object_bodies(
    env: ManagerBasedRLEnv,
    std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward closing the gripper when the selected grasp bodies are close to the object."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, robot_body_cfg.body_ids, :]
    offset = torch.tensor(object_offset, device=obj.data.root_pos_w.device, dtype=obj.data.root_pos_w.dtype)
    target_pos_w = obj.data.root_pos_w[:, None, :] + offset
    distance = torch.norm(target_pos_w - body_pos_w, dim=-1)
    proximity = torch.mean(1.0 - torch.tanh(distance / std), dim=1)

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    closed = torch.clamp((open_joint_pos - gripper_pos) / open_joint_pos, min=0.0, max=1.0)
    return proximity * torch.mean(closed, dim=1)


def closed_gripper_lift_target(
    env: ManagerBasedRLEnv,
    std: float,
    open_joint_pos: float,
    lift_offset: float,
    object_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_ee_tcp"]),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward pulling upward after the gripper is closed near the object."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, robot_body_cfg.body_ids, :]
    offset = torch.tensor(object_offset, device=obj.data.root_pos_w.device, dtype=obj.data.root_pos_w.dtype)
    target_pos_w = obj.data.root_pos_w[:, None, :] + offset
    distance = torch.norm(target_pos_w - body_pos_w, dim=-1)
    proximity = torch.mean(1.0 - torch.tanh(distance / std), dim=1)

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    closed = torch.clamp((open_joint_pos - gripper_pos) / open_joint_pos, min=0.0, max=1.0)
    closed_mean = torch.mean(closed, dim=1)

    body_height = torch.mean(body_pos_w[:, :, 2], dim=1)
    target_height = obj.data.root_pos_w[:, 2] + lift_offset
    height_error = torch.clamp(target_height - body_height, min=0.0)
    height_reward = 1.0 - torch.tanh(height_error / std)
    return proximity * closed_mean * height_reward


def object_bodies_average_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
) -> torch.Tensor:
    """Reward multiple robot bodies for approaching one object-relative target."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, robot_body_cfg.body_ids, :]
    offset = torch.tensor(object_offset, device=obj.data.root_pos_w.device, dtype=obj.data.root_pos_w.dtype)
    target_pos_w = obj.data.root_pos_w[:, None, :] + offset
    distance = torch.norm(target_pos_w - body_pos_w, dim=-1)
    return torch.mean(1.0 - torch.tanh(distance / std), dim=1)


def _closed_grasp_gate(
    env: ManagerBasedRLEnv,
    std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg,
    grasp_body_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[grasp_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, grasp_body_cfg.body_ids, :]
    offset = torch.tensor(object_offset, device=obj.data.root_pos_w.device, dtype=obj.data.root_pos_w.dtype)
    target_pos_w = obj.data.root_pos_w[:, None, :] + offset
    distance = torch.norm(target_pos_w - body_pos_w, dim=-1)
    proximity = torch.mean(1.0 - torch.tanh(distance / std), dim=1)

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    closed = torch.clamp((open_joint_pos - gripper_pos) / open_joint_pos, min=0.0, max=1.0)
    return proximity * torch.mean(closed, dim=1)


def object_goal_distance_with_grasp(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward object-goal distance only while the object is in a stable closed grasp."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance = torch.norm(goal_pos_w - obj.data.root_pos_w, dim=1)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return lifted.float() * grasp_gate * (1.0 - torch.tanh(distance / std))


def object_to_goal_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return the object-to-goal translation vector in the robot root frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w)
    return command[:, :3] - object_pos_b


def object_goal_axis_distance_with_grasp(
    env: ManagerBasedRLEnv,
    axis: int,
    std: float,
    minimal_height: float,
    command_name: str,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward one coordinate of a grasped object independently from the full 3D goal distance."""

    obj: RigidObject = env.scene[object_cfg.name]
    delta_b = object_to_goal_position_in_robot_root_frame(env, command_name, robot_cfg, object_cfg)
    axis_error = torch.abs(delta_b[:, axis])
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return lifted.float() * grasp_gate * (1.0 - torch.tanh(axis_error / std))


def object_goal_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    minimal_height: float,
    command_name: str,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward a grasped object smoothly inside a loose goal basin while it is lifted."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance = torch.norm(goal_pos_w - obj.data.root_pos_w, dim=1)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    basin_reward = torch.clamp((threshold - distance) / threshold, min=0.0, max=1.0)
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return lifted.float() * grasp_gate * basin_reward


def object_goal_xy_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward horizontal object-goal alignment after the object is lifted and grasped."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance_xy = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return lifted.float() * grasp_gate * (1.0 - torch.tanh(distance_xy / std))


def object_goal_height_tracking(
    env: ManagerBasedRLEnv,
    std: float,
    xy_threshold: float,
    minimal_height: float,
    command_name: str,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward matching goal height once the grasped object is horizontally near the goal."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance_xy = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    near_xy = torch.clamp((xy_threshold - distance_xy) / xy_threshold, min=0.0, max=1.0)
    height_error = torch.abs(goal_pos_w[:, 2] - obj.data.root_pos_w[:, 2])
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return lifted.float() * grasp_gate * near_xy * (1.0 - torch.tanh(height_error / std))


def body_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Return a robot body position expressed in the robot root frame."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    body_id = robot_body_cfg.body_ids[0]
    body_pos_w = robot.data.body_pos_w[:, body_id, :]
    body_quat_w = robot.data.body_quat_w[:, body_id, :]
    body_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, body_pos_w, body_quat_w)
    return body_pos_b


def object_body_axis_distance(
    env: ManagerBasedRLEnv,
    axis: int,
    std: float,
    object_offset: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Reward matching one world-axis coordinate between an object offset point and a robot body."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_id = robot_body_cfg.body_ids[0]
    target = obj.data.root_pos_w[:, axis] + object_offset
    error = torch.abs(target - robot.data.body_pos_w[:, body_id, axis])
    return 1.0 - torch.tanh(error / std)


def body_axis_above_minimum(
    env: ManagerBasedRLEnv,
    axis: int,
    minimum_value: float,
    std: float,
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Reward a body for staying above a minimum coordinate on a world axis."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    body_pos = robot.data.body_pos_w[:, robot_body_cfg.body_ids, axis]
    if body_pos.ndim > 1:
        body_pos = torch.min(body_pos, dim=1).values
    violation = torch.clamp(minimum_value - body_pos, min=0.0)
    return 1.0 - torch.tanh(violation / std)


def body_axis_above_minimum_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    axis: int,
    minimum_value: float,
    std: float,
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Reward a body for staying above a robot-root-frame coordinate."""

    body_pos_b = body_position_in_robot_root_frame(env, robot_body_cfg)
    violation = torch.clamp(minimum_value - body_pos_b[:, axis], min=0.0)
    return 1.0 - torch.tanh(violation / std)


def body_axis_below_minimum(
    env: ManagerBasedRLEnv,
    axis: int,
    minimum_value: float,
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Return the body coordinate violation below a minimum world-axis value."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    body_pos = robot.data.body_pos_w[:, robot_body_cfg.body_ids, axis]
    if body_pos.ndim > 1:
        body_pos = torch.min(body_pos, dim=1).values
    return torch.clamp(minimum_value - body_pos, min=0.0)


def body_axis_below_minimum_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    axis: int,
    minimum_value: float,
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Return violation below a minimum robot-root-frame coordinate."""

    body_pos_b = body_position_in_robot_root_frame(env, robot_body_cfg)
    return torch.clamp(minimum_value - body_pos_b[:, axis], min=0.0)


def body_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Return the body height violation in meters for clearance penalties."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    body_pos_z = robot.data.body_pos_w[:, robot_body_cfg.body_ids, 2]
    if body_pos_z.ndim > 1:
        body_pos_z = torch.min(body_pos_z, dim=1).values
    return torch.clamp(minimum_height - body_pos_z, min=0.0)


def joint_position_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_left_joint.*"]),
) -> torch.Tensor:
    """Return squared joint-position deviation from the reset/default pose."""

    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_pos - default_joint_pos), dim=1)
