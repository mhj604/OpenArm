from __future__ import annotations

import math
import os
from dataclasses import MISSING
from pathlib import Path

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_assets.robots.openarm import OPENARM_BI_HIGH_PD_CFG

from .original_usd_spawner import OriginalUsdSceneCfg, spawn_original_usd_scene


PROJECT_ROOT = Path(__file__).resolve().parents[5]
ORIGINAL_USD_PATH = os.environ.get("OPENARM_TEST_USD_PATH", str(PROJECT_ROOT / "assets" / "openarm_test.usd"))
ROBOT_ROOT_POS = (0.0, 0.0, 0.35)
ROBOT_ROOT_ROT = (1.0, 0.0, 0.0, 0.0)


def _openarm_robot_from_original_usd_cfg() -> ArticulationCfg:
    cfg = OPENARM_BI_HIGH_PD_CFG.copy()
    cfg.prim_path = "{ENV_REGEX_NS}/Scene/openarm_bimanual"
    cfg.spawn = None
    cfg.articulation_root_prim_path = "/root_joint"
    cfg.init_state.pos = ROBOT_ROOT_POS
    cfg.init_state.rot = ROBOT_ROOT_ROT
    cfg.init_state.joint_pos = {
        "openarm_left_joint.*": 0.0,
        "openarm_right_joint.*": 0.0,
        "openarm_left_finger_joint.*": 0.044,
        "openarm_right_finger_joint.*": 0.044,
    }
    cfg.init_state.joint_vel = {".*": 0.0}
    return cfg


@configclass
class OriginalTestUsdSceneCfg(InteractiveSceneCfg):
    """Scene based on the original openarm_test.usd."""

    original_usd: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=OriginalUsdSceneCfg(func=spawn_original_usd_scene, usd_path=ORIGINAL_USD_PATH),
    )

    robot: ArticulationCfg = MISSING


@configclass
class CommandsCfg:
    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.3),
            pos_y=(0.15, 0.25),
            pos_z=(0.3, 0.5),
            roll=(-math.pi / 6, math.pi / 6),
            pitch=(3 * math.pi / 2, 3 * math.pi / 2),
            yaw=(8 * math.pi / 9, 10 * math.pi / 9),
        ),
    )

    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.3),
            pos_y=(-0.25, -0.15),
            pos_z=(0.3, 0.5),
            roll=(-math.pi / 6, math.pi / 6),
            pitch=(3 * math.pi / 2, 3 * math.pi / 2),
            yaw=(8 * math.pi / 9, 10 * math.pi / 9),
        ),
    )


@configclass
class ActionsCfg:
    left_arm_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "left_ee_pose"})
        right_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "right_ee_pose"})
        left_actions = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        right_actions = ObsTerm(func=mdp.last_action, params={"action_name": "right_arm_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RewardsCfg:
    left_end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["openarm_left_hand"]), "command_name": "left_ee_pose"},
    )
    right_end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["openarm_right_hand"]), "command_name": "right_ee_pose"},
    )
    left_end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["openarm_left_hand"]),
            "std": 0.1,
            "command_name": "left_ee_pose",
        },
    )
    right_end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
            "std": 0.1,
            "command_name": "right_ee_pose",
        },
    )
    left_end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["openarm_left_hand"]), "command_name": "left_ee_pose"},
    )
    right_end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["openarm_right_hand"]), "command_name": "right_ee_pose"},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint.*"])},
    )
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint.*"])},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500},
    )
    left_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_joint_vel", "weight": -0.001, "num_steps": 4500},
    )
    right_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_joint_vel", "weight": -0.001, "num_steps": 4500},
    )


@configclass
class OpenArmTestUsdReachEnvCfg(ManagerBasedRLEnvCfg):
    scene: OriginalTestUsdSceneCfg = OriginalTestUsdSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.scene.robot = _openarm_robot_from_original_usd_cfg()
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.commands.left_ee_pose.body_name = "openarm_left_hand"
        self.commands.right_ee_pose.body_name = "openarm_right_hand"
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 24.0
        self.viewer.eye = (1.7028798, 0.0835766, 0.9590004)
        self.viewer.lookat = (-0.7228655, -0.0446134, 0.6415680)
        self.sim.dt = 1.0 / 60.0


@configclass
class OpenArmTestUsdReachEnvCfg_PLAY(OpenArmTestUsdReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
