from __future__ import annotations

import os
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_assets.robots.openarm import OPENARM_BI_HIGH_PD_CFG

from . import mdp
from ..testusd_reach.original_usd_spawner import OriginalUsdSceneCfg, spawn_original_usd_scene


PROJECT_ROOT = Path(__file__).resolve().parents[5]
ORIGINAL_USD_PATH = os.environ.get("OPENARM_TEST_USD_PATH", str(PROJECT_ROOT / "assets" / "openarm_test.usd"))
ROBOT_ROOT_POS = (0.0, 0.0, 0.35)
ROBOT_ROOT_ROT = (1.0, 0.0, 0.0, 0.0)
EPISODE_LENGTH_S = 12.0

# Workspace coordinates are expressed in the robot root frame unless noted otherwise.
OBJECT_START_POS_W = (0.40, -0.10, 0.625)
BASKET_GOAL_POS_B = (0.52, 0.32, 0.42)
MIN_LIFT_HEIGHT_W = 0.70
EARLY_LIFT_HEIGHT_W = 0.66
GOAL_BONUS_COARSE_THRESHOLD = 0.30
GOAL_BONUS_MID_THRESHOLD = 0.20
GOAL_BONUS_TIGHT_THRESHOLD = 0.12
GOAL_HEIGHT_XY_THRESHOLD = 0.25
GRASP_GATE_STD = 0.08
GOAL_AXIS_STD = 0.12
PRE_GRASP_OFFSET_W = (0.0, 0.0, 0.03)
GRASP_OFFSET_W = (0.0, 0.0, 0.0)
RIGHT_HAND_MIN_CLEARANCE_W = 0.56
RIGHT_HAND_BACKWARD_LIMIT_X_B = 0.02
RIGHT_HAND_TARGET_FORWARD_X_B = 0.40
RIGHT_HAND_TARGET_HEIGHT_B = 0.22
GRIPPER_OPEN_POS = 0.044
LEFT_ARM_JOINT_NAMES = ["openarm_left_joint.*"]
LEFT_GRIPPER_JOINT_NAMES = ["openarm_left_finger_joint.*"]
LEFT_HOLD_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_GRIPPER_JOINT_NAMES
RIGHT_ARM_JOINT_NAMES = ["openarm_right_joint.*"]
RIGHT_GRIPPER_JOINT_NAMES = ["openarm_right_finger_joint.*"]
RIGHT_POLICY_JOINT_NAMES = RIGHT_ARM_JOINT_NAMES + RIGHT_GRIPPER_JOINT_NAMES
RIGHT_HAND_BODY_NAME = "openarm_right_hand"
RIGHT_EE_BODY_NAME = "openarm_right_ee_tcp"
RIGHT_FINGER_BODY_NAMES = ["openarm_right_left_finger", "openarm_right_right_finger"]
RIGHT_READY_JOINT_POS = {
    "openarm_right_joint1": 0.84664,
    "openarm_right_joint2": -0.17450,
    "openarm_right_joint3": -0.57209,
    "openarm_right_joint4": 0.00010,
    "openarm_right_joint5": 1.04863,
    "openarm_right_joint6": -0.45328,
    "openarm_right_joint7": -0.09522,
}


def _openarm_robot_from_original_usd_cfg() -> ArticulationCfg:
    cfg = OPENARM_BI_HIGH_PD_CFG.copy()
    cfg.prim_path = "{ENV_REGEX_NS}/Scene/openarm_bimanual"
    cfg.spawn = None
    cfg.articulation_root_prim_path = "/root_joint"
    cfg.init_state.pos = ROBOT_ROOT_POS
    cfg.init_state.rot = ROBOT_ROOT_ROT
    cfg.init_state.joint_pos = {
        "openarm_left_joint.*": 0.0,
        "openarm_left_finger_joint.*": 0.044,
        "openarm_right_finger_joint.*": 0.044,
        **RIGHT_READY_JOINT_POS,
    }
    cfg.init_state.joint_vel = {".*": 0.0}
    return cfg


def _train_object_cfg() -> RigidObjectCfg:
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_START_POS_W, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=rigid_props,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
        ),
    )


@configclass
class OpenArmTestUsdPickPlaceSceneCfg(InteractiveSceneCfg):
    """openarm_test.usd scene plus one trainable object."""

    original_usd: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=OriginalUsdSceneCfg(func=spawn_original_usd_scene, usd_path=ORIGINAL_USD_PATH),
    )

    robot: ArticulationCfg = MISSING
    object: RigidObjectCfg = MISSING


@configclass
class CommandsCfg:
    object_pose = mdp.StagedObjectPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        object_name="object",
        resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S),
        start_offset_from_object=(0.04, 0.05, 0.14),
        curriculum_num_steps=96_000,
        curriculum_power=1.5,
        debug_vis=True,
        ranges=mdp.StagedObjectPoseCommandCfg.Ranges(
            pos_x=(BASKET_GOAL_POS_B[0] - 0.03, BASKET_GOAL_POS_B[0] + 0.03),
            pos_y=(BASKET_GOAL_POS_B[1] - 0.03, BASKET_GOAL_POS_B[1] + 0.03),
            pos_z=(BASKET_GOAL_POS_B[2] - 0.02, BASKET_GOAL_POS_B[2] + 0.02),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=RIGHT_POLICY_JOINT_NAMES,
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=RIGHT_POLICY_JOINT_NAMES,
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        right_hand_position = ObsTerm(
            func=mdp.body_position_in_robot_root_frame,
            params={"robot_body_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY_NAME])},
        )
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        object_to_goal_position = ObsTerm(
            func=mdp.object_to_goal_position_in_robot_root_frame,
            params={"command_name": "object_pose"},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    hold_left_gripper_open = EventTerm(
        func=mdp.hold_joint_position_target,
        mode="interval",
        interval_range_s=(1.0 / 30.0, 1.0 / 30.0),
        params={
            "joint_pos": GRIPPER_OPEN_POS,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_GRIPPER_JOINT_NAMES),
        },
    )

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.015, 0.015),
                "y": (-0.015, 0.015),
                "z": (0.0, 0.0),
                "yaw": (-0.35, 0.35),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    reaching_pre_grasp = RewTerm(
        func=mdp.object_body_offset_distance,
        params={
            "std": 0.50,
            "object_offset": PRE_GRASP_OFFSET_W,
            "robot_body_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY_NAME]),
        },
        weight=2.0,
    )

    right_hand_forward_alignment = RewTerm(
        func=mdp.object_body_axis_distance,
        params={
            "axis": 0,
            "std": 0.35,
            "object_offset": -0.02,
            "robot_body_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY_NAME]),
        },
        weight=3.0,
    )

    right_hand_forward_progress = RewTerm(
        func=mdp.body_axis_above_minimum_in_robot_root_frame,
        params={
            "axis": 0,
            "minimum_value": RIGHT_HAND_TARGET_FORWARD_X_B,
            "std": 0.20,
            "robot_body_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY_NAME]),
        },
        weight=4.0,
    )

    right_hand_lift_to_pre_grasp = RewTerm(
        func=mdp.body_axis_above_minimum_in_robot_root_frame,
        params={
            "axis": 2,
            "minimum_value": RIGHT_HAND_TARGET_HEIGHT_B,
            "std": 0.12,
            "robot_body_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY_NAME]),
        },
        weight=0.5,
    )

    reaching_object = RewTerm(
        func=mdp.object_body_offset_distance,
        params={
            "std": 0.35,
            "object_offset": GRASP_OFFSET_W,
            "robot_body_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY_NAME]),
        },
        weight=4.0,
    )

    reaching_object_fine = RewTerm(
        func=mdp.object_body_offset_distance,
        params={
            "std": 0.07,
            "object_offset": GRASP_OFFSET_W,
            "robot_body_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY_NAME]),
        },
        weight=18.0,
    )

    right_fingers_near_object = RewTerm(
        func=mdp.object_bodies_average_distance,
        params={
            "std": 0.07,
            "object_offset": GRASP_OFFSET_W,
            "robot_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
        },
        weight=20.0,
    )

    gripper_closed_near_object = RewTerm(
        func=mdp.gripper_closed_near_object_bodies,
        params={
            "std": 0.07,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "robot_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=20.0,
    )

    closed_gripper_lift_target = RewTerm(
        func=mdp.closed_gripper_lift_target,
        params={
            "std": 0.08,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "lift_offset": 0.08,
            "object_offset": GRASP_OFFSET_W,
            "robot_body_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY_NAME]),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=10.0,
    )

    lifting_object_early = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": EARLY_LIFT_HEIGHT_W},
        weight=24.0,
    )

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": MIN_LIFT_HEIGHT_W}, weight=35.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance_with_grasp,
        params={
            "std": 0.30,
            "minimal_height": MIN_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=18.0,
    )

    object_goal_xy_tracking = RewTerm(
        func=mdp.object_goal_xy_distance,
        params={
            "std": 0.18,
            "minimal_height": MIN_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=12.0,
    )

    object_goal_x_tracking = RewTerm(
        func=mdp.object_goal_axis_distance_with_grasp,
        params={
            "axis": 0,
            "std": GOAL_AXIS_STD,
            "minimal_height": MIN_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=8.0,
    )

    object_goal_y_tracking = RewTerm(
        func=mdp.object_goal_axis_distance_with_grasp,
        params={
            "axis": 1,
            "std": GOAL_AXIS_STD,
            "minimal_height": MIN_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=18.0,
    )

    object_goal_z_tracking = RewTerm(
        func=mdp.object_goal_axis_distance_with_grasp,
        params={
            "axis": 2,
            "std": 0.08,
            "minimal_height": MIN_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=8.0,
    )

    object_goal_height_tracking = RewTerm(
        func=mdp.object_goal_height_tracking,
        params={
            "std": 0.08,
            "xy_threshold": GOAL_HEIGHT_XY_THRESHOLD,
            "minimal_height": MIN_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=8.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance_with_grasp,
        params={
            "std": 0.08,
            "minimal_height": MIN_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=12.0,
    )

    object_goal_bonus_coarse = RewTerm(
        func=mdp.object_goal_bonus,
        params={
            "threshold": GOAL_BONUS_COARSE_THRESHOLD,
            "minimal_height": MIN_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=8.0,
    )

    object_goal_bonus_mid = RewTerm(
        func=mdp.object_goal_bonus,
        params={
            "threshold": GOAL_BONUS_MID_THRESHOLD,
            "minimal_height": MIN_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=16.0,
    )

    object_goal_bonus_tight = RewTerm(
        func=mdp.object_goal_bonus,
        params={
            "threshold": GOAL_BONUS_TIGHT_THRESHOLD,
            "minimal_height": MIN_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=28.0,
    )

    object_goal_stable_hold = RewTerm(
        func=mdp.object_goal_stable_hold,
        params={
            "threshold": GOAL_BONUS_MID_THRESHOLD,
            "minimal_height": MIN_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "velocity_std": 0.05,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=24.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    right_hand_table_clearance = RewTerm(
        func=mdp.body_height_below_minimum,
        params={
            "minimum_height": RIGHT_HAND_MIN_CLEARANCE_W,
            "robot_body_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY_NAME]),
        },
        weight=-2.0,
    )

    right_hand_backward_guard = RewTerm(
        func=mdp.body_axis_below_minimum_in_robot_root_frame,
        params={
            "axis": 0,
            "minimum_value": RIGHT_HAND_BACKWARD_LIMIT_X_B,
            "robot_body_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY_NAME]),
        },
        weight=-1.0,
    )

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=RIGHT_POLICY_JOINT_NAMES,
            )
        },
    )

    left_joint_pos_hold = RewTerm(
        func=mdp.joint_position_deviation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_HOLD_JOINT_NAMES)},
    )

    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_HOLD_JOINT_NAMES)},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.45, "asset_cfg": SceneEntityCfg("object")},
    )

    object_reached_goal = None


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.001, "num_steps": 120000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 120000},
    )


@configclass
class OpenArmTestUsdPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: OpenArmTestUsdPickPlaceSceneCfg = OpenArmTestUsdPickPlaceSceneCfg(
        num_envs=1024, env_spacing=2.5, replicate_physics=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.scene.robot = _openarm_robot_from_original_usd_cfg()
        self.scene.object = _train_object_cfg()

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_ARM_JOINT_NAMES,
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_GRIPPER_JOINT_NAMES,
            open_command_expr={"openarm_right_finger_joint.*": 0.044},
            close_command_expr={"openarm_right_finger_joint.*": 0.0},
        )

        self.commands.object_pose.body_name = RIGHT_HAND_BODY_NAME

        self.decimation = 2
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.viewer.eye = (1.6, -1.2, 1.2)
        self.viewer.lookat = (0.45, 0.0, 0.65)


@configclass
class OpenArmTestUsdPickPlaceEnvCfg_PLAY(OpenArmTestUsdPickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
