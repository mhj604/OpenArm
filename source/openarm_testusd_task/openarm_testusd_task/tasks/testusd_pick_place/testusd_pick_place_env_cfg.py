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
from ..testusd_reach.original_usd_spawner import (
    OPENARM_ARTICULATION_ROOT_PRIM_PATH,
    OPENARM_ROBOT_REL_PATH,
    OriginalUsdSceneCfg,
    apply_openarm_self_collision_physx_profile,
    assert_openarm_self_collision_contract,
    spawn_original_usd_scene,
)


PROJECT_ROOT = Path(__file__).resolve().parents[5]
ORIGINAL_USD_PATH = os.environ.get("OPENARM_TEST_USD_PATH", str(PROJECT_ROOT / "assets" / "openarm_test.usd"))
ROBOT_ROOT_POS = (0.0, 0.0, 0.35)
ROBOT_ROOT_ROT = (1.0, 0.0, 0.0, 0.0)
EPISODE_LENGTH_S = 16.0

# Workspace coordinates are expressed in the robot root frame unless noted otherwise.
# Keep enough finger clearance from the basket wall on the cube-facing side.
OBJECT_START_POS_W = (0.40, -0.15, 0.625)
OBJECT_START_POS_B = (
    OBJECT_START_POS_W[0] - ROBOT_ROOT_POS[0],
    OBJECT_START_POS_W[1] - ROBOT_ROOT_POS[1],
    OBJECT_START_POS_W[2] - ROBOT_ROOT_POS[2],
)
BASKET_CENTER_POS_B = (0.48, 0.17, 0.30)
# The packing-table parent maps -0.05 local X to -0.03 world Y.
BASKET_PRIM_TRANSLATION = (0.2833333333333334, -0.28, 0.9925656570552239)
# Reachable point inside the basket for the right arm at the final place height.
BASKET_SETTLE_POS_B = (0.37, 0.07, 0.30)
# Place the cube shallowly inside the basket before release.
BASKET_GOAL_POS_B = (BASKET_SETTLE_POS_B[0], BASKET_SETTLE_POS_B[1], 0.320)
LIFT_PROGRESS_START_HEIGHT_W = OBJECT_START_POS_W[2]
EARLY_LIFT_HEIGHT_W = 0.645
TRANSPORT_LIFT_HEIGHT_W = 0.64
MIN_LIFT_HEIGHT_W = 0.68
CARRY_LIFT_HEIGHT_W = 0.72
STAGE_CARRY_MIN_HEIGHT_W = 0.73
STAGE_CARRY_TARGET_HEIGHT_W = 0.75
STAGE_CARRY_TARGET_HEIGHT_B = STAGE_CARRY_TARGET_HEIGHT_W - ROBOT_ROOT_POS[2]
STAGE_CARRY_MAX_HEIGHT_W = 0.79
LOW_CLEARANCE_PENALTY_START_HEIGHT_W = 0.710
LOW_CLEARANCE_PENALTY_FULL_HEIGHT_W = STAGE_CARRY_MIN_HEIGHT_W
LOW_CLEARANCE_BASKET_APPROACH_XY = 0.17
# Cover the 52 mm pre-rim gap with 10 mm free motion and 8 mm overlap.
PRELIFT_INWARD_FREE_TOLERANCE = 0.010
PRELIFT_INWARD_MAX_CHARGED_EXCURSION = 0.050
STAGE_RELEASE_TARGET_HEIGHT_W = 0.670
STAGE_ASSIST_ENTRY_XY_THRESHOLD = 0.14
STAGE_ASSIST_OUTER_XY_THRESHOLD = 0.24
HAND_LIFT_START_HEIGHT_W = OBJECT_START_POS_W[2] - 0.12
HAND_LIFT_TARGET_HEIGHT_W = CARRY_LIFT_HEIGHT_W + 0.02
GOAL_BONUS_COARSE_THRESHOLD = 0.36
GOAL_BONUS_MID_THRESHOLD = 0.28
GOAL_BONUS_TIGHT_THRESHOLD = 0.20
GOAL_RELEASE_THRESHOLD = 0.11
GOAL_RELEASE_XY_THRESHOLD = 0.16
GOAL_RELEASE_Z_THRESHOLD = 0.08
GOAL_RELEASE_MIN_HEIGHT_W = 0.65
GOAL_RELEASE_VELOCITY_STD = 0.12
GOAL_RELEASE_MIN_PROGRESS = 0.95
GOAL_SETTLE_XY_THRESHOLD = GOAL_RELEASE_XY_THRESHOLD
GOAL_SETTLE_Z_THRESHOLD = 0.10
GOAL_SETTLE_MIN_PROGRESS = 0.95
GOAL_SUCCESS_XY_THRESHOLD = GOAL_RELEASE_XY_THRESHOLD
GOAL_SUCCESS_Z_THRESHOLD = GOAL_SETTLE_Z_THRESHOLD
GOAL_SUCCESS_MAX_SPEED = 0.12
GOAL_SUCCESS_MIN_PROGRESS = 0.98
GOAL_HEIGHT_XY_THRESHOLD = 0.28
GOAL_XY_BONUS_WIDE_THRESHOLD = 0.26
GOAL_XY_BONUS_MID_THRESHOLD = 0.18
GOAL_XY_BONUS_TIGHT_THRESHOLD = 0.10
GOAL_XY_SHORTFALL_THRESHOLD = 0.16
GOAL_XY_SHORTFALL_MAX_DISTANCE = 0.36
GRASP_GATE_STD = 0.14
GOAL_AXIS_STD = 0.12
GOAL_AXIS_BROAD_STD = 0.30
PRE_GRASP_OFFSET_W = (0.0, 0.0, 0.03)
GRASP_OFFSET_W = (0.0, 0.0, 0.0)
RIGHT_HAND_MIN_CLEARANCE_W = 0.61
RIGHT_HAND_BACKWARD_LIMIT_X_B = 0.02
RIGHT_HAND_TARGET_FORWARD_X_B = 0.40
RIGHT_HAND_TARGET_HEIGHT_B = 0.30
RIGHT_HAND_CARRY_TARGET_Y_B = BASKET_GOAL_POS_B[1]
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
    "openarm_right_joint1": 0.710210,
    "openarm_right_joint2": 0.028596,
    "openarm_right_joint3": -0.575045,
    "openarm_right_joint4": 0.328137,
    "openarm_right_joint5": 0.987903,
    "openarm_right_joint6": -0.490837,
    "openarm_right_joint7": -0.247898,
}


def _openarm_robot_from_original_usd_cfg() -> ArticulationCfg:
    cfg = OPENARM_BI_HIGH_PD_CFG.copy()
    cfg.prim_path = f"{{ENV_REGEX_NS}}/Scene/{OPENARM_ROBOT_REL_PATH}"
    cfg.spawn = None
    cfg.articulation_root_prim_path = OPENARM_ARTICULATION_ROOT_PRIM_PATH
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
        spawn=OriginalUsdSceneCfg(
            func=spawn_original_usd_scene,
            usd_path=ORIGINAL_USD_PATH,
            translation_overrides={"packing_table/container_h20": BASKET_PRIM_TRANSLATION},
            disable_gravity_paths=(OPENARM_ROBOT_REL_PATH,),
        ),
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
        start_offset_from_object=(0.04, 0.12, 0.18),
        curriculum_num_steps=12_000,
        curriculum_power=0.8,
        low_clearance_height=LOW_CLEARANCE_PENALTY_FULL_HEIGHT_W,
        near_basket_xy_threshold=LOW_CLEARANCE_BASKET_APPROACH_XY,
        debug_vis=True,
        ranges=mdp.StagedObjectPoseCommandCfg.Ranges(
            pos_x=(BASKET_GOAL_POS_B[0] - 0.01, BASKET_GOAL_POS_B[0] + 0.01),
            pos_y=(BASKET_GOAL_POS_B[1] - 0.01, BASKET_GOAL_POS_B[1] + 0.01),
            pos_z=(BASKET_GOAL_POS_B[2] - 0.01, BASKET_GOAL_POS_B[2] + 0.01),
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
    verify_self_collisions = EventTerm(func=assert_openarm_self_collision_contract, mode="startup")

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
        weight=10.0,
    )

    right_fingers_near_object = RewTerm(
        func=mdp.object_bodies_average_distance,
        params={
            "std": 0.07,
            "object_offset": GRASP_OFFSET_W,
            "robot_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
        },
        weight=12.0,
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
        weight=12.0,
    )

    gripper_open_after_grasp_outside_goal_xy = RewTerm(
        func=mdp.gripper_open_after_grasp_outside_goal_xy,
        params={
            "xy_threshold": GOAL_RELEASE_XY_THRESHOLD,
            "command_name": "object_pose",
            "open_joint_pos": GRIPPER_OPEN_POS,
            "proximity_std": GRASP_GATE_STD,
            "proximity_threshold": 0.45,
            "closed_threshold": 0.60,
            "acquisition_steps": 3,
            "min_progress": GOAL_RELEASE_MIN_PROGRESS,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=-160.0,
    )

    closed_gripper_lift_target = RewTerm(
        func=mdp.closed_gripper_lift_target,
        params={
            "std": 0.08,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "lift_offset": 0.12,
            "object_offset": GRASP_OFFSET_W,
            "robot_body_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY_NAME]),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=36.0,
    )

    right_hand_lift_after_grasp = RewTerm(
        func=mdp.body_lift_progress_with_closed_gripper,
        params={
            "start_height": HAND_LIFT_START_HEIGHT_W,
            "target_height": HAND_LIFT_TARGET_HEIGHT_W,
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "robot_body_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY_NAME]),
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=42.0,
    )

    right_hand_y_transport_after_grasp = RewTerm(
        func=mdp.body_axis_progress_with_closed_gripper,
        params={
            "axis": 1,
            "start_value_b": OBJECT_START_POS_B[1],
            "target_value_b": RIGHT_HAND_CARRY_TARGET_Y_B,
            "object_minimal_height": LIFT_PROGRESS_START_HEIGHT_W,
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "robot_body_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY_NAME]),
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=56.0,
    )

    lifting_object_early = RewTerm(
        func=mdp.object_lift_progress_with_grasp,
        params={
            "start_height": LIFT_PROGRESS_START_HEIGHT_W,
            "target_height": TRANSPORT_LIFT_HEIGHT_W,
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=60.0,
    )

    lifting_object = RewTerm(
        func=mdp.object_lift_progress_with_grasp,
        params={
            "start_height": TRANSPORT_LIFT_HEIGHT_W,
            "target_height": MIN_LIFT_HEIGHT_W,
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=40.0,
    )

    carrying_object_high = RewTerm(
        func=mdp.object_lift_progress_with_grasp,
        params={
            "start_height": MIN_LIFT_HEIGHT_W,
            "target_height": CARRY_LIFT_HEIGHT_W,
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=48.0,
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance_with_grasp,
        params={
            "std": 0.30,
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=50.0,
    )

    object_goal_xy_tracking = RewTerm(
        func=mdp.object_goal_xy_distance,
        params={
            "std": 0.18,
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=60.0,
    )

    object_goal_xy_progress = RewTerm(
        func=mdp.object_goal_xy_progress_with_grasp,
        params={
            "start_pos_b": OBJECT_START_POS_B,
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=42.0,
    )

    object_goal_xy_potential_progress = RewTerm(
        func=mdp.object_goal_xy_potential_progress,
        params={
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "minimum_grasp_gate": 0.25,
            "max_step_distance": 0.03,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=600.0,
    )

    object_goal_xy_bonus_wide = RewTerm(
        func=mdp.object_goal_xy_bonus,
        params={
            "threshold": GOAL_XY_BONUS_WIDE_THRESHOLD,
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=24.0,
    )

    object_goal_xy_bonus_mid = RewTerm(
        func=mdp.object_goal_xy_bonus,
        params={
            "threshold": GOAL_XY_BONUS_MID_THRESHOLD,
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=50.0,
    )

    object_goal_xy_bonus_tight = RewTerm(
        func=mdp.object_goal_xy_bonus,
        params={
            "threshold": GOAL_XY_BONUS_TIGHT_THRESHOLD,
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=80.0,
    )

    object_goal_xy_shortfall = RewTerm(
        func=mdp.object_goal_xy_shortfall,
        params={
            "threshold": GOAL_XY_SHORTFALL_THRESHOLD,
            "max_distance": GOAL_XY_SHORTFALL_MAX_DISTANCE,
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=-42.0,
    )

    object_goal_x_tracking = RewTerm(
        func=mdp.object_goal_axis_distance_with_grasp,
        params={
            "axis": 0,
            "std": GOAL_AXIS_STD,
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=16.0,
    )

    object_goal_y_tracking = RewTerm(
        func=mdp.object_goal_axis_distance_with_grasp,
        params={
            "axis": 1,
            "std": GOAL_AXIS_STD,
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=36.0,
    )

    object_goal_y_tracking_broad = RewTerm(
        func=mdp.object_goal_axis_distance_with_grasp,
        params={
            "axis": 1,
            "std": GOAL_AXIS_BROAD_STD,
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=46.0,
    )

    object_goal_y_progress = RewTerm(
        func=mdp.object_goal_axis_progress_with_grasp,
        params={
            "axis": 1,
            "start_value_b": OBJECT_START_POS_B[1],
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=72.0,
    )

    object_goal_y_shortfall = RewTerm(
        func=mdp.object_goal_axis_shortfall_with_grasp,
        params={
            "axis": 1,
            "start_value_b": OBJECT_START_POS_B[1],
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=-58.0,
    )

    object_goal_z_tracking = RewTerm(
        func=mdp.object_goal_axis_distance_with_grasp,
        params={
            "axis": 2,
            "std": 0.08,
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=18.0,
    )

    object_goal_height_tracking = RewTerm(
        func=mdp.object_goal_height_tracking,
        params={
            "std": 0.08,
            "xy_threshold": GOAL_HEIGHT_XY_THRESHOLD,
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=18.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance_with_grasp,
        params={
            "std": 0.08,
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=30.0,
    )

    object_goal_bonus_coarse = RewTerm(
        func=mdp.object_goal_bonus,
        params={
            "threshold": GOAL_BONUS_COARSE_THRESHOLD,
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=16.0,
    )

    object_goal_bonus_mid = RewTerm(
        func=mdp.object_goal_bonus,
        params={
            "threshold": GOAL_BONUS_MID_THRESHOLD,
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=44.0,
    )

    object_goal_bonus_tight = RewTerm(
        func=mdp.object_goal_bonus,
        params={
            "threshold": GOAL_BONUS_TIGHT_THRESHOLD,
            "minimal_height": TRANSPORT_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "grasp_std": GRASP_GATE_STD,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "object_offset": GRASP_OFFSET_W,
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=70.0,
    )

    object_goal_release_readiness = RewTerm(
        func=mdp.object_goal_release_readiness,
        params={
            "xy_threshold": GOAL_RELEASE_XY_THRESHOLD,
            "z_threshold": GOAL_RELEASE_Z_THRESHOLD,
            "minimal_height": GOAL_RELEASE_MIN_HEIGHT_W,
            "command_name": "object_pose",
            "velocity_std": GOAL_RELEASE_VELOCITY_STD,
            "min_progress": GOAL_RELEASE_MIN_PROGRESS,
        },
        weight=100.0,
    )

    gripper_open_near_goal = RewTerm(
        func=mdp.gripper_open_near_goal,
        params={
            "threshold": GOAL_RELEASE_THRESHOLD,
            "minimal_height": GOAL_RELEASE_MIN_HEIGHT_W,
            "command_name": "object_pose",
            "open_joint_pos": GRIPPER_OPEN_POS,
            "velocity_std": GOAL_RELEASE_VELOCITY_STD,
            "min_progress": GOAL_RELEASE_MIN_PROGRESS,
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=80.0,
    )

    gripper_open_near_goal_xy = RewTerm(
        func=mdp.gripper_open_near_goal_xy,
        params={
            "xy_threshold": GOAL_RELEASE_XY_THRESHOLD,
            "z_threshold": GOAL_RELEASE_Z_THRESHOLD,
            "minimal_height": GOAL_RELEASE_MIN_HEIGHT_W,
            "command_name": "object_pose",
            "open_joint_pos": GRIPPER_OPEN_POS,
            "velocity_std": GOAL_RELEASE_VELOCITY_STD,
            "min_progress": GOAL_RELEASE_MIN_PROGRESS,
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=120.0,
    )

    gripper_open_far_from_goal_xy = RewTerm(
        func=mdp.gripper_open_far_from_goal_xy,
        params={
            "xy_threshold": GOAL_RELEASE_XY_THRESHOLD,
            "minimal_height": LIFT_PROGRESS_START_HEIGHT_W,
            "command_name": "object_pose",
            "open_joint_pos": GRIPPER_OPEN_POS,
            "min_progress": GOAL_RELEASE_MIN_PROGRESS,
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=-120.0,
    )

    gripper_closed_near_goal = RewTerm(
        func=mdp.gripper_closed_near_goal,
        params={
            "threshold": GOAL_RELEASE_THRESHOLD,
            "minimal_height": GOAL_RELEASE_MIN_HEIGHT_W,
            "command_name": "object_pose",
            "open_joint_pos": GRIPPER_OPEN_POS,
            "velocity_std": GOAL_RELEASE_VELOCITY_STD,
            "min_progress": GOAL_RELEASE_MIN_PROGRESS,
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=-40.0,
    )

    gripper_closed_near_goal_xy = RewTerm(
        func=mdp.gripper_closed_near_goal_xy,
        params={
            "xy_threshold": GOAL_RELEASE_XY_THRESHOLD,
            "z_threshold": GOAL_RELEASE_Z_THRESHOLD,
            "minimal_height": GOAL_RELEASE_MIN_HEIGHT_W,
            "command_name": "object_pose",
            "open_joint_pos": GRIPPER_OPEN_POS,
            "velocity_std": GOAL_RELEASE_VELOCITY_STD,
            "min_progress": GOAL_RELEASE_MIN_PROGRESS,
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=-160.0,
    )

    gripper_closed_far_from_goal_xy = RewTerm(
        func=mdp.gripper_closed_far_from_goal_xy,
        params={
            "xy_threshold": GOAL_RELEASE_XY_THRESHOLD,
            "minimal_height": LIFT_PROGRESS_START_HEIGHT_W,
            "command_name": "object_pose",
            "open_joint_pos": GRIPPER_OPEN_POS,
            "min_progress": GOAL_RELEASE_MIN_PROGRESS,
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=24.0,
    )

    gripper_open_before_final_goal = RewTerm(
        func=mdp.gripper_open_before_final_goal,
        params={
            "minimal_height": MIN_LIFT_HEIGHT_W,
            "command_name": "object_pose",
            "open_joint_pos": GRIPPER_OPEN_POS,
            "min_progress": GOAL_RELEASE_MIN_PROGRESS,
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=-12.0,
    )

    object_goal_settled_after_release = RewTerm(
        func=mdp.object_goal_settled_after_release,
        params={
            "xy_threshold": GOAL_SETTLE_XY_THRESHOLD,
            "z_threshold": GOAL_SETTLE_Z_THRESHOLD,
            "command_name": "object_pose",
            "velocity_std": 0.05,
            "min_progress": GOAL_SETTLE_MIN_PROGRESS,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
        weight=160.0,
    )

    successful_release = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": "object_released_at_goal"},
        weight=3000.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2_clamped, weight=-1e-4, params={"max_rate": 2.0})

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
    object_released_at_goal = DoneTerm(
        func=mdp.object_released_at_goal,
        params={
            "xy_threshold": GOAL_SUCCESS_XY_THRESHOLD,
            "z_threshold": GOAL_SUCCESS_Z_THRESHOLD,
            "max_speed": GOAL_SUCCESS_MAX_SPEED,
            "min_progress": GOAL_SUCCESS_MIN_PROGRESS,
            "command_name": "object_pose",
            "open_joint_pos": GRIPPER_OPEN_POS,
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
    )


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
class CurriculumCfgV18(CurriculumCfg):
    """Keep task outcome costs stationary while a fresh policy discovers grasping."""

    action_rate = None
    joint_vel = None


@configclass
class ObservationsCfgV18:
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # Raw sampled actions differ from deterministic play and from targets overridden by the task FSM.
        actions = None
        target_object_position = ObsTerm(
            func=mdp.transport_goal_command,
            params={
                "command_name": "object_pose",
                "carry_height_b": STAGE_CARRY_TARGET_HEIGHT_B,
            },
        )
        object_to_goal_position = ObsTerm(
            func=mdp.object_to_transport_goal_position_in_robot_root_frame,
            params={
                "command_name": "object_pose",
                "carry_height_b": STAGE_CARRY_TARGET_HEIGHT_B,
            },
        )
        object_linear_velocity = ObsTerm(func=mdp.object_linear_velocity_in_robot_root_frame)
        task_state = ObsTerm(func=mdp.task_state_observation)
        grasp_quality = ObsTerm(func=mdp.task_grasp_quality_observation)
        episode_progress = ObsTerm(func=mdp.episode_progress_observation)
        release_target = ObsTerm(func=mdp.task_release_target_observation)

    @configclass
    class AuthorityCfg(ObsGroup):
        actor = ObsTerm(func=mdp.task_actor_authority_observation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    authority: AuthorityCfg = AuthorityCfg()


@configclass
class RewardsCfgV18:
    """Keep the verified pre-grasp contract and mask only post-grasp assist motion."""

    pregrasp_centering = RewTerm(
        func=mdp.task_pregrasp_centering_reward,
        params={
            "quality_scale": 5.0,
            "dwell_scale": 1.0,
        },
        weight=1.0,
    )
    lift_progress = RewTerm(
        func=mdp.task_potential_progress,
        params={"mode": "lift", "max_step_change": 0.03},
        weight=150.0,
    )
    verified_lift_progress = RewTerm(
        func=mdp.task_verified_paired_lift_progress,
        params={
            "target_height": STAGE_CARRY_MIN_HEIGHT_W,
            "maximum_credit_height": 0.093,
            "bootstrap_length_scale": 0.010,
            "bootstrap_multiplier": 4.0,
        },
        weight=150.0,
    )
    transport_progress = RewTerm(
        func=mdp.task_potential_progress,
        params={
            "mode": "transport",
            "max_step_change": 0.03,
            "minimum_transport_height": STAGE_CARRY_MIN_HEIGHT_W,
        },
        weight=150.0,
    )
    carry_height_progress = RewTerm(
        func=mdp.task_potential_progress,
        params={
            "mode": "carry_height",
            "max_step_change": 0.03,
            "target_height": STAGE_CARRY_TARGET_HEIGHT_W,
        },
        weight=150.0,
    )

    grasp_ready = RewTerm(
        func=mdp.task_event_reward,
        params={"event_name": "just_grasp_ready"},
        weight=8.0,
    )
    grasp_candidate = RewTerm(
        func=mdp.task_event_reward,
        params={"event_name": "just_grasp_candidate"},
        weight=15.0,
    )
    grasp_acquired = RewTerm(
        func=mdp.task_event_reward,
        params={"event_name": "just_grasped"},
        weight=50.0,
    )
    lift_completed = RewTerm(
        func=mdp.task_event_reward,
        params={"event_name": "just_lifted"},
        weight=15.0,
    )
    assist_engaged = RewTerm(
        func=mdp.task_event_reward,
        params={"event_name": "just_assist_engaged"},
        weight=40.0,
    )
    object_released = RewTerm(
        func=mdp.task_event_reward,
        params={"event_name": "just_released"},
        weight=25.0,
    )
    successful_place = RewTerm(
        func=mdp.task_event_reward,
        params={"event_name": "succeeded"},
        weight=200.0,
    )
    failed_place = RewTerm(
        func=mdp.task_failure_with_preassist_cost,
        params={"preassist_multiplier": 2.0},
        weight=-55.0,
    )

    incomplete_timeout = RewTerm(
        func=mdp.task_timeout_without_success,
        params={"preassist_multiplier": 2.0},
        weight=-50.0,
    )
    elapsed_time = RewTerm(func=mdp.constant_penalty, weight=-0.5)
    pregrasp_delay = RewTerm(
        func=mdp.task_pregrasp_delay_penalty,
        params={"retry_event_scale": 0.20},
        weight=-7.5,
    )
    preassist_delay = RewTerm(func=mdp.task_preassist_delay_penalty, weight=-1.5)
    preclearance_inward_progress = RewTerm(
        func=mdp.task_preclearance_inward_progress,
        params={
            "clearance_start_height": LOW_CLEARANCE_PENALTY_START_HEIGHT_W,
            "clearance_full_height": LOW_CLEARANCE_PENALTY_FULL_HEIGHT_W,
            "near_basket_xy_threshold": LOW_CLEARANCE_BASKET_APPROACH_XY,
            "max_step_change": 0.03,
        },
        weight=-200.0,
    )
    poststicky_low_clearance_exposure = RewTerm(
        func=mdp.task_poststicky_low_clearance_exposure,
        params={
            "clearance_height": STAGE_CARRY_MIN_HEIGHT_W,
            "near_basket_xy_threshold": LOW_CLEARANCE_BASKET_APPROACH_XY,
        },
        weight=-5.0,
    )
    prelift_inward_progress = RewTerm(
        func=mdp.task_prelift_inward_progress,
        params={
            "free_tolerance": PRELIFT_INWARD_FREE_TOLERANCE,
            "maximum_charged_excursion": PRELIFT_INWARD_MAX_CHARGED_EXCURSION,
        },
        weight=-300.0,
    )
    gripper_action_phase_error = RewTerm(func=mdp.task_gripper_action_phase_error, weight=-1.0)
    carry_height_band = RewTerm(
        func=mdp.task_carry_height_band_error,
        params={
            "minimum_height": STAGE_CARRY_MIN_HEIGHT_W,
            "maximum_height": STAGE_CARRY_MAX_HEIGHT_W,
        },
        weight=-10.0,
    )
    right_hand_table_clearance = RewTerm(
        func=mdp.task_actor_body_clearance,
        params={
            "minimum_height": RIGHT_HAND_MIN_CLEARANCE_W,
            "robot_body_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY_NAME]),
        },
        weight=-2.0,
    )
    joint_vel = RewTerm(
        func=mdp.task_actor_joint_velocity_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_POLICY_JOINT_NAMES),
        },
    )


@configclass
class TerminationsCfgV18:
    task_state = DoneTerm(
        func=mdp.TaskStateTracker,
        params={
            "object_start_height": OBJECT_START_POS_W[2],
            "carry_height": STAGE_CARRY_MIN_HEIGHT_W,
            "release_height_levels": (STAGE_RELEASE_TARGET_HEIGHT_W,),
            "release_height_probabilities": (1.0,),
            "release_height_normalization_bounds": (0.670, 0.75),
            "settle_height": 0.65,
            "centering_height": STAGE_CARRY_TARGET_HEIGHT_W,
            "descent_start_xy_threshold": 0.12,
            "descent_transition_xy_threshold": 0.04,
            "descent_end_xy_threshold": 0.005,
            "preplace_height_above_release": 0.020,
            "post_center_descent_xy_threshold": 0.030,
            "post_center_descent_steps": 12,
            "post_center_descent_max_tracking_error": 0.012,
            "assist_entry_xy_threshold": STAGE_ASSIST_ENTRY_XY_THRESHOLD,
            "basket_entry_xy_threshold": 0.10,
            "place_center_xy_threshold": 0.04,
            "place_center_max_xy_speed": 0.05,
            "release_xy_threshold": 0.04,
            "success_xy_threshold": 0.05,
            "basket_exit_xy_threshold": 0.20,
            "release_height_above_tolerance": 0.002,
            "release_height_below_tolerance": 0.003,
            "settle_height_tolerance": 0.045,
            "release_max_speed": 0.06,
            "settle_max_speed": 0.08,
            "proximity_std": GRASP_GATE_STD,
            "candidate_proximity_threshold": 0.48,
            "attempt_hold_proximity_threshold": 0.38,
            "hold_proximity_threshold": 0.25,
            "candidate_closed_threshold": 0.40,
            "hold_closed_threshold": 0.20,
            "grasp_confirm_steps": 5,
            "lift_confirm_steps": 3,
            "transport_confirm_steps": 3,
            "candidate_min_open_fraction": 0.08,
            "hold_min_open_fraction": 0.08,
            "alignment_target": (-0.008, 0.003, -0.028),
            "alignment_lower_bounds": (-0.020, -0.005, -0.040),
            "alignment_upper_bounds": (0.010, 0.008, -0.012),
            "alignment_hold_lower_bounds": (-0.026, -0.009, -0.046),
            "alignment_hold_upper_bounds": (0.016, 0.013, -0.006),
            "alignment_quality_scales": (0.040, 0.025, 0.040),
            "alignment_max_finger_asymmetry": 0.007,
            "alignment_hold_max_finger_asymmetry": 0.012,
            "alignment_max_relative_speed": 0.10,
            "preclose_min_open_fraction": 0.85,
            "alignment_confirm_steps": 3,
            "contact_max_gripper_speed": 0.03,
            "contact_max_aperture_asymmetry": 0.20,
            "closing_min_steps": 10,
            "contact_confirm_steps": 3,
            "attempt_lost_confirm_steps": 6,
            "closing_timeout_steps": 30,
            "empty_close_confirm_steps": 3,
            "closing_contact_grace_steps": 12,
            "verification_timeout_steps": 90,
            "reopen_arm_hold_steps": 12,
            "retry_cooldown_steps": 6,
            "coupling_object_lift": 0.012,
            "coupling_tcp_lift": 0.010,
            "coupling_max_displacement_error": 0.012,
            "candidate_max_relative_speed": 0.10,
            "hold_max_offset_error": 0.025,
            "hold_max_relative_speed": 0.20,
            "transport_min_displacement": 0.05,
            "place_center_confirm_steps": 3,
            "release_ready_confirm_steps": 3,
            "release_open_grace_steps": 10,
            "lost_confirm_steps": 3,
            "assist_lost_confirm_steps": 6,
            "basket_exit_confirm_steps": 24,
            "released_outside_confirm_steps": 10,
            "settle_confirm_steps": 10,
            "open_joint_pos": GRIPPER_OPEN_POS,
            "command_name": "object_pose",
            "grasp_body_cfg": SceneEntityCfg("robot", body_names=RIGHT_FINGER_BODY_NAMES),
            "tcp_body_cfg": SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY_NAME]),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINT_NAMES),
        },
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    task_failed = DoneTerm(func=mdp.task_state_failed)
    task_succeeded = DoneTerm(func=mdp.task_state_succeeded)


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
        apply_openarm_self_collision_physx_profile(self.sim.physx)
        self.sim.physx.friction_correlation_distance = 0.00625

        self.viewer.eye = (1.6, -1.2, 1.2)
        self.viewer.lookat = (0.45, 0.0, 0.65)


@configclass
class OpenArmTestUsdPickPlaceEnvCfg_PLAY(OpenArmTestUsdPickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.object_pose.force_final_goal = True
        self.commands.object_pose.ranges.pos_x = (BASKET_GOAL_POS_B[0], BASKET_GOAL_POS_B[0])
        self.commands.object_pose.ranges.pos_y = (BASKET_GOAL_POS_B[1], BASKET_GOAL_POS_B[1])
        self.commands.object_pose.ranges.pos_z = (BASKET_GOAL_POS_B[2], BASKET_GOAL_POS_B[2])


@configclass
class OpenArmTestUsdPickPlaceEnvCfg_FINETUNE(OpenArmTestUsdPickPlaceEnvCfg):
    """Fine-tune an existing grasp policy directly against randomized final basket goals."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.force_final_goal = True


@configclass
class OpenArmTestUsdPickPlaceEnvCfg_STAGE(OpenArmTestUsdPickPlaceEnvCfg):
    """Stage-aligned task with observable grasp state and outcome-based termination."""

    observations: ObservationsCfgV18 = ObservationsCfgV18()
    rewards: RewardsCfgV18 = RewardsCfgV18()
    terminations: TerminationsCfgV18 = TerminationsCfgV18()
    curriculum: CurriculumCfgV18 = CurriculumCfgV18()

    def __post_init__(self):
        super().__post_init__()
        self.is_finite_horizon = True
        self.scene.replicate_physics = True
        self.commands.object_pose.force_final_goal = True
        self.commands.object_pose.debug_vis = False
        self.actions.arm_action = mdp.PlaceAssistJointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_ARM_JOINT_NAMES,
            scale=0.5,
            use_default_offset=True,
            body_name=RIGHT_EE_BODY_NAME,
            object_name="object",
            command_name="object_pose",
            horizontal_position_gain=0.20,
            horizontal_velocity_damping=0.25,
            vertical_position_gain=0.16,
            post_center_vertical_position_gain=0.20,
            vertical_velocity_damping=0.30,
            max_horizontal_step=0.0025,
            max_vertical_step=0.0030,
            max_upward_step=0.0025,
            max_joint_step=0.03,
            max_target_lead=0.12,
            max_horizontal_target_lead=0.012,
            max_upward_target_lead=0.025,
            max_downward_target_lead=0.012,
            max_angular_target_lead=0.03,
            nullspace_gain=0.04,
            max_nullspace_step=0.003,
            velocity_ema_decay=0.80,
            max_object_speed_for_control=0.25,
            assist_entry_blend_steps=12,
            assist_entry_target_lead_steps=3,
            assist_target_lead_steps=3,
            max_horizontal_acceleration=0.60,
            max_vertical_acceleration=0.45,
            max_horizontal_jerk=6.0,
            max_vertical_jerk=4.5,
            horizontal_position_deadband=0.005,
            vertical_position_deadband=0.0015,
            cartesian_leak_tolerance=0.0002,
            cartesian_hard_leak_tolerance=0.0010,
            minimum_active_progress_ratio=0.25,
            minimum_companion_progress_ratio=0.05,
            slip_watchdog_threshold=0.015,
            slip_watchdog_steps=3,
            slip_recovery_hold_steps=3,
            handoff_steps=12,
            damping=0.08,
            outer_xy_threshold=STAGE_ASSIST_OUTER_XY_THRESHOLD,
            grasp_handoff_steps=12,
        )
        self.actions.gripper_action = mdp.HystereticBinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_GRIPPER_JOINT_NAMES,
            open_command_expr={"openarm_right_finger_joint.*": GRIPPER_OPEN_POS},
            close_command_expr={"openarm_right_finger_joint.*": 0.0},
            open_threshold=0.15,
            close_threshold=-0.15,
            enforce_task_phase=True,
            automatic_phase_control=True,
        )


@configclass
class OpenArmTestUsdPickPlaceEnvCfg_STAGE_PLAY(OpenArmTestUsdPickPlaceEnvCfg_STAGE):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.commands.object_pose.debug_vis = True
        self.commands.object_pose.ranges.pos_x = (BASKET_GOAL_POS_B[0], BASKET_GOAL_POS_B[0])
        self.commands.object_pose.ranges.pos_y = (BASKET_GOAL_POS_B[1], BASKET_GOAL_POS_B[1])
        self.commands.object_pose.ranges.pos_z = (BASKET_GOAL_POS_B[2], BASKET_GOAL_POS_B[2])
