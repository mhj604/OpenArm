#!/usr/bin/env python3
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import openarm_can as oa


GRIPPER_POSITION_LIMIT_RAD = 0.7854
GRIPPER_SCALE_RAD = float(os.environ.get("OPENARM_GRIPPER_SCALE_RAD", "0.70"))
if not 0.0 < GRIPPER_SCALE_RAD <= GRIPPER_POSITION_LIMIT_RAD:
    raise ValueError(
        "OPENARM_GRIPPER_SCALE_RAD must be in "
        f"(0, {GRIPPER_POSITION_LIMIT_RAD}], got {GRIPPER_SCALE_RAD}"
    )


class ExoToOpenArm(Node):
    def __init__(self):
        super().__init__("exo_to_openarm")

        self.right = oa.OpenArm("can0", False)
        self.left = oa.OpenArm("can1", False)

        motor_types = [oa.MotorType.DM8009, oa.MotorType.DM8009,
                       oa.MotorType.DM4340, oa.MotorType.DM4340,
                       oa.MotorType.DM4310, oa.MotorType.DM4310,
                       oa.MotorType.DM4310]

        self.right.init_arm_motors(motor_types, [1,2,3,4,5,6,7], [17,18,19,20,21,22,23])
        self.left.init_arm_motors(motor_types, [1,2,3,4,5,6,7], [17,18,19,20,21,22,23])

        self.right.init_gripper_motor(oa.MotorType.DM4310, 8, 24)
        self.left.init_gripper_motor(oa.MotorType.DM4310, 8, 24)

        self.right.set_callback_mode_all(oa.CallbackMode.STATE)
        self.left.set_callback_mode_all(oa.CallbackMode.STATE)

        self.right.enable_all()
        self.left.enable_all()

        self.sub = self.create_subscription(
            JointState, "/exo/joint_command", self.cb, 10
        )

    def cb(self, msg):
        if len(msg.position) < 14:
            return

        scale = 1.8

        if not hasattr(self, "exo_home_left"):
            self.exo_home_left = list(msg.position[0:7])
            self.exo_home_right = list(msg.position[7:14])
            self.open_home_left = [0.0] * 7
            self.open_home_right = [0.0] * 7
            self.get_logger().info("Home pose captured")
            return

        left_sign = [1, 1, -1, 1, 1, -1, 1]
        right_sign = [1, 1, -1, 1, 1, -1, -1]

        left_q = [
            self.open_home_left[i] + scale * left_sign[i] * (msg.position[[0,1,2,3,4,6,5][i]] - self.exo_home_left[[0,1,2,3,4,6,5][i]])
            for i in range(7)
        ]

        right_q = [
            self.open_home_right[i] + scale * right_sign[i] * (msg.position[[7,8,9,10,11,13,12][i]] - self.exo_home_right[[0,1,2,3,4,6,5][i]])
            for i in range(7)
        ]

        kp = [20,20,12,12,5,5,4]
        kd = [1.0,1.0,0.8,0.8,0.3,0.3,0.3]

        self.left.get_arm().mit_control_all([
            oa.MITParam(kp[i], kd[i], left_q[i], 0.0, 0.0) for i in range(7)
        ])
        self.right.get_arm().mit_control_all([
            oa.MITParam(kp[i], kd[i], right_q[i], 0.0, 0.0) for i in range(7)
        ])

        # trigger: position[14]=left_trigger, position[15]=right_trigger
        if len(msg.position) >= 16:
            left_trigger = min(max(float(msg.position[14]), 0.0), 1.0)
            right_trigger = min(max(float(msg.position[15]), 0.0), 1.0)
            left_grip = -left_trigger * GRIPPER_SCALE_RAD
            right_grip = -right_trigger * GRIPPER_SCALE_RAD
            self.left.get_gripper().mit_control_all([oa.MITParam(5, 0.3, left_grip, 0.0, 0.0)])
            self.right.get_gripper().mit_control_all([oa.MITParam(5, 0.3, right_grip, 0.0, 0.0)])

        self.left.recv_all()
        self.right.recv_all()

    def destroy_node(self):
        self.left.disable_all()
        self.right.disable_all()
        super().destroy_node()

def main():
    rclpy.init()
    node = ExoToOpenArm()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
