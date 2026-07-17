#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

SCALE = 0.2

class DryRun(Node):
    def __init__(self):
        super().__init__('dry_run_exo_mapping')
        self.zero = None
        self.create_subscription(JointState, '/exo/joint_command', self.cb, 10)

    def cb(self, msg):
        if len(msg.position) < 16:
            return

        q = list(msg.position)

        if self.zero is None:
            self.zero = q
            self.get_logger().info('Captured ExoArm zero pose. Move slowly now.')
            return

        left = [SCALE * (q[i] - self.zero[i]) for i in range(7)]
        right = [SCALE * (q[i] - self.zero[i]) for i in range(7, 14)]

        print('\nLEFT -> can0:', ['%.3f' % v for v in left])
        print('RIGHT -> can1:', ['%.3f' % v for v in right])
        print('TRIGGER:', q[14], q[15])

def main():
    rclpy.init()
    rclpy.spin(DryRun())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
