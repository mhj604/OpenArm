#!/usr/bin/env bash

set -euo pipefail

# ROS Humble's generated setup files probe optional variables before defining
# them and are not compatible with nounset during sourcing.
set +u
source /opt/ros/humble/setup.bash
source /home/mh/foxtech_exoarm/qnbot-exo-std-demo/install/setup.bash
set -u

exec ros2 launch qnbot_teleoperator websocket_teleoperator.launch.py \
  enable_left_arm:=true \
  enable_right_arm:=true \
  enable_left_joystick:=true \
  enable_right_joystick:=true \
  enable_vehicle_control:=false \
  websocket_host:=0.0.0.0 \
  websocket_port:=19091 \
  publish_rate_hz:=100.0 \
  timeout_ms:=100
