#!/bin/bash
ARM_SIDE=${1:-right_arm}
CAN_IF=${2:-can0}
SCALE=${3:-0.2}

WS_DIR=~/openarm_ws
XACRO_PATH="$WS_DIR/src/openarm_description/assets/robot/openarm_v1.0/urdf/openarm_v10.urdf.xacro"
TMPDIR="/tmp/openarm_urdf_gen"
URDF_OUT="$TMPDIR/openarm_v10_exo.urdf"
BIN_PATH=~/openarm_teleop/build/exo_unilateral_control

source /opt/ros/humble/setup.bash
source "$WS_DIR/install/setup.bash"

mkdir -p "$TMPDIR"
xacro "$XACRO_PATH" bimanual:=true -o "$URDF_OUT"

"$BIN_PATH" "$URDF_OUT" "$ARM_SIDE" "$CAN_IF" "$SCALE"
