#!/usr/bin/env bash

set -euo pipefail

export OPENARM_DP3_ROOT="/home/mh/openarm_dp3"
export OPENARM_ACT_ROOT="/home/mh/openarm_act"
export OPENARM_DP3_PYTHON="${OPENARM_ACT_ROOT}/.venv/bin/python"
export DP3_UPSTREAM_ROOT="${OPENARM_DP3_ROOT}/vendor/3D-Diffusion-Policy"
export DP3_SOURCE_ROOT="${DP3_UPSTREAM_ROOT}/3D-Diffusion-Policy"

paths=(
  "${OPENARM_DP3_ROOT}/src"
  "${OPENARM_DP3_ROOT}/.python-packages"
  "${OPENARM_ACT_ROOT}/plugins/lerobot_robot_openarm_can_passive/src"
  "${OPENARM_ACT_ROOT}/python/pyrealsense2"
  "${DP3_SOURCE_ROOT}"
)

joined="$(IFS=:; echo "${paths[*]}")"
export PYTHONPATH="${joined}${PYTHONPATH:+:${PYTHONPATH}}"
