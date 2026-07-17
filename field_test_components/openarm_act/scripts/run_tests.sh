#!/usr/bin/env bash

set -euo pipefail

source /home/mh/openarm_act/scripts/env.sh

# ROS Humble installs Python 3.10 pytest entry points globally. They cannot be
# imported by this isolated Python 3.12 environment, so test only this project.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="/home/mh/openarm_act${PYTHONPATH:+:${PYTHONPATH}}"

ruff check \
  /home/mh/openarm_act/ros2_adapter \
  /home/mh/openarm_act/scripts/test_cameras.py \
  /home/mh/openarm_act/tests \
  /home/mh/openarm_act/plugins/lerobot_robot_openarm_can_passive/src
pytest -q /home/mh/openarm_act/tests
