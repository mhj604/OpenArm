#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mh/openarm_imitation"
BOOTSTRAP="$ROOT/.virtualenv-bootstrap"
if [[ ! -d "$BOOTSTRAP/virtualenv" ]]; then
  python3 -m pip install --target "$BOOTSTRAP" virtualenv==20.31.2
fi
PYTHONPATH="$BOOTSTRAP" python3 -m virtualenv --system-site-packages "$ROOT/.venv-collect"
"$ROOT/.venv-collect/bin/python" -m pip install --upgrade pip 'setuptools<80' wheel
"$ROOT/.venv-collect/bin/python" -m pip install -r "$ROOT/requirements-collect.txt"
"$ROOT/.venv-collect/bin/python" -m pip install --no-deps -e "$ROOT"

set +u
source /opt/ros/humble/setup.bash
set -u
"$ROOT/.venv-collect/bin/python" -c "import rclpy, sensor_msgs, pyrealsense2, can, zarr, openarm_imitation; print('collection environment OK')"
