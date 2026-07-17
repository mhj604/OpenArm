#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <task_name> <num_episodes> [dataset.zarr]" >&2
  exit 2
fi

ROOT="/home/mh/openarm_imitation"
TASK="$1"
EPISODES="$2"
OUTPUT="${3:-$ROOT/datasets/$TASK/dataset.zarr}"

set +u
source /opt/ros/humble/setup.bash
source /home/mh/foxtech_exoarm/qnbot-exo-std-demo/install/setup.bash
set -u
source "$ROOT/.venv-collect/bin/activate"

python "$ROOT/scripts/preflight.py" --hardware
exec python "$ROOT/scripts/collect_bimanual.py" \
  --output "$OUTPUT" \
  --num-episodes "$EPISODES" \
  --task "$TASK"
