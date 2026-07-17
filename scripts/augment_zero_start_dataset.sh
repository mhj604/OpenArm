#!/usr/bin/env bash

set -euo pipefail
source /home/mh/openarm_dp3/scripts/env.sh

readonly SOURCE="/home/mh/openarm_dp3/data/processed/openarm_cube_to_point.zarr"
readonly OUTPUT="/home/mh/openarm_dp3/data/processed/openarm_cube_to_point_zero_start.zarr"

exec "${OPENARM_DP3_PYTHON}" -m openarm_dp3.augment_zero_start \
  --source "${SOURCE}" \
  --output "${OUTPUT}" \
  --prefix-frames "${DP3_ZERO_START_PREFIX_FRAMES:-30}"
