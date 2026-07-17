#!/usr/bin/env bash

set -euo pipefail
source /home/mh/openarm_dp3/scripts/env.sh

exec "${OPENARM_DP3_PYTHON}" -m openarm_dp3.rollout_profile \
  --dataset /home/mh/openarm_dp3/data/processed/openarm_cube_to_point.zarr \
  --output /home/mh/openarm_dp3/configs/dp3_rollout_profile.json
