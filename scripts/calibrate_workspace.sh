#!/usr/bin/env bash

set -euo pipefail
source /home/mh/openarm_dp3/scripts/env.sh

if ! "${OPENARM_DP3_PYTHON}" -c 'import zarr' >/dev/null 2>&1; then
  /home/mh/openarm_dp3/scripts/setup_python_overlay.sh
fi

exec "${OPENARM_DP3_PYTHON}" -m openarm_dp3.calibrate_workspace "$@"
