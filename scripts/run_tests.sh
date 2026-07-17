#!/usr/bin/env bash

set -euo pipefail
source /home/mh/openarm_dp3/scripts/env.sh

if ! "${OPENARM_DP3_PYTHON}" -c 'import zarr' >/dev/null 2>&1; then
  /home/mh/openarm_dp3/scripts/setup_python_overlay.sh
fi
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec "${OPENARM_DP3_PYTHON}" -m pytest /home/mh/openarm_dp3/tests "$@"
