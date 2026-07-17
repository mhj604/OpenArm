#!/usr/bin/env bash

set -euo pipefail
source /home/mh/openarm_dp3/scripts/env.sh

exec "${OPENARM_DP3_PYTHON}" -m openarm_dp3.validate_dataset "$@"
