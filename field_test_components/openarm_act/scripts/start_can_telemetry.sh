#!/usr/bin/env bash

set -euo pipefail

OPENARM_ACT_ROOT="/home/mh/openarm_act"
HARDWARE_ENV="${HARDWARE_ENV:-${OPENARM_ACT_ROOT}/hardware.env}"
if [[ ! -f "${HARDWARE_ENV}" ]]; then
  echo "Missing ${HARDWARE_ENV}; copy configs/hardware.env.example and fill the current interfaces" >&2
  exit 2
fi
set -a
source "${HARDWARE_ENV}"
set +a

: "${OPENARM_LEFT_CAN:?Set the existing bridge's physical left CAN interface}"
: "${OPENARM_RIGHT_CAN:?Set the existing bridge's physical right CAN interface}"

exec /usr/bin/python3 "${OPENARM_ACT_ROOT}/ros2_adapter/openarm_can_telemetry_udp.py" \
  --left-interface "${OPENARM_LEFT_CAN}" \
  --right-interface "${OPENARM_RIGHT_CAN}" \
  --host "${OPENARM_TELEMETRY_HOST:-127.0.0.1}" \
  --port "${OPENARM_TELEMETRY_PORT:-29502}" \
  --publish-hz "${OPENARM_TELEMETRY_HZ:-100}"
