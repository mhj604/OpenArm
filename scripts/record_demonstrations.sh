#!/usr/bin/env bash

set -euo pipefail
source /home/mh/openarm_dp3/scripts/env.sh

NUM_EPISODES="${1:-40}"
shift || true
if [[ ! "${NUM_EPISODES}" =~ ^[0-9]+$ ]] || (( NUM_EPISODES < 1 )); then
  echo "Usage: $0 [positive episode count]" >&2
  exit 2
fi
if ! "${OPENARM_DP3_PYTHON}" -c 'import zarr' >/dev/null 2>&1; then
  /home/mh/openarm_dp3/scripts/setup_python_overlay.sh
fi

telemetry_pid=""
cleanup() {
  if [[ -n "${telemetry_pid}" ]] && kill -0 "${telemetry_pid}" 2>/dev/null; then
    kill -TERM "${telemetry_pid}" 2>/dev/null || true
    wait "${telemetry_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if ! pgrep -af '/home/mh/openarm_act/ros2_adapter/openarm_can_telemetry_udp.py' >/dev/null; then
  log="/home/mh/openarm_dp3/logs/passive_telemetry_$(date +%Y%m%d_%H%M%S).log"
  /home/mh/openarm_act/scripts/start_can_telemetry.sh >"${log}" 2>&1 &
  telemetry_pid=$!
  sleep 1
  if ! kill -0 "${telemetry_pid}" 2>/dev/null; then
    echo "Receive-only telemetry failed. See ${log}" >&2
    exit 3
  fi
  echo "Started receive-only CAN telemetry (PID ${telemetry_pid}); log=${log}"
fi

echo "This collector does not enable or command OpenArm motors."
"${OPENARM_DP3_PYTHON}" -m openarm_dp3.record_demonstrations \
  --num-episodes "${NUM_EPISODES}" "$@"
