#!/usr/bin/env bash

set -euo pipefail

readonly ROOT="/home/mh/openarm_act"
readonly CONFIG="${ACT_ROLLOUT_CONFIG:-${ROOT}/configs/act_rollout.env}"

if [[ "${OPENARM_ACT_EXECUTE:-}" != "YES" ]]; then
  echo "This command enables and moves both OpenArm arms." >&2
  echo "Run it only with explicit execution consent:" >&2
  echo "  OPENARM_ACT_EXECUTE=YES ${ROOT}/scripts/rollout_act.sh" >&2
  exit 2
fi

set -a
source "${CONFIG}"
set +a
source "${ROOT}/scripts/env.sh"

if pgrep -f '/home/mh/openarm_teleop/exo_openarm_bridge/exo_to_openarm.py' >/dev/null; then
  echo "The ExoArm-to-OpenArm bridge is already running. Stop it before ACT rollout." >&2
  exit 3
fi

readonly SESSION_TAG="$(date +%Y%m%d_%H%M%S)"
readonly MOTOR_LOG="${ROOT}/logs/act_motor_${SESSION_TAG}.log"
readonly POLICY_LOG="${ROOT}/logs/act_policy_${SESSION_TAG}.log"

motor_pid=""
cleanup() {
  if [[ -n "${motor_pid}" ]] && kill -0 "${motor_pid}" 2>/dev/null; then
    kill -TERM "${motor_pid}" 2>/dev/null || true
    wait "${motor_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${ACT_DURATION_S}" == "0" || "${ACT_DURATION_S}" == "0.0" ]]; then
  echo "ACTIVE ACT ROLLOUT: both arms may move until Ctrl+C"
else
  echo "ACTIVE ACT ROLLOUT: both arms may move for ${ACT_DURATION_S}s"
fi
echo "Checkpoint: ${ACT_CHECKPOINT}"
echo "Motor log: ${MOTOR_LOG}"
echo "Policy log: ${POLICY_LOG}"

/usr/bin/python3 "${ROOT}/rollout/openarm_act_motor_server.py" \
  --left-can "${OPENARM_LEFT_CAN}" \
  --right-can "${OPENARM_RIGHT_CAN}" \
  --host "${ACT_CONTROL_HOST}" \
  --action-port "${ACT_ACTION_PORT}" \
  --state-port "${ACT_STATE_PORT}" \
  --fps "${ACT_FPS}" \
  --stale-timeout "${ACT_COMMAND_STALE_TIMEOUT_S}" \
  --max-arm-delta "${ACT_MAX_ARM_DELTA_RAD}" \
  --max-gripper-delta "${ACT_MAX_GRIPPER_DELTA_RAD}" \
  --stats-json "${ACT_TRAINING_STATS}" \
  --log-file "${MOTOR_LOG}" &
motor_pid=$!

python "${ROOT}/rollout/act_policy_client.py" \
  --checkpoint "${ACT_CHECKPOINT}" \
  --chest-serial "${CHEST_SERIAL}" \
  --wrist-left-serial "${WRIST_LEFT_SERIAL}" \
  --host "${ACT_CONTROL_HOST}" \
  --action-port "${ACT_ACTION_PORT}" \
  --state-port "${ACT_STATE_PORT}" \
  --fps "${ACT_FPS}" \
  --duration "${ACT_DURATION_S}" \
  --device cuda \
  --log-file "${POLICY_LOG}"

wait "${motor_pid}"
motor_pid=""
