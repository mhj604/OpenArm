#!/usr/bin/env bash

set -euo pipefail

readonly ROOT="/home/mh/openarm_dp3"
readonly CONFIG="${DP3_REPLAY_CONFIG:-${ROOT}/configs/dp3_rollout.env}"
readonly RAW_ROOT="${DP3_REPLAY_RAW_ROOT:-${ROOT}/data/raw/episodes}"
readonly EPISODE_INDEX="${DP3_REPLAY_EPISODE:-1}"

if [[ "${OPENARM_DP3_REPLAY:-}" != "YES" ]]; then
  echo "This command replays recorded targets and moves both OpenArm arms." >&2
  echo "Run it only with explicit replay consent:" >&2
  echo "  OPENARM_DP3_REPLAY=YES DP3_REPLAY_EPISODE=1 ${ROOT}/scripts/replay_demonstration.sh" >&2
  exit 2
fi
if [[ ! "${EPISODE_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "DP3_REPLAY_EPISODE must be a non-negative integer." >&2
  exit 3
fi

set -a
source "${CONFIG}"
set +a
source "${ROOT}/scripts/env.sh"

printf -v EPISODE_PATH '%s/episode_%06d.zarr' "${RAW_ROOT}" "${EPISODE_INDEX}"
required_paths=(
  "${EPISODE_PATH}"
  "${DP3_PROFILE}"
  "${ROOT}/rollout/openarm_dp3_motor_server.py"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing replay prerequisite: ${path}" >&2
    exit 4
  fi
done

if pgrep -f 'exo_to_openarm.py' >/dev/null; then
  echo "The ExoArm-to-OpenArm bridge is running. Stop it before replay." >&2
  exit 5
fi
if pgrep -f 'openarm_.*motor_server.py' >/dev/null; then
  echo "Another OpenArm policy motor server is already running." >&2
  exit 6
fi

readonly SESSION_TAG="$(date +%Y%m%d_%H%M%S)"
readonly MOTOR_LOG="${ROOT}/logs/replay_motor_${SESSION_TAG}.log"
readonly REPLAY_LOG="${ROOT}/logs/replay_client_${SESSION_TAG}.log"

replay_client_args=(
  --raw-root "${RAW_ROOT}"
  --episode-index "${EPISODE_INDEX}"
  --profile "${DP3_PROFILE}"
  --host "${DP3_CONTROL_HOST}"
  --action-port "${DP3_ACTION_PORT}"
  --state-port "${DP3_STATE_PORT}"
  --state-timeout "${DP3_COMMAND_STALE_TIMEOUT_S}"
  --maximum-initial-delta "${DP3_MAXIMUM_INITIAL_ZERO_DELTA_RAD}"
  --start-max-step "${DP3_ZERO_MAX_ARM_STEP_RAD}"
  --start-tolerance "${DP3_ZERO_TOLERANCE_RAD}"
  --start-stable-frames "${DP3_ZERO_STABLE_FRAMES}"
  --start-timeout "${DP3_ZERO_TIMEOUT_S}"
  --maximum-tracking-error "${DP3_MAXIMUM_TRACKING_ERROR_RAD}"
  --tracking-error-frames "${DP3_TRACKING_ERROR_FRAMES}"
  --final-hold-seconds "${DP3_REPLAY_FINAL_HOLD_S:-1.0}"
  --log-file "${REPLAY_LOG}"
)

required_deltas="$(
  "${OPENARM_DP3_PYTHON}" -m openarm_dp3.replay_demonstration \
    "${replay_client_args[@]}" --print-required-deltas
)"
read -r required_arm_delta required_gripper_delta <<< "${required_deltas}"
if [[ -z "${required_arm_delta}" || -z "${required_gripper_delta}" ]]; then
  echo "Failed to derive replay rate limits from ${EPISODE_PATH}." >&2
  exit 7
fi
readonly REPLAY_MAX_ARM_DELTA_RAD="${DP3_REPLAY_MAX_ARM_DELTA_RAD:-${required_arm_delta}}"
readonly REPLAY_MAX_GRIPPER_DELTA_RAD="${DP3_REPLAY_MAX_GRIPPER_DELTA_RAD:-${required_gripper_delta}}"

motor_pid=""
cleanup() {
  if [[ -n "${motor_pid}" ]] && kill -0 "${motor_pid}" 2>/dev/null; then
    kill -TERM "${motor_pid}" 2>/dev/null || true
    wait "${motor_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "ACTIVE RAW DEMONSTRATION REPLAY: episode ${EPISODE_INDEX}"
echo "Episode: ${EPISODE_PATH}"
echo "No policy inference and no camera input will be used."
echo "Raw-target limits: arm=${REPLAY_MAX_ARM_DELTA_RAD}rad/frame gripper=${REPLAY_MAX_GRIPPER_DELTA_RAD}rad/frame"
echo "Motor log: ${MOTOR_LOG}"
echo "Replay log: ${REPLAY_LOG}"

env PYTHONPATH="${ROOT}/src" /usr/bin/python3 "${ROOT}/rollout/openarm_dp3_motor_server.py" \
  --left-can "${OPENARM_LEFT_CAN}" \
  --right-can "${OPENARM_RIGHT_CAN}" \
  --host "${DP3_CONTROL_HOST}" \
  --action-port "${DP3_ACTION_PORT}" \
  --state-port "${DP3_STATE_PORT}" \
  --fps "${DP3_FPS}" \
  --stale-timeout "${DP3_COMMAND_STALE_TIMEOUT_S}" \
  --max-arm-delta "${REPLAY_MAX_ARM_DELTA_RAD}" \
  --max-gripper-delta "${REPLAY_MAX_GRIPPER_DELTA_RAD}" \
  --stats-json "${DP3_PROFILE}" \
  --feedback-timeout "${DP3_FEEDBACK_TIMEOUT_S}" \
  --feedback-margin "${DP3_FEEDBACK_MARGIN_RAD}" \
  --feedback-settle "${DP3_FEEDBACK_SETTLE_S}" \
  --log-file "${MOTOR_LOG}" &
motor_pid=$!

"${OPENARM_DP3_PYTHON}" -m openarm_dp3.replay_demonstration "${replay_client_args[@]}"

wait "${motor_pid}"
motor_pid=""
