#!/usr/bin/env bash

set -euo pipefail

readonly ROOT="/home/mh/openarm_dp3"
readonly ACT_ROOT="/home/mh/openarm_act"
readonly CONFIG="${DP3_ROLLOUT_CONFIG:-${ROOT}/configs/dp3_rollout.env}"

if [[ "${OPENARM_DP3_EXECUTE:-}" != "YES" ]]; then
  echo "This command enables and moves both OpenArm arms." >&2
  echo "Run it only with explicit execution consent:" >&2
  echo "  OPENARM_DP3_EXECUTE=YES ${ROOT}/scripts/rollout_dp3.sh" >&2
  exit 2
fi

set -a
source "${CONFIG}"
set +a
source "${ROOT}/scripts/env.sh"

required_paths=(
  "${DP3_CHECKPOINT}"
  "${DP3_DATASET}"
  "${DP3_PROFILE}"
  "${DP3_ZERO_POSE_CONFIG}"
  "${DP3_HARDWARE_CONFIG}"
  "${DP3_COLLECTION_CONFIG}"
  "${DP3_WORKSPACE_CONFIG}"
  "${ROOT}/rollout/openarm_dp3_motor_server.py"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing active-rollout prerequisite: ${path}" >&2
    exit 3
  fi
done

if pgrep -f 'exo_to_openarm.py' >/dev/null; then
  echo "The ExoArm-to-OpenArm bridge is running. Stop it before DP3 rollout." >&2
  exit 4
fi
if pgrep -f 'openarm_.*motor_server.py' >/dev/null; then
  echo "Another OpenArm policy motor server is already running." >&2
  exit 5
fi

readonly SESSION_TAG="$(date +%Y%m%d_%H%M%S)"
readonly MOTOR_LOG="${ROOT}/logs/dp3_motor_${SESSION_TAG}.log"
readonly POLICY_LOG="${ROOT}/logs/dp3_policy_${SESSION_TAG}.log"

motor_pid=""
cleanup() {
  if [[ -n "${motor_pid}" ]] && kill -0 "${motor_pid}" 2>/dev/null; then
    kill -TERM "${motor_pid}" 2>/dev/null || true
    wait "${motor_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${DP3_CONTINUOUS}" == "1" ]]; then
  echo "ACTIVE LEARNED DP3: ZERO -> PICK/PLACE -> ZERO -> wait/repeat until Ctrl+C"
elif [[ "${DP3_DURATION_S}" == "0" || "${DP3_DURATION_S}" == "0.0" ]]; then
  echo "ACTIVE LEARNED DP3: one task with no timeout after typing START"
else
  echo "ACTIVE LEARNED DP3: one task with a ${DP3_DURATION_S}s timeout after typing START"
fi
echo "Checkpoint: ${DP3_CHECKPOINT}"
echo "Chest camera: D435ICB from ${DP3_HARDWARE_CONFIG}"
echo "Motor log: ${MOTOR_LOG}"
echo "Policy log: ${POLICY_LOG}"

env PYTHONPATH="${ROOT}/src" /usr/bin/python3 "${ROOT}/rollout/openarm_dp3_motor_server.py" \
  --left-can "${OPENARM_LEFT_CAN}" \
  --right-can "${OPENARM_RIGHT_CAN}" \
  --host "${DP3_CONTROL_HOST}" \
  --action-port "${DP3_ACTION_PORT}" \
  --state-port "${DP3_STATE_PORT}" \
  --fps "${DP3_FPS}" \
  --stale-timeout "${DP3_COMMAND_STALE_TIMEOUT_S}" \
  --max-arm-delta "${DP3_MAX_ARM_DELTA_RAD}" \
  --max-gripper-delta "${DP3_MAX_GRIPPER_DELTA_RAD}" \
  --stats-json "${DP3_PROFILE}" \
  --feedback-timeout "${DP3_FEEDBACK_TIMEOUT_S}" \
  --feedback-margin "${DP3_FEEDBACK_MARGIN_RAD}" \
  --feedback-settle "${DP3_FEEDBACK_SETTLE_S}" \
  --log-file "${MOTOR_LOG}" &
motor_pid=$!

continuous_args=()
if [[ "${DP3_CONTINUOUS}" == "1" ]]; then
  continuous_args+=(--continuous)
fi
horizon_args=()
if [[ "${DP3_EXECUTE_FULL_PREDICTION_HORIZON}" == "1" ]]; then
  horizon_args+=(--execute-full-horizon)
fi

"${OPENARM_DP3_PYTHON}" -m openarm_dp3.dp3_policy_client \
  --checkpoint "${DP3_CHECKPOINT}" \
  --dataset "${DP3_DATASET}" \
  --profile "${DP3_PROFILE}" \
  --zero-pose "${DP3_ZERO_POSE_CONFIG}" \
  --hardware "${DP3_HARDWARE_CONFIG}" \
  --collection "${DP3_COLLECTION_CONFIG}" \
  --workspace "${DP3_WORKSPACE_CONFIG}" \
  --host "${DP3_CONTROL_HOST}" \
  --action-port "${DP3_ACTION_PORT}" \
  --state-port "${DP3_STATE_PORT}" \
  --fps "${DP3_FPS}" \
  --duration "${DP3_DURATION_S}" \
  --device cuda \
  --state-timeout "${DP3_COMMAND_STALE_TIMEOUT_S}" \
  --selection-frames "${DP3_SELECTION_FRAMES}" \
  --start-centroid-frames "${DP3_START_CENTROID_FRAMES}" \
  --goal-centroid-frames "${DP3_GOAL_CENTROID_FRAMES}" \
  --minimum-yellow-points "${DP3_MINIMUM_YELLOW_POINTS}" \
  --start-region-margin "${DP3_START_REGION_MARGIN_M}" \
  --maximum-goal-distance "${DP3_MAXIMUM_GOAL_DISTANCE_M}" \
  --goal-minimum-elapsed "${DP3_GOAL_MINIMUM_ELAPSED_S}" \
  --goal-stable-frames "${DP3_GOAL_STABLE_FRAMES}" \
  --goal-open-fraction "${DP3_GOAL_OPEN_FRACTION}" \
  --minimum-gripper-open-seconds "${DP3_MINIMUM_GRIPPER_OPEN_S}" \
  --maximum-initial-zero-delta "${DP3_MAXIMUM_INITIAL_ZERO_DELTA_RAD}" \
  --zero-tolerance "${DP3_ZERO_TOLERANCE_RAD}" \
  --zero-stable-frames "${DP3_ZERO_STABLE_FRAMES}" \
  --zero-timeout "${DP3_ZERO_TIMEOUT_S}" \
  --zero-max-arm-step "${DP3_ZERO_MAX_ARM_STEP_RAD}" \
  --maximum-tracking-error "${DP3_MAXIMUM_TRACKING_ERROR_RAD}" \
  --tracking-error-frames "${DP3_TRACKING_ERROR_FRAMES}" \
  --restart-stable-frames "${DP3_RESTART_STABLE_FRAMES}" \
  --restart-cooldown-seconds "${DP3_RESTART_COOLDOWN_S}" \
  --restart-report-seconds "${DP3_RESTART_REPORT_S}" \
  --inference-seed "${DP3_INFERENCE_SEED}" \
  --inference-samples "${DP3_INFERENCE_SAMPLES}" \
  --arm-action-gain "${DP3_ARM_ACTION_GAIN}" \
  --initial-ramp-seconds "${DP3_INITIAL_RAMP_S}" \
  --initial-action-alpha "${DP3_INITIAL_ACTION_ALPHA}" \
  --action-alpha "${DP3_ACTION_ALPHA}" \
  --initial-max-arm-step "${DP3_INITIAL_MAX_ARM_STEP_RAD}" \
  --policy-max-arm-step "${DP3_POLICY_MAX_ARM_STEP_RAD}" \
  --log-file "${POLICY_LOG}" \
  "${horizon_args[@]}" \
  "${continuous_args[@]}"

wait "${motor_pid}"
motor_pid=""
