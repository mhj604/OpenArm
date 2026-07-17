#!/usr/bin/env bash

set -euo pipefail

readonly ROOT="/home/mh/openarm_dp3"
readonly CONFIG="${DP3_ROLLOUT_CONFIG:-${ROOT}/configs/dp3_rollout.env}"

set -a
source "${CONFIG}"
set +a
source "${ROOT}/scripts/env.sh"

readonly SESSION_TAG="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${ROOT}/logs/dp3_learned_preview_${SESSION_TAG}.log"

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
  --device cuda \
  --selection-frames "${DP3_SELECTION_FRAMES}" \
  --start-centroid-frames "${DP3_START_CENTROID_FRAMES}" \
  --goal-centroid-frames "${DP3_GOAL_CENTROID_FRAMES}" \
  --minimum-yellow-points "${DP3_MINIMUM_YELLOW_POINTS}" \
  --start-region-margin "${DP3_START_REGION_MARGIN_M}" \
  --maximum-goal-distance "${DP3_MAXIMUM_GOAL_DISTANCE_M}" \
  --minimum-gripper-open-seconds "${DP3_MINIMUM_GRIPPER_OPEN_S}" \
  --inference-seed "${DP3_INFERENCE_SEED}" \
  --inference-samples "${DP3_INFERENCE_SAMPLES}" \
  --arm-action-gain "${DP3_ARM_ACTION_GAIN}" \
  --initial-ramp-seconds "${DP3_INITIAL_RAMP_S}" \
  --initial-action-alpha "${DP3_INITIAL_ACTION_ALPHA}" \
  --action-alpha "${DP3_ACTION_ALPHA}" \
  --initial-max-arm-step "${DP3_INITIAL_MAX_ARM_STEP_RAD}" \
  --policy-max-arm-step "${DP3_POLICY_MAX_ARM_STEP_RAD}" \
  --dry-run \
  --log-file "${LOG_FILE}" \
  "${horizon_args[@]}"
