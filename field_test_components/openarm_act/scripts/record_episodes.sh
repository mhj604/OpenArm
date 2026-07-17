#!/usr/bin/env bash

set -euo pipefail

OPENARM_ACT_ROOT="/home/mh/openarm_act"
HARDWARE_ENV="${HARDWARE_ENV:-${OPENARM_ACT_ROOT}/hardware.env}"
if [[ ! -f "${HARDWARE_ENV}" ]]; then
  echo "Missing ${HARDWARE_ENV}; copy configs/hardware.env.example and fill physical values" >&2
  exit 2
fi
set -a
source "${HARDWARE_ENV}"
set +a
source "${OPENARM_ACT_ROOT}/scripts/env.sh"
: "${CHEST_SERIAL:?Set the D435ICB chest camera serial}"
: "${WRIST_LEFT_SERIAL:?Set the D405 left-wrist camera serial}"
: "${LEROBOT_DATASET_REPO_ID:?Set a local dataset repo_id}"
: "${LEROBOT_TASK:?Set the task description}"

if [[ ! "${CHEST_SERIAL}" =~ ^[0-9]+$ ]] || [[ ! "${WRIST_LEFT_SERIAL}" =~ ^[0-9]+$ ]]; then
  echo "RealSense serials must be numeric" >&2
  exit 4
fi
NUM_EPISODES="${LEROBOT_NUM_EPISODES:-1}"
EPISODE_TIME_S="${LEROBOT_EPISODE_TIME_S:-30}"
RESET_TIME_S="${LEROBOT_RESET_TIME_S:-30}"
SESSION_TAG="$(date +%Y%m%d_%H%M%S)"
DATASET_ROOT="${LEROBOT_DATASET_ROOT:-${OPENARM_ACT_ROOT}/data/recordings/${SESSION_TAG}}"

CAMERAS="{chest: {type: intelrealsense, serial_number_or_name: '${CHEST_SERIAL}', width: 640, height: 480, fps: 30, warmup_s: 2, use_rgb: true, use_depth: false}, wrist_left: {type: intelrealsense, serial_number_or_name: '${WRIST_LEFT_SERIAL}', width: 640, height: 480, fps: 30, warmup_s: 2, use_rgb: true, use_depth: false}}"

exec lerobot-record \
  --robot.type=openarm_can_passive \
  --robot.id=openarm_v1_bimanual_passive \
  --robot.broker_id=openarm_v1_bimanual \
  --robot.telemetry_host="${OPENARM_TELEMETRY_HOST:-127.0.0.1}" \
  --robot.telemetry_port="${OPENARM_TELEMETRY_PORT:-29502}" \
  --robot.stale_timeout_s="${OPENARM_TELEMETRY_STALE_TIMEOUT_S:-0.1}" \
  --robot.cameras="${CAMERAS}" \
  --teleop.type=openarm_can_action_tap \
  --teleop.id=openarm_existing_bridge_action \
  --teleop.broker_id=openarm_v1_bimanual \
  --dataset.repo_id="${LEROBOT_DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.single_task="${LEROBOT_TASK}" \
  --dataset.fps=30 \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.video=true \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=false \
  --play_sounds=false
