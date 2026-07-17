#!/usr/bin/env bash

set -euo pipefail

OPENARM_ACT_ROOT="/home/mh/openarm_act"
source "${OPENARM_ACT_ROOT}/scripts/env.sh"

: "${LEROBOT_DATASET_REPO_ID:?Set the recorded dataset repo_id}"
: "${LEROBOT_DATASET_ROOT:?Set the concrete local dataset root}"

TRAIN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${ACT_OUTPUT_DIR:-${OPENARM_ACT_ROOT}/outputs/act_${TRAIN_TAG}}"

TRAIN_ARGS=(
  --dataset.repo_id="${LEROBOT_DATASET_REPO_ID}"
  --dataset.root="${LEROBOT_DATASET_ROOT}"
  --policy.type=act
  --policy.device="${ACT_DEVICE:-cuda}"
  --policy.push_to_hub=false
  --output_dir="${OUTPUT_DIR}"
  --job_name="${ACT_JOB_NAME:-openarm_v1_bimanual_act}"
  --batch_size="${ACT_BATCH_SIZE:-1}"
  --num_workers="${ACT_NUM_WORKERS:-2}"
  --steps="${ACT_STEPS:-100000}"
  --save_freq="${ACT_SAVE_FREQ:-10000}"
  --log_freq="${ACT_LOG_FREQ:-100}"
  --env_eval_freq=0
  --eval_steps=0
  --save_checkpoint=true
  --save_checkpoint_to_hub=false
  --wandb.enable="${WANDB_ENABLE:-false}"
)

if [[ -n "${LEROBOT_DATASET_EPISODES:-}" ]]; then
  TRAIN_ARGS+=(--dataset.episodes="${LEROBOT_DATASET_EPISODES}")
fi

exec lerobot-train "${TRAIN_ARGS[@]}"
