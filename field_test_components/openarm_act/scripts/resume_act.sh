#!/usr/bin/env bash

set -euo pipefail

source /home/mh/openarm_act/scripts/env.sh
: "${ACT_TRAIN_CONFIG:?Set ACT_TRAIN_CONFIG to a checkpoint train_config.json}"

exec lerobot-train \
  --config_path="${ACT_TRAIN_CONFIG}" \
  --resume=true
