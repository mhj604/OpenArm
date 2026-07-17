#!/usr/bin/env bash

set -euo pipefail
export LEROBOT_NUM_EPISODES=1
exec /home/mh/openarm_act/scripts/record_episodes.sh
