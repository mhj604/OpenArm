#!/usr/bin/env bash

set -euo pipefail
export LEROBOT_NUM_EPISODES=5
exec /home/mh/openarm_act/scripts/record_episodes.sh
