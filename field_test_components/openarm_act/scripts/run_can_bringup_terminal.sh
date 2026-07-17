#!/usr/bin/env bash

set -o pipefail

readonly LOG_FILE="/home/mh/openarm_act/logs/can_bringup_20260716.log"
: >"${LOG_FILE}"

echo "OpenArm CAN receive-only preparation"
echo "This sets can0/can1 to Classic CAN 1 Mbit/s; it sends no CAN command."
echo "Enter the sudo password for mh, then press Enter."
sudo /home/mh/openarm_act/scripts/bring_up_can_receive_only.sh 2>&1 | tee "${LOG_FILE}"
status="${PIPESTATUS[0]}"
echo "RESULT=${status}" | tee -a "${LOG_FILE}"
echo "You may close this window."
sleep 60
exit "${status}"
