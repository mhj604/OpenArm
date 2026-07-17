#!/usr/bin/env bash

set -o pipefail

readonly LOG_FILE="/home/mh/openarm_act/logs/reboot_to_kernel_106_20260716.log"
: >"${LOG_FILE}"

echo "One-time reboot to Linux 6.8.0-106-generic"
echo "Enter the sudo password for mh, then press Enter."
sudo /home/mh/openarm_act/scripts/reboot_to_nvidia_kernel_106.sh 2>&1 | tee "${LOG_FILE}"
status="${PIPESTATUS[0]}"
echo "RESULT=${status}" | tee -a "${LOG_FILE}"
if [[ "${status}" -ne 0 ]]; then
  echo "Reboot was not started. Keep this window open for inspection."
  read -r
fi
exit "${status}"
