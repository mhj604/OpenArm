#!/usr/bin/env bash
set -o pipefail

readonly TERMINAL_LOG="/home/mh/openarm_act/logs/realsense_reload_terminal.log"
: >"${TERMINAL_LOG}"

echo "RealSense DKMS reload"
echo "Enter the sudo password for mh, then press Enter."
sudo bash /home/mh/openarm_act/scripts/reload_realsense_dkms.sh 2>&1 | tee "${TERMINAL_LOG}"
status="${PIPESTATUS[0]}"
echo "RESULT=${status}" | tee -a "${TERMINAL_LOG}"
echo "Keep this window open; Codex is checking the result."
sleep 300
exit "${status}"
