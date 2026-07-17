#!/usr/bin/env bash

set -euo pipefail

readonly TARGET_KERNEL="6.8.0-106-generic"
readonly SUBMENU_ID="gnulinux-advanced-7758a159-ab80-4b05-b244-72c9d30ae77d"
readonly ENTRY_ID="gnulinux-6.8.0-106-generic-advanced-7758a159-ab80-4b05-b244-72c9d30ae77d"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script through sudo." >&2
  exit 2
fi

if pgrep -af 'exo_to_openarm\.py|lerobot-record|lerobot-train|openarm_can_telemetry_udp\.py' >/dev/null; then
  echo "Refusing to reboot while a robot or LeRobot process is active:" >&2
  pgrep -af 'exo_to_openarm\.py|lerobot-record|lerobot-train|openarm_can_telemetry_udp\.py' >&2
  exit 3
fi

if ! modinfo -k "${TARGET_KERNEL}" nvidia >/dev/null 2>&1; then
  echo "NVIDIA module is missing for ${TARGET_KERNEL}." >&2
  exit 4
fi

sync
grub-reboot "${SUBMENU_ID}>${ENTRY_ID}"

if ! grub-editenv list | grep -Fxq "next_entry=${SUBMENU_ID}>${ENTRY_ID}"; then
  echo "Failed to verify the one-time GRUB entry." >&2
  exit 5
fi

echo "One-time boot target verified: ${TARGET_KERNEL}"
echo "Rebooting in 5 seconds..."
sleep 5
systemctl reboot
