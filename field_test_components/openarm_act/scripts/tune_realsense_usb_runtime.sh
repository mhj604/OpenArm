#!/usr/bin/env bash
set -euo pipefail

readonly USBFS_MEMORY="/sys/module/usbcore/parameters/usbfs_memory_mb"
readonly D435_POWER="/sys/bus/usb/devices/3-1/power/control"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run through pkexec or sudo." >&2
  exit 1
fi

case "${1:-apply}" in
  apply)
    # Runtime-only values. Both revert at reboot; no firmware or persistent
    # kernel configuration is changed.
    echo 256 >"${USBFS_MEMORY}"
    echo on >"${D435_POWER}"
    ;;
  restore)
    echo 16 >"${USBFS_MEMORY}"
    echo auto >"${D435_POWER}"
    ;;
  *)
    echo "Usage: $0 [apply|restore]" >&2
    exit 2
    ;;
esac

printf 'usbfs_memory_mb=%s\n' "$(cat "${USBFS_MEMORY}")"
printf 'D435 power/control=%s\n' "$(cat "${D435_POWER}")"
