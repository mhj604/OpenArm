#!/usr/bin/env bash
set -euo pipefail

readonly LOG="/home/mh/openarm_act/logs/realsense_dkms_reload_20260716.log"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this helper through pkexec or sudo." >&2
  exit 1
fi

exec > >(tee -a "${LOG}") 2>&1
date --iso-8601=seconds

declare -A parent_set=()
shopt -s nullglob
for binding in /sys/bus/usb/drivers/uvcvideo/*:*; do
  interface="$(basename "${binding}")"
  parent_set["${interface%%:*}"]=1
done
parents=("${!parent_set[@]}")

if [[ "${#parents[@]}" -eq 0 ]]; then
  echo "No UVC USB parent devices were found." >&2
  exit 1
fi

restore_devices() {
  modprobe uvcvideo 2>/dev/null || true
  for parent in "${parents[@]}"; do
    if [[ -e "/sys/bus/usb/devices/${parent}" && ! -L "/sys/bus/usb/devices/${parent}/driver" ]]; then
      printf '%s' "${parent}" >/sys/bus/usb/drivers/usb/bind 2>/dev/null || true
    fi
  done
  udevadm settle 2>/dev/null || true
}
trap restore_devices EXIT

printf 'Temporarily detaching UVC USB parents:'
printf ' %s' "${parents[@]}"
echo
for parent in "${parents[@]}"; do
  if [[ -L "/sys/bus/usb/devices/${parent}/driver" ]]; then
    printf '%s' "${parent}" >/sys/bus/usb/drivers/usb/unbind
  fi
done

modprobe -r uvcvideo
for module in videobuf2_v4l2 videobuf2_vmalloc videobuf2_common videobuf2_memops videodev mc uvc; do
  if [[ -d "/sys/module/${module}" ]]; then
    modprobe -r "${module}" 2>/dev/null || true
  fi
done

modprobe uvcvideo
restore_devices
trap - EXIT

echo "--- Loaded module versions ---"
printf 'uvcvideo='; cat /sys/module/uvcvideo/version
printf 'videodev='; cat /sys/module/videodev/version
printf 'uvc='; cat /sys/module/uvc/version
echo "--- Loaded module paths ---"
modinfo uvcvideo | grep '^filename:'
modinfo videodev | grep '^filename:'
modinfo uvc | grep '^filename:'
