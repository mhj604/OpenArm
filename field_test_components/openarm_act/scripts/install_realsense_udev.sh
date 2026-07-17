#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_RULE="/home/mh/openarm_act/vendor/librealsense-rsusb/config/99-realsense-libusb.rules"
readonly TARGET_RULE="/etc/udev/rules.d/99-realsense-libusb.rules"
readonly EXPECTED_SHA256="21dd6bc40a117c9bb9dc93d78c9f77f2f74fae9f9eba75aa4714b25939d7532a"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this helper through pkexec or sudo." >&2
  exit 1
fi

actual_sha256="$(sha256sum "${SOURCE_RULE}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "Refusing to install an unexpected RealSense rule file." >&2
  exit 1
fi

case "${1:-install}" in
  install)
    if [[ -e "${TARGET_RULE}" ]] && ! cmp -s "${SOURCE_RULE}" "${TARGET_RULE}"; then
      echo "Refusing to overwrite an existing different RealSense rule." >&2
      exit 1
    fi
    install -o root -g root -m 0644 "${SOURCE_RULE}" "${TARGET_RULE}"
    ;;
  uninstall)
    if [[ -e "${TARGET_RULE}" ]] && ! cmp -s "${SOURCE_RULE}" "${TARGET_RULE}"; then
      echo "Refusing to remove a RealSense rule not installed by this helper." >&2
      exit 1
    fi
    rm -f "${TARGET_RULE}"
    ;;
  *)
    echo "Usage: $0 [install|uninstall]" >&2
    exit 2
    ;;
esac

udevadm control --reload-rules
udevadm trigger --subsystem-match=usb --action=add
udevadm trigger --subsystem-match=iio --action=add || true

echo "RealSense udev action completed: ${1:-install}"
