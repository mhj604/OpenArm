#!/usr/bin/env bash
set -euo pipefail

readonly DEB="/home/mh/openarm_act/downloads/realsense-driver/librealsense2-dkms_1.3.31-0ubuntu1_all.deb"
readonly EXPECTED_SHA256="19e1285fcbd9ae344fbacc68aa57d7aabd354b551c50d4062ff0f4241cb976b5"
readonly LOG="/home/mh/openarm_act/logs/realsense_dkms_install_20260716.log"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this helper through pkexec or sudo." >&2
  exit 1
fi

actual_sha256="$(sha256sum "${DEB}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "Refusing to install a DKMS package with an unexpected checksum." >&2
  exit 1
fi

exec > >(tee -a "${LOG}") 2>&1
date --iso-8601=seconds
echo "Installing Ubuntu DKMS dependency and verified RealSense DKMS package."
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends dkms "${DEB}"
depmod -a

echo "--- DKMS status ---"
dkms status
echo "--- Module candidates ---"
modinfo uvcvideo | grep -E '^(filename|version|srcversion|vermagic):'
modinfo videodev | grep -E '^(filename|version|srcversion|vermagic):'
