#!/usr/bin/env bash
# Bring up the two existing SocketCAN interfaces without transmitting CAN
# frames. This only applies the documented Classic CAN 1 Mbit/s link
# configuration. It does not import OpenArm code, start ROS, enable motors, or
# open a CAN transmit socket. Run as root because ip-link needs CAP_NET_ADMIN.

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script through sudo." >&2
  exit 2
fi

readonly BITRATE=1000000
readonly INTERFACES=(can0 can1)

for interface in "${INTERFACES[@]}"; do
  if [[ ! "${interface}" =~ ^can[0-9]+$ ]] || [[ ! -d "/sys/class/net/${interface}" ]]; then
    echo "Missing expected SocketCAN interface: ${interface}" >&2
    exit 2
  fi
done

for interface in "${INTERFACES[@]}"; do
  before_tx="$(<"/sys/class/net/${interface}/statistics/tx_packets")"
  ip link set "${interface}" down
  ip link set "${interface}" type can bitrate "${BITRATE}"
  ip link set "${interface}" up
  after_tx="$(<"/sys/class/net/${interface}/statistics/tx_packets")"
  if [[ "${after_tx}" != "${before_tx}" ]]; then
    echo "Unexpected TX packet counter change on ${interface}: ${before_tx} -> ${after_tx}" >&2
    exit 3
  fi
  echo "PASS ${interface}: Classic CAN ${BITRATE} bit/s, application TX packets unchanged (${after_tx})"
  ip -details link show "${interface}"
done
