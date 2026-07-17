#!/usr/bin/env bash

set -euo pipefail

OPENARM_ACT_ROOT="/home/mh/openarm_act"
OPENARM_REALSENSE_RSUSB_PYTHON="${OPENARM_ACT_ROOT}/python"
if [[ ! -x "${OPENARM_ACT_ROOT}/.venv/bin/python" ]]; then
  echo "LeRobot environment is not installed: ${OPENARM_ACT_ROOT}/.venv" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -f "${OPENARM_REALSENSE_RSUSB_PYTHON}/pyrealsense2/__init__.py" ]]; then
  echo "Isolated RealSense RSUSB bindings are missing: ${OPENARM_REALSENSE_RSUSB_PYTHON}" >&2
  return 1 2>/dev/null || exit 1
fi

export OPENARM_ACT_ROOT
export OPENARM_REALSENSE_RSUSB_PYTHON
export VIRTUAL_ENV="${OPENARM_ACT_ROOT}/.venv"
export PATH="${VIRTUAL_ENV}/bin:${OPENARM_ACT_ROOT}/tools:${PATH}"
case ":${PYTHONPATH:-}:" in
  *":${OPENARM_REALSENSE_RSUSB_PYTHON}:"*) ;;
  *) export PYTHONPATH="${OPENARM_REALSENSE_RSUSB_PYTHON}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
export HF_LEROBOT_HOME="${OPENARM_ACT_ROOT}/data/huggingface/lerobot"
export HF_HOME="${OPENARM_ACT_ROOT}/data/huggingface"
export UV_CACHE_DIR="${OPENARM_ACT_ROOT}/.uv-cache"
export PYTHONUNBUFFERED=1

mkdir -p "${HF_LEROBOT_HOME}" "${OPENARM_ACT_ROOT}/logs" "${OPENARM_ACT_ROOT}/outputs"
