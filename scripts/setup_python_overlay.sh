#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/mh/openarm_dp3"
ACT_ROOT="/home/mh/openarm_act"
PYTHON="${ACT_ROOT}/.venv/bin/python"
UV="${ACT_ROOT}/tools/uv"
TARGET="${ROOT}/.python-packages"

[[ -x "${PYTHON}" ]] || { echo "Missing existing Python: ${PYTHON}" >&2; exit 2; }
[[ -x "${UV}" ]] || { echo "Missing uv: ${UV}" >&2; exit 2; }

mkdir -p "${TARGET}"
"${UV}" pip install \
  --python "${PYTHON}" \
  --target "${TARGET}" \
  --requirements "${ROOT}/requirements-overlay.txt"

source "${ROOT}/scripts/env.sh"
"${OPENARM_DP3_PYTHON}" - <<'PY'
import diffusers
import hydra
import numba
import numcodecs
import omegaconf
import zarr

print("DP3 overlay ready")
print("zarr", zarr.__version__)
print("numba", numba.__version__)
print("diffusers", diffusers.__version__)
print("hydra", hydra.__version__)
print("omegaconf", omegaconf.__version__)
print("numcodecs", numcodecs.__version__)
PY
