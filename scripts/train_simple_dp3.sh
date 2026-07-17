#!/usr/bin/env bash

set -euo pipefail
source /home/mh/openarm_dp3/scripts/env.sh

DATASET="/home/mh/openarm_dp3/data/processed/openarm_cube_to_point.zarr"
[[ -d "${DATASET}" ]] || {
  echo "Missing processed DP3 dataset: ${DATASET}" >&2
  echo "Run /home/mh/openarm_dp3/scripts/convert_dataset.sh first." >&2
  exit 2
}
if ! "${OPENARM_DP3_PYTHON}" -c 'import zarr, hydra, diffusers, numba' >/dev/null 2>&1; then
  /home/mh/openarm_dp3/scripts/setup_python_overlay.sh
fi
if ! "${OPENARM_DP3_PYTHON}" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("CUDA is unavailable to PyTorch", file=sys.stderr)
    raise SystemExit(1)
print(torch.cuda.get_device_name(0))
print(f"VRAM bytes: {torch.cuda.get_device_properties(0).total_memory}")
PY
then
  echo "GPU device nodes may need: /sbin/ub-device-create --verbose" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "${DP3_SOURCE_ROOT}"
exec "${OPENARM_DP3_PYTHON}" train.py \
  --config-name=simple_dp3_openarm.yaml \
  "$@"
