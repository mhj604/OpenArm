#!/usr/bin/env bash

set -euo pipefail
source /home/mh/openarm_dp3/scripts/env.sh

readonly DATASET="/home/mh/openarm_dp3/data/processed/openarm_cube_to_point_zero_start.zarr"
readonly RUN_TAG="$(date +%Y%m%d_%H%M%S)"
readonly OUTPUT_DIR="${DP3_ZERO_START_OUTPUT_DIR:-/home/mh/openarm_dp3/outputs/zero_start_retrain_${RUN_TAG}}"

[[ -d "${DATASET}" ]] || {
  echo "Missing zero-start DP3 dataset: ${DATASET}" >&2
  echo "Run /home/mh/openarm_dp3/scripts/augment_zero_start_dataset.sh first." >&2
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

echo "Zero-start dataset: ${DATASET}"
echo "Training output: ${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "${DP3_SOURCE_ROOT}"
exec "${OPENARM_DP3_PYTHON}" train.py \
  --config-name=simple_dp3_openarm.yaml \
  task.dataset.zarr_path="${DATASET}" \
  name=train_simple_dp3_openarm_zero_start \
  exp_name=openarm_cube_to_point_zero_start_simple_dp3 \
  training.num_epochs="${DP3_ZERO_START_EPOCHS:-200}" \
  training.max_train_steps=null \
  training.checkpoint_every="${DP3_ZERO_START_CHECKPOINT_EVERY:-10}" \
  training.resume="${DP3_ZERO_START_RESUME:-false}" \
  hydra.run.dir="${OUTPUT_DIR}"
