#!/usr/bin/env bash

set -euo pipefail

readonly ROOT="/home/mh/openarm_act"
source "${ROOT}/scripts/env.sh"

set -a
source "${ROOT}/configs/act_cube_clean10.env"
set +a

python - <<'PY'
import sys

import torch

if not torch.cuda.is_available():
    print(
        "CUDA is unavailable; refusing to silently run ACT on CPU. "
        "Check nvidia-smi and the loaded NVIDIA kernel module.",
        file=sys.stderr,
    )
    raise SystemExit(2)
print(f"CUDA PASS: {torch.cuda.get_device_name(0)}")
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"GPU memory: {free_bytes / 2**30:.2f}/{total_bytes / 2**30:.2f} GiB free")
PY

export ACT_OUTPUT_DIR="${ACT_OUTPUT_DIR:-${ROOT}/outputs/act_cube_clean10_$(date +%Y%m%d_%H%M%S)}"
echo "Episodes: ${LEROBOT_DATASET_EPISODES}"
echo "Output: ${ACT_OUTPUT_DIR}"

exec "${ROOT}/scripts/train_act.sh"
