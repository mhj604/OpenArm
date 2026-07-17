#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mh/openarm_imitation"
DP3="$ROOT/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy"
BOOTSTRAP="$ROOT/.virtualenv-bootstrap"
if [[ ! -d "$BOOTSTRAP/virtualenv" ]]; then
  python3 -m pip install --target "$BOOTSTRAP" virtualenv==20.31.2
fi
PYTHONPATH="$BOOTSTRAP" python3 -m virtualenv "$ROOT/.venv-dp3"
PY="$ROOT/.venv-dp3/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
"$PY" -m pip install -r "$ROOT/requirements-dp3.txt"
"$PY" -m pip install --no-deps -e "$DP3"
"$PY" -m pip install --no-deps -e "$ROOT"
"$PY" -c "import torch; import diffusion_policy_3d; import openarm_imitation; print('dp3 environment OK; cuda=', torch.cuda.is_available(), 'gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
