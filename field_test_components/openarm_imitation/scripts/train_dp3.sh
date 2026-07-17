#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <task_config_name> [epochs]" >&2
  exit 2
fi

ROOT="/home/mh/openarm_imitation"
TASK="$1"
EPOCHS="${2:-1000}"
DP3="$ROOT/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy"

source "$ROOT/.venv-dp3/bin/activate"
export PYTHONPATH="$ROOT:$DP3:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
cd "$DP3"

# RTX 2050 4 GiB conservative defaults. Increase batch size only after checking VRAM.
exec python train.py --config-name=simple_dp3 \
  task="$TASK" \
  horizon=16 \
  n_action_steps=8 \
  dataloader.batch_size=4 \
  dataloader.num_workers=2 \
  val_dataloader.batch_size=4 \
  val_dataloader.num_workers=2 \
  training.device=cuda:0 \
  training.num_epochs="$EPOCHS" \
  logging.mode=offline \
  checkpoint.save_ckpt=True \
  checkpoint.save_last_ckpt=True \
  exp_name="$TASK"

