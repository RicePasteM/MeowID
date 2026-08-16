#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

exec "$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  tools/train.py \
  --config configs/experiments/meowid_base.yaml \
  "$@"
