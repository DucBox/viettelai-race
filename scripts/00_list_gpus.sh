#!/usr/bin/env bash
# Optional: list GPUs on this host. Only needed if card 0 (the default) is busy
# and you want to pin vLLM to a different one via GPU_ID in serve/.env.
#
#   ./scripts/00_list_gpus.sh
#
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "!! nvidia-smi not found — is this a GPU host with NVIDIA drivers installed?"
  exit 1
fi

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo
echo ">> Defaults to GPU_ID=0. Override in serve/.env only if card 0 is busy."
