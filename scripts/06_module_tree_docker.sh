#!/usr/bin/env bash
# Print the real nn.Module tree of Qwen3.5-2B using a Python 3.11 Docker container
# (qwen3_5 needs transformers-from-source, which needs Python >=3.10; the Mac has
# 3.9). Mounts the local flat model directory read-only — fully offline, no pull.
#
#   ./scripts/06_module_tree_docker.sh
#
# First run installs torch(CPU)+transformers inside the container (~a few minutes);
# it uses a named volume for pip cache so re-runs are fast.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f serve/.env ]]; then set -a; source serve/.env; set +a; fi
MODEL_DIR="${MODEL_DIR:-serve/models/qwen3.5-2b}"
case "$MODEL_DIR" in
  /*|serve/*) : ;;
  ./*) MODEL_DIR="serve/${MODEL_DIR#./}" ;;
  *) MODEL_DIR="serve/$MODEL_DIR" ;;
esac

if [[ ! -s "$MODEL_DIR/config.json" ]]; then
  echo "!! $MODEL_DIR/config.json not found — run scripts/01_check_model.sh to see what's missing."; exit 1
fi

docker run --rm \
  -v "$PWD/$MODEL_DIR:/model:ro" \
  -v "$PWD/scripts:/scripts:ro" \
  -v aiperf_race_pip:/root/.cache/pip \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e MODEL_ID=/model \
  python:3.11-slim bash -c '
    set -e
    echo ">> installing torch (CPU) + transformers-from-source ..."
    pip install -q --index-url https://download.pytorch.org/whl/cpu torch
    # transformers main as a zip (no git needed in slim image); qwen3_5 is not in
    # any stable release yet.
    pip install -q "https://github.com/huggingface/transformers/archive/refs/heads/main.zip" accelerate
    echo ">> running module-tree inspector ..."
    python /scripts/06_module_tree.py
  '
