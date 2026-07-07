#!/usr/bin/env bash
# One command to bring vLLM up on this host: verify the model is complete,
# start the container (pinned to GPU_ID, default 0), and wait for it to report
# healthy. Model weights and the vLLM image are assumed to already be in place
# (serve/models/<name>/ on disk, image already in your registry) — this script
# never downloads or pulls a model.
#
#   ./scripts/serve_up.sh
#
# Uses `docker compose` (v2 plugin) or `docker-compose` (v1) if either is
# available; otherwise falls back to a plain `docker run` that reproduces
# serve/docker-compose.yml exactly (same image/env/volume/GPU pinning) — this
# keeps things working on locked-down hosts with no internet access to install
# the Compose plugin.
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/01_check_model.sh
echo

if [[ -f serve/.env ]]; then set -a; source serve/.env; set +a; fi
IMAGE="${IMAGE:-vllm/vllm-openai:v0.22.1}"
GPU_ID="${GPU_ID:-0}"
MODEL_DIR="${MODEL_DIR:-./models/qwen3.5-2b}"
MODEL_PATH="${MODEL_PATH:-/models/qwen3.5-2b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-2b}"
# MODEL_DIR in .env is relative to serve/ — resolve to an absolute host path
# (docker run needs one; compose handles relative paths itself).
case "$MODEL_DIR" in
  /*) MODEL_DIR_ABS="$MODEL_DIR" ;;
  *) MODEL_DIR_ABS="$(cd "serve/$MODEL_DIR" && pwd)" ;;
esac

USE_COMPOSE=""
if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose); USE_COMPOSE=1
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose); USE_COMPOSE=1
fi

if [[ -n "$USE_COMPOSE" ]]; then
  echo ">> Starting vLLM via '${COMPOSE[*]}' (GPU_ID=$GPU_ID) ..."
  ( cd serve && "${COMPOSE[@]}" up -d vllm )
  LOGS_CMD="cd serve && ${COMPOSE[*]} logs -f vllm"
else
  echo ">> No compose CLI found (v2 plugin or v1 binary) — using plain 'docker run' instead."
  echo ">> Starting vLLM via docker run (GPU_ID=$GPU_ID) ..."
  docker rm -f vllm-qwen35 >/dev/null 2>&1 || true
  docker run -d \
    --name vllm-qwen35 \
    --ipc=host \
    --gpus "\"device=$GPU_ID\"" \
    -p 8000:8000 \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_LOGGING_LEVEL=INFO \
    -v "$MODEL_DIR_ABS:$MODEL_PATH:ro" \
    "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enable-prefix-caching
  LOGS_CMD="docker logs -f vllm-qwen35"
fi

echo ">> Waiting for /health to report ready (model load + CUDA graph compile can take a few minutes) ..."
URL="http://localhost:8000"
for i in $(seq 1 120); do
  if curl -fsS "$URL/health" >/dev/null 2>&1; then
    echo ">> vLLM is UP after ~$((i*5))s. $URL"
    echo
    echo ">> Next steps:"
    echo "     ./scripts/02_smoke_test.sh          # confirm inference actually works"
    echo "     ./scripts/03_watch_metrics.sh        # live server metrics in another terminal"
    echo "     source bench/.venv/bin/activate && ./bench/run_aiperf_baseline.sh   # collect + display via AIPerf"
    exit 0
  fi
  sleep 5
done

echo "!! Not healthy after 10 minutes. Check logs:"
echo "     $LOGS_CMD"
exit 1
