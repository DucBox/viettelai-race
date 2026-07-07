#!/usr/bin/env bash
# One command to bring vLLM up: verify the model is complete, start serving
# (pinned to GPU_ID, default 0), and wait for it to report healthy. Model
# weights are assumed to already be in place (serve/models/<name>/ on disk) —
# this script never downloads a model.
#
#   ./scripts/serve_up.sh
#
# Auto-detects which of two modes applies, no flags needed:
#
#   native — this shell is ALREADY inside an environment with the `vllm` CLI
#            and a GPU directly visible (e.g. a Kubernetes pod built FROM the
#            vllm-openai image, with GPU granted straight to the pod — no
#            nested Docker involved). Just runs `vllm serve` as a background
#            process; GPU pinning via CUDA_VISIBLE_DEVICES.
#
#   docker — otherwise, launches the vllm-openai image ourselves. Uses
#            `docker compose` (v2) / `docker-compose` (v1) if either is
#            available, else a plain `docker run` with a --gpus → CDI-failure
#            → --runtime=nvidia fallback ladder (for Docker-in-Docker hosts
#            with no CDI spec generated).
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
# MODEL_DIR in .env is relative to serve/ — resolve to an absolute path either
# way (needed for docker run's -v, and just as correct as the vllm serve arg
# in native mode).
case "$MODEL_DIR" in
  /*) MODEL_DIR_ABS="$MODEL_DIR" ;;
  *) MODEL_DIR_ABS="$(cd "serve/$MODEL_DIR" && pwd)" ;;
esac

if command -v vllm >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  MODE="native"
else
  MODE="docker"
fi
echo ">> Serving mode: $MODE"

if [[ "$MODE" == "native" ]]; then
  # Already running inside a container/pod with vllm + GPU present directly —
  # no Docker layer to manage. Pin the GPU the native way (CUDA_VISIBLE_DEVICES)
  # and just launch `vllm serve` as a background process.
  PIDFILE="serve/.vllm.pid"
  LOGFILE="serve/vllm.log"
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo ">> Stopping previous vllm serve (pid $(cat "$PIDFILE")) ..."
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    sleep 2
  fi

  echo ">> Starting 'vllm serve' natively (GPU_ID=$GPU_ID, CUDA_VISIBLE_DEVICES) ..."
  CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    nohup vllm serve "$MODEL_DIR_ABS" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --enable-prefix-caching \
      > "$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  LOGS_CMD="tail -f $LOGFILE"
else
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
    docker rm -f vllm-qwen35 >/dev/null 2>&1 || true

    RUN_COMMON=(-d --name vllm-qwen35 --ipc=host -p 8000:8000
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_LOGGING_LEVEL=INFO
      -v "$MODEL_DIR_ABS:$MODEL_PATH:ro")
    VLLM_ARGS=("$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME"
      --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL"
      --max-num-seqs "$MAX_NUM_SEQS" --enable-prefix-caching)

    # Attempt 1: modern `--gpus` flag (goes through CDI vendor discovery in
    # recent Docker versions). Known to fail with "failed to discover GPU
    # vendor from CDI" on Docker-in-Docker / Kubernetes-pod hosts where no CDI
    # spec (/etc/cdi/nvidia.yaml) has been generated, even though the host's
    # own nvidia-smi works fine. If that happens, fall back to the classic
    # --runtime=nvidia + NVIDIA_VISIBLE_DEVICES mechanism, which bypasses CDI.
    echo ">> Attempt 1/2: docker run --gpus (GPU_ID=$GPU_ID, CDI-based discovery) ..."
    GPU_ERR=""
    if ! GPU_ERR="$(docker run "${RUN_COMMON[@]}" --gpus "\"device=$GPU_ID\"" "$IMAGE" "${VLLM_ARGS[@]}" 2>&1 >/dev/null)"; then
      if echo "$GPU_ERR" | grep -qi "cdi\|gpu vendor"; then
        echo ">> '--gpus' failed via CDI vendor discovery (common on Docker-in-Docker / k8s pods)."
        echo ">> Attempt 2/2: classic --runtime=nvidia + NVIDIA_VISIBLE_DEVICES ..."
        docker rm -f vllm-qwen35 >/dev/null 2>&1 || true
        docker run "${RUN_COMMON[@]}" \
          --runtime=nvidia \
          -e NVIDIA_VISIBLE_DEVICES="$GPU_ID" \
          -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
          "$IMAGE" "${VLLM_ARGS[@]}"
      else
        echo "!! docker run failed (not a CDI/GPU-vendor issue) — full error:"
        echo "$GPU_ERR"
        exit 1
      fi
    fi
    LOGS_CMD="docker logs -f vllm-qwen35"
  fi
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
