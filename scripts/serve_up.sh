#!/usr/bin/env bash
# One command to bring vLLM up on this host: verify the model is complete,
# start the container (pinned to GPU_ID, default 0), and wait for it to report
# healthy. Model weights and the vLLM image are assumed to already be in place
# (serve/models/<name>/ on disk, image already in your registry) — this script
# never downloads or pulls a model.
#
#   ./scripts/serve_up.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

# Detect which Compose CLI is available: v2 plugin (`docker compose`) or the
# standalone v1 binary (`docker-compose`). Some minimal/managed Docker installs
# (e.g. a bare vscode-server pod) ship only one of the two.
if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "!! Neither 'docker compose' (v2 plugin) nor 'docker-compose' (v1) is available."
  echo "   Install the Compose plugin: https://docs.docker.com/compose/install/linux/"
  exit 1
fi

./scripts/01_check_model.sh

echo
echo ">> Starting vLLM (serve/docker-compose.yml, GPU_ID=${GPU_ID:-0 (default)}) via '${COMPOSE[*]}' ..."
( cd serve && "${COMPOSE[@]}" up -d vllm )

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
echo "     cd serve && ${COMPOSE[*]} logs -f vllm"
exit 1
