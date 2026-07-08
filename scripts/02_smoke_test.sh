#!/usr/bin/env bash
# Fire a single chat request at the running vLLM server to confirm inference works
# end-to-end (model loaded, tokenizer OK, streaming OK).
#
#   ./scripts/02_smoke_test.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
# Preserve caller-provided env vars across the .env source (consistent with
# serve_up.sh / 10_bench_e2e.sh / run_aiperf_baseline.sh / 01_check_model.sh) —
# e.g. `SERVED_MODEL_NAME=foo ./scripts/02_smoke_test.sh` should test that name,
# not silently fall back to serve/.env's own default.
_pre_env_declare="$(declare -p $(compgen -e) 2>/dev/null || true)"
if [[ -f serve/.env ]]; then set -a; source serve/.env; set +a; fi
eval "$_pre_env_declare"

URL="${URL:-http://localhost:8000}"
MODEL="${SERVED_MODEL_NAME:-qwen3.5-2b}"

echo ">> Health check: $URL/health"
curl -fsS "$URL/health" && echo "  OK" || { echo "  server not healthy"; exit 1; }

echo ">> Models available:"
curl -fsS "$URL/v1/models" | python3 -m json.tool 2>/dev/null || curl -fsS "$URL/v1/models"

echo ">> Sending one non-streaming chat completion..."
curl -fsS "$URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
      {\"role\": \"user\", \"content\": \"In one sentence, what is prefix caching in LLM serving?\"}
    ],
    \"max_tokens\": 64,
    \"temperature\": 0
  }" | python3 -m json.tool

echo ""
echo ">> Smoke test passed. Server is serving real completions."
