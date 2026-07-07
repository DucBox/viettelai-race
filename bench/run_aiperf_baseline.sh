#!/usr/bin/env bash
# Drive the vLLM endpoint with AIPerf and print a TTFT / TPOT / throughput report.
# This is the CLIENT-SIDE view. Watch scripts/03_watch_metrics.sh (or Grafana)
# at the same time for the SERVER-SIDE view (KV cache, prefix hit rate).
#
#   source bench/.venv/bin/activate
#   ./bench/run_aiperf_baseline.sh              # light smoke run
#   MODE=trace ./bench/run_aiperf_baseline.sh   # competition-like shape
#
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f serve/.env ]]; then set -a; source serve/.env; set +a; fi

URL="${URL:-http://localhost:8000}"
MODEL="${SERVED_MODEL_NAME:-qwen3.5-2b}"
# Local model dir (same one vLLM serves from) — avoids any network fetch for
# tokenizer files; AIPerf's --tokenizer accepts a filesystem path directly.
# MODEL_DIR in serve/.env is relative to serve/ — normalize to repo root.
_MODEL_DIR="${MODEL_DIR:-./models/qwen3.5-2b}"
case "$_MODEL_DIR" in
  /*|serve/*) : ;;
  ./*) _MODEL_DIR="serve/${_MODEL_DIR#./}" ;;
  *) _MODEL_DIR="serve/$_MODEL_DIR" ;;
esac
TOKENIZER="$_MODEL_DIR"
MODE="${MODE:-smoke}"

if [[ "$MODE" == "trace" ]]; then
  # Approximate the real workload: 20 parallel sessions, long prefill (~15k),
  # short output (200). This is a SHAPE approximation, not the exact trace —
  # exact-timestamp replay of trace-round1.jsonl comes in a later step.
  CONCURRENCY="${CONCURRENCY:-20}"
  REQUEST_COUNT="${REQUEST_COUNT:-120}"
  ISL="${ISL:-15000}"   # input sequence length (tokens)
  OSL="${OSL:-200}"     # output sequence length (max_tokens is fixed 200 in trace)
else
  # Fast sanity run just to prove the AIPerf -> vLLM -> report loop works.
  CONCURRENCY="${CONCURRENCY:-4}"
  REQUEST_COUNT="${REQUEST_COUNT:-20}"
  ISL="${ISL:-1024}"
  OSL="${OSL:-128}"
fi

echo ">> AIPerf [$MODE]  concurrency=$CONCURRENCY count=$REQUEST_COUNT isl=$ISL osl=$OSL"
echo ">> target: $URL  model: $MODEL"
echo

# Note: AIPerf auto-collects vLLM /metrics (KV cache usage, request queue, prefix
# cache / prompt-token source mix, generation throughput) every 333ms by default —
# no extra flag needed. Look for server_metrics_export.* in the artifact dir.
aiperf profile \
  --model "$MODEL" \
  --url "$URL" \
  --endpoint-type chat \
  --endpoint /v1/chat/completions \
  --streaming \
  --tokenizer "$TOKENIZER" \
  --concurrency "$CONCURRENCY" \
  --request-count "$REQUEST_COUNT" \
  --synthetic-input-tokens-mean "$ISL" \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean "$OSL" \
  --output-tokens-stddev 0 \
  --extra-inputs "max_tokens:$OSL" \
  --extra-inputs "temperature:0" \
  --extra-inputs "ignore_eos:true"

echo
echo ">> Client report + server_metrics_export.* saved under ./artifacts/ (AIPerf default)."
echo ">> Later, to replay the REAL competition trace by timestamp, use:"
echo "     aiperf profile --model $MODEL --url $URL --endpoint-type chat --streaming \\"
echo "       --tokenizer $TOKENIZER \\"
echo "       --input-file trace-round1.jsonl --custom-dataset-type mooncake_trace --fixed-schedule"
