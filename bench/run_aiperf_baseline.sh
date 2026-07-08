#!/usr/bin/env bash
# Drive the vLLM endpoint with AIPerf and print a TTFT / TPOT / throughput report.
# This is the CLIENT-SIDE view. Watch scripts/03_watch_metrics.sh (or Grafana)
# at the same time for the SERVER-SIDE view (KV cache, prefix hit rate).
#
#   source bench/.venv/bin/activate
#   ./bench/run_aiperf_baseline.sh              # light smoke run
#   MODE=trace    ./bench/run_aiperf_baseline.sh   # flat shape approximation (no shared prefix)
#   MODE=replay   ./bench/run_aiperf_baseline.sh   # THE REAL TRACE: replays data/trace-round1.jsonl
#                                                   # (actual competition requests) verbatim on the
#                                                   # trace's own fixed timestamp schedule. This is
#                                                   # the faithful benchmark — no synthetic content.
#                                                   # Run bench/convert_trace_to_aiperf.py first.
#
set -euo pipefail
cd "$(dirname "$0")/.."

# Preserve EVERY caller-provided env var (e.g. 11_multi_bench.sh's per-row
# `env SERVED_MODEL_NAME=... MODEL_DIR=... ...`, or 10_bench_e2e.sh forwarding
# them) across the .env source below. serve/.env ships REAL non-empty defaults
# for SERVED_MODEL_NAME, MODEL_DIR, GPU_MEM_UTIL, etc. — sourcing it is a plain
# assignment that unconditionally overwrites ANY of these the caller set. This
# is the same class of bug fixed in serve_up.sh / 10_bench_e2e.sh (see
# 6337c30 / c7cbca5): without this, the SERVER can be running one row's config
# (e.g. a different MODEL_DIR or SERVED_MODEL_NAME) while AIPerf silently
# benchmarks against serve/.env's own (different) default instead — wrong
# tokenizer, wrong --model, or both, with no error. Snapshotting every exported
# var beforehand and re-declaring them after restores exactly what the caller
# passed in, while untouched vars still pick up .env's value normally.
_pre_env_declare="$(declare -p $(compgen -e) 2>/dev/null || true)"
if [[ -f serve/.env ]]; then set -a; source serve/.env; set +a; fi
eval "$_pre_env_declare"

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

if [[ "$MODE" == "replay" ]]; then
  # THE REAL BENCHMARK. Replay the actual competition trace verbatim: each of the
  # 120 records is a complete request (full conversation history baked in) that
  # AIPerf fires at its own timestamp_ms via --fixed-schedule. No synthetic
  # content, no --concurrency (the trace's timestamps drive the arrival pattern
  # and thus the natural concurrency). max_tokens/temperature/seed travel inside
  # the converted file, so we don't force them here (and deliberately no
  # ignore_eos — the model stops at its real EOS, which is what gets scored).
  #
  # --use-server-token-count auto-enables stream_options.include_usage, so vLLM
  # returns usage (prompt_tokens + prompt_tokens_details.cached_tokens) per
  # request. That fills the report's in_tok and, crucially, the per-request
  # prefix-cache-hit columns (cache_rd / hit%) — AIPerf skips input tokenization
  # in mooncake_trace `messages` mode, so without this those columns are blank.
  TRACE_IN="${TRACE_IN:-data/trace-round1.jsonl}"
  REPLAY_FILE="${REPLAY_FILE:-data/trace-round1.aiperf.jsonl}"

  if [[ ! -f "$REPLAY_FILE" || "$TRACE_IN" -nt "$REPLAY_FILE" ]]; then
    echo ">> converting $TRACE_IN -> $REPLAY_FILE (AIPerf mooncake_trace format)"
    python3 bench/convert_trace_to_aiperf.py "$TRACE_IN" "$REPLAY_FILE"
    echo
  fi

  echo ">> AIPerf [replay]  file=$REPLAY_FILE  (fixed-schedule, real trace content)"
  echo ">> target: $URL  model: $MODEL"
  echo

  aiperf profile \
    --model "$MODEL" \
    --url "$URL" \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --tokenizer "$TOKENIZER" \
    --input-file "$REPLAY_FILE" \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule \
    --use-server-token-count \
    --server-metrics-formats json csv jsonl \
    --random-seed 42

  echo
  echo ">> Per-request breakdown (TTFT, TPOT, KV cache %, prefix cache) — run:"
  echo "     ./.venv/bin/python scripts/07_per_request_report.py"
  echo ">> (uses the most recent ./artifacts/*/profile_export.jsonl + server_metrics_export.jsonl)"
  exit 0
fi

if [[ "$MODE" == "trace" ]]; then
  # Approximate the real workload's SHAPE only: 20 parallel, long prefill
  # (~15k), short output (200) — but requests are INDEPENDENT (no shared
  # prefix), so prefix_cache_hit_rate will correctly read 0% here. Use
  # MODE=replay to exercise real prefix caching on the actual trace.
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
echo ">> To benchmark the REAL competition trace (verbatim requests, fixed schedule):"
echo "     MODE=replay ./bench/run_aiperf_baseline.sh"
