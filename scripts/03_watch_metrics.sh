#!/usr/bin/env bash
# Poll vLLM's /metrics and print the numbers that matter for this competition,
# in a refreshing view. No Prometheus/Grafana needed — quick eyeball tool.
#
#   ./scripts/03_watch_metrics.sh            # refresh every 2s
#   INTERVAL=1 ./scripts/03_watch_metrics.sh
#
# Run this in a second terminal WHILE AIPerf is hammering the server, so you can
# watch KV cache usage and prefix cache hit rate move under load.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f serve/.env ]]; then set -a; source serve/.env; set +a; fi

URL="${URL:-http://localhost:8000}"
INTERVAL="${INTERVAL:-2}"

# Metric substrings we care about. vLLM metric names are prefixed with "vllm:".
#   num_requests_running/waiting  → live queue depth (see the burst)
#   gpu_cache_usage_perc          → KV cache occupancy (0..1)
#   prefix_cache_queries/hits     → cache hit rate = hits/queries
#   prompt_tokens/generation_tokens → total processed
KEYS='num_requests_running|num_requests_waiting|gpu_cache_usage_perc|gpu_prefix_cache|prefix_cache_queries|prefix_cache_hits|prompt_tokens_total|generation_tokens_total|num_preemptions'

while true; do
  clear
  echo "=== vLLM /metrics @ $(date '+%H:%M:%S') ($URL) ==="
  echo "    (Ctrl-C to stop; refresh ${INTERVAL}s)"
  echo
  if ! raw="$(curl -fsS "$URL/metrics" 2>/dev/null)"; then
    echo "  !! cannot reach $URL/metrics — is the server up?"
  else
    echo "$raw" | grep -E "^vllm:($KEYS)" | grep -v '^#' | sort
    echo
    # Compute prefix cache hit rate from the two counters if present.
    q=$(echo "$raw" | grep -E '^vllm:.*prefix_cache_queries' | grep -v '^#' | awk '{s+=$2} END{print s+0}')
    h=$(echo "$raw" | grep -E '^vllm:.*prefix_cache_hits'    | grep -v '^#' | awk '{s+=$2} END{print s+0}')
    if [[ "${q:-0}" != "0" ]]; then
      echo "  >> prefix cache hit rate = hits/queries = $h / $q = $(python3 -c "print(f'{$h/$q*100:.1f}%')")"
    else
      echo "  >> prefix cache hit rate: no queries yet (send some requests first)"
    fi
  fi
  sleep "$INTERVAL"
done
