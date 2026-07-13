#!/bin/bash
# A/B: --api-server-count  (parallel hoa FRONTEND xu ly input) -> tan cong
# frontend_prep (~35% TTFT, serial-hoa duoi burst). Cung config: fp8-all seqs20
# batch3216. Fully-instrumented (loggers+residual+sched) + reconcile moi rep.
# Interleaved cold, cung session. Doc: frontend_prep per config.
set -u
PY=/venv/main/bin/python
OUT=/root/ab_apisrv
mkdir -p "$OUT"
source /root/env_pins.sh 2>/dev/null || true
SRV_PIN=${SRV_PIN:-}; CLI_PIN=${CLI_PIN:-}
COUNTS="${COUNTS:-1 8}"; REPS="${REPS:-3}"

kill_server() {
  pkill -15 -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 3
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  pkill -9 -f "multiprocessing.spawn" 2>/dev/null
  pkill -9 -f "resource_tracker" 2>/dev/null; sleep 2
}

$PY /root/patch_loggers.py apply
$PY /root/patch_residual_ts.py apply
$PY /root/patch_sched_trace.py apply

run_one() {
  local c=$1 rep=$2
  local tag="c${c}_rep${rep}"
  echo "=================== $tag ==================="
  kill_server
  export VLLM_LOGGING_LEVEL=INFO
  export RESIDUAL_TRACE=$OUT/${tag}_rests.jsonl
  export SCHED_TRACE=$OUT/${tag}_sched.jsonl
  : > "$RESIDUAL_TRACE"; : > "$SCHED_TRACE"
  nohup $SRV_PIN $PY -m vllm.entrypoints.openai.api_server \
    --model=/root/model --served-model-name=Qwen3.5-2B --host=0.0.0.0 --port=8000 \
    --max-model-len=48000 --gpu-memory-utilization=0.37 --tensor-parallel-size=1 \
    --enable-prefix-caching --language-model-only \
    --kv-cache-dtype=fp8 --calculate-kv-scales --quantization=fp8 \
    --max-num-seqs=20 --max-num-batched-tokens=3216 --gdn-prefill-backend=flashinfer \
    --api-server-count=$c \
    > "$OUT/${tag}_serve.log" 2>&1 &
  echo $! > /root/vllm.pid
  local up=0
  for i in $(seq 1 400); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)" = "200" ] \
      && { echo "[health] UP ${i}s"; up=1; break; }
    kill -0 "$(cat /root/vllm.pid 2>/dev/null)" 2>/dev/null || { echo "[FAIL proc chet]"; tail -30 "$OUT/${tag}_serve.log"; return 1; }
    sleep 1
  done
  [ "$up" = 1 ] || { echo "[health TIMEOUT]"; return 1; }
  $CLI_PIN $PY /root/warmup_stub.py > "$OUT/${tag}_warmup.txt" 2>&1
  sleep 2
  $CLI_PIN $PY /root/replay_trace_detailed.py > "$OUT/${tag}_replay.txt" 2>&1
  cp /root/replay_detailed_samples.json "$OUT/${tag}_samples.json" 2>/dev/null
  cp /root/replay_detailed_requests.json "$OUT/${tag}_requests.json" 2>/dev/null
  $PY /root/env_gate.py --samples "$OUT/${tag}_samples.json" || echo "  ⚠️ GATE FAIL $tag"
  $PY /root/merge_request_metrics.py --requests /root/replay_detailed_requests.json \
    --log "$OUT/${tag}_serve.log" --out "$OUT/${tag}_full.json" > "$OUT/${tag}_merge.txt" 2>&1
  $PY /root/reconcile_trace.py --full "$OUT/${tag}_full.json" \
    --rests "$OUT/${tag}_rests.jsonl" --sched "$OUT/${tag}_sched.jsonl" \
    --out "$OUT/${tag}_recon.json" > "$OUT/${tag}_recon.txt" 2>&1
  echo "[$tag] $(grep -E 'frontend_prep|TTFT [0-9]' "$OUT/${tag}_recon.txt" | head -2 | tr '\n' ' ')"
}

echo "########## A/B api-server-count [$COUNTS] x $REPS rep interleaved ##########"
for rep in $(seq 1 $REPS); do
  for c in $COUNTS; do run_one "$c" "$rep"; done
done
kill_server
echo "AB_APISRV_DONE -> $OUT"
