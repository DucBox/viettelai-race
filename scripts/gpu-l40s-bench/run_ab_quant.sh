#!/bin/bash
# A/B TACH QUANT — tra loi DUT DIEM "vi sao fp8 khong hon noquant TTFT" o muc
# component moi (residual/frontend_prep/queue/prefill/tpot + reconcile).
#   noquant : bf16 W + bf16 KV
#   fp8w    : fp8 W + bf16 KV   (co lap tac dong WEIGHT)
#   fp8all  : fp8 W + fp8 KV    (hien tai; them tac dong KV: block-align 1072)
# Cung seqs20 batch3216 mm-cache0. Fully-instrumented, interleaved cold, cung session.
# Ky vong: frontend_prep GIONG NHAU (hash doc lap quant); prefill fp8 thap hon;
#          queue fp8all cao hon (block-align). -> biet fp8 thua/thang TTFT o LA nao.
set -u
PY=/venv/main/bin/python
OUT=/root/ab_quant
mkdir -p "$OUT"
source /root/env_pins.sh 2>/dev/null || true
SRV_PIN=${SRV_PIN:-}; CLI_PIN=${CLI_PIN:-}
CONFIGS="${CONFIGS:-noquant fp8w fp8all}"; REPS="${REPS:-3}"
COMMON="--mm-processor-cache-gb=0"

extra_for() {
  case "$1" in
    noquant) echo "" ;;
    fp8w)    echo "--quantization=fp8" ;;
    fp8all)  echo "--quantization=fp8 --kv-cache-dtype=fp8 --calculate-kv-scales" ;;
  esac
}

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
  local cfg=$1 rep=$2 tag="${1}_rep${2}"
  local EXTRA; EXTRA=$(extra_for "$cfg")
  echo "=================== $tag  [$EXTRA] ==================="
  kill_server
  export VLLM_LOGGING_LEVEL=INFO
  export RESIDUAL_TRACE=$OUT/${tag}_rests.jsonl
  export SCHED_TRACE=$OUT/${tag}_sched.jsonl
  : > "$RESIDUAL_TRACE"; : > "$SCHED_TRACE"
  nohup $SRV_PIN $PY -m vllm.entrypoints.openai.api_server \
    --model=/root/model --served-model-name=Qwen3.5-2B --host=0.0.0.0 --port=8000 \
    --max-model-len=48000 --gpu-memory-utilization=0.37 --tensor-parallel-size=1 \
    --enable-prefix-caching --language-model-only \
    --max-num-seqs=20 --max-num-batched-tokens=3216 --gdn-prefill-backend=flashinfer \
    $COMMON $EXTRA > "$OUT/${tag}_serve.log" 2>&1 &
  echo $! > /root/vllm.pid
  local up=0
  for i in $(seq 1 400); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)" = "200" ] \
      && { echo "[health] UP ${i}s"; up=1; break; }
    kill -0 "$(cat /root/vllm.pid 2>/dev/null)" 2>/dev/null || { echo "[FAIL proc chet]"; grep -iE "error|oom|memory|traceback" "$OUT/${tag}_serve.log" | head -4; return 1; }
    sleep 1
  done
  [ "$up" = 1 ] || { echo "[TIMEOUT]"; return 1; }
  $CLI_PIN $PY /root/warmup_stub.py > "$OUT/${tag}_warmup.txt" 2>&1
  sleep 2
  $CLI_PIN $PY /root/replay_trace_detailed.py > "$OUT/${tag}_replay.txt" 2>&1
  cp /root/replay_detailed_samples.json "$OUT/${tag}_samples.json" 2>/dev/null
  cp /root/replay_detailed_requests.json "$OUT/${tag}_requests.json" 2>/dev/null
  # dem preemption (bf16 KV co the khong fit -> preempt)
  local npre; npre=$(grep -c "PREEMPTED\|preempted and moved" "$OUT/${tag}_serve.log" 2>/dev/null)
  echo "  preemptions=$npre"
  $PY /root/env_gate.py --samples "$OUT/${tag}_samples.json" || echo "  ⚠️ GATE FAIL $tag"
  $PY /root/merge_request_metrics.py --requests /root/replay_detailed_requests.json \
    --log "$OUT/${tag}_serve.log" --out "$OUT/${tag}_full.json" > "$OUT/${tag}_merge.txt" 2>&1
  $PY /root/reconcile_trace.py --full "$OUT/${tag}_full.json" \
    --rests "$OUT/${tag}_rests.jsonl" --sched "$OUT/${tag}_sched.jsonl" \
    --out "$OUT/${tag}_recon.json" > "$OUT/${tag}_recon.txt" 2>&1
  echo "[$tag]"; grep -E "TTFT [0-9]|queue  |prefill  |residual  |frontend_prep|TPOT|pure-decode|mixed step" "$OUT/${tag}_recon.txt" 2>/dev/null | head -9
}

echo "########## A/B quant [$CONFIGS] x $REPS rep interleaved ##########"
for rep in $(seq 1 $REPS); do
  for c in $CONFIGS; do run_one "$c" "$rep"; done
done
kill_server
echo "AB_QUANT_DONE -> $OUT"
