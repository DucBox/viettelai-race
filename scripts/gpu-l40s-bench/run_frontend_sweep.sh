#!/bin/bash
# SWEEP FRONTEND/COMPILE — base co dinh = fp8 + kv-fp8 + budget 3216 + seqs 24 (winner sweep1).
# Danh 2 muc chua cham: frontend_prep (15%) + prefill-overhead (55% prefill-wall).
#   ref     : base, khong them gi
#   front   : --api-server-count=2 --renderer-num-workers=2 --mm-processor-cache-gb=0   (song song tokenize/accept)
#   compile : --compilation-config compile_sizes=[1072,2144,3216]                        (phu prefill-shape -> bot launch)
#   all     : front + compile
# GIU kv-cache=fp8 (theo yeu cau).  Moi combo 1 rep.
set -u
PY=/venv/main/bin/python
OUT=/root/frontend_sweep
mkdir -p "$OUT"
TAGS="${TAGS:-ref front compile all}"

extra_flags() {
  case "$1" in
    ref)     echo "" ;;
    front)   echo "--api-server-count=2 --renderer-num-workers=2 --mm-processor-cache-gb=0" ;;
    compile) echo '--compilation-config={"compile_sizes":[1072,2144,3216]}' ;;
    all)     echo '--api-server-count=2 --renderer-num-workers=2 --mm-processor-cache-gb=0 --compilation-config={"compile_sizes":[1072,2144,3216]}' ;;
    *)       echo "" ;;
  esac
}

kill_server() {
  pkill -15 -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 3
  pkill -9  -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  pkill -9  -f "multiprocessing.spawn" 2>/dev/null
  pkill -9  -f "resource_tracker" 2>/dev/null
  pkill -9  -f "api_server" 2>/dev/null; sleep 2
}

$PY /root/patch_loggers.py >/dev/null 2>&1
$PY /root/patch_residual_ts.py apply >/dev/null 2>&1
$PY /root/patch_sched_trace.py apply >/dev/null 2>&1

run_combo() {
  local tag=$1
  local EXTRA; EXTRA=$(extra_flags "$tag")
  echo "=================== combo $tag  [extra: ${EXTRA:-none}] ==================="
  kill_server
  export VLLM_LOGGING_LEVEL=INFO
  export RESIDUAL_TRACE=$OUT/${tag}_rests.jsonl
  export SCHED_TRACE=$OUT/${tag}_sched.jsonl
  : > "$RESIDUAL_TRACE"; : > "$SCHED_TRACE"
  # base co dinh + EXTRA (bash word-split; JSON KHONG co space nen an toan)
  nohup $PY -m vllm.entrypoints.openai.api_server \
    --model=/root/model --served-model-name=Qwen3.5-2B --host=0.0.0.0 --port=8000 \
    --max-model-len=32768 --gpu-memory-utilization=0.40 --tensor-parallel-size=1 \
    --enable-prefix-caching --language-model-only \
    --default-chat-template-kwargs='{"enable_thinking": false}' \
    --quantization=fp8 --kv-cache-dtype=fp8 \
    --no-enable-log-requests --async-scheduling \
    --max-num-seqs=24 --max-num-batched-tokens=3216 \
    $EXTRA \
    > "$OUT/${tag}_serve.log" 2>&1 &
  echo $! > /root/vllm.pid
  local up=0
  for i in $(seq 1 500); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)" = "200" ] \
      && { echo "[health] UP ${i}s"; up=1; break; }
    pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null || { echo "[FAIL proc chet]"; grep -iE "error|oom|traceback|unrecognized|invalid" "$OUT/${tag}_serve.log" | tail -10; return 1; }
    sleep 1
  done
  [ "$up" = 1 ] || { echo "[TIMEOUT]"; return 1; }
  grep -m1 "attention block size" "$OUT/${tag}_serve.log" | sed 's/^.*INFO/  [blk]/'
  grep -m1 "api_server_count\|API server\|Started server process" "$OUT/${tag}_serve.log" | head -1

  $PY /root/warmup_stub.py > "$OUT/${tag}_warmup.txt" 2>&1
  sleep 2
  $PY /root/replay_trace_detailed.py > "$OUT/${tag}_replay.txt" 2>&1
  cp /root/replay_detailed_requests.json "$OUT/${tag}_requests.json" 2>/dev/null
  cp /root/replay_detailed_samples.json  "$OUT/${tag}_samples.json"  2>/dev/null
  echo "  preemptions=$(grep -c 'PREEMPT\|preempted' "$OUT/${tag}_serve.log" 2>/dev/null)"

  $PY /root/merge_request_metrics.py --requests /root/replay_detailed_requests.json \
    --log "$OUT/${tag}_serve.log" --out "$OUT/${tag}_full.json" > "$OUT/${tag}_merge.txt" 2>&1
  $PY /root/reconcile_trace.py --full "$OUT/${tag}_full.json" \
    --rests "$RESIDUAL_TRACE" --sched "$SCHED_TRACE" \
    --out "$OUT/${tag}_recon.json" > "$OUT/${tag}_recon.txt" 2>&1
  echo "  -> $OUT/${tag}_recon.txt"
}

echo "########## FRONTEND/COMPILE SWEEP (base=fp8/kvfp8/b3216/s24) ##########"
for t in $TAGS; do run_combo "$t"; done
kill_server

echo ""; echo "########## BANG SO SANH ##########"
$PY - "$OUT" $TAGS <<'PYEOF'
import sys, re, os
OUT = sys.argv[1]; tags = sys.argv[2:]
def grab(t, pat):
    m = re.search(pat, t); return float(m.group(1)) if m else float('nan')
print(f"{'combo':9} {'TTFT':>7} {'queue':>7} {'residual':>9} {'front_prep':>11} {'TPOT':>6} {'mixed%':>7}")
for tag in tags:
    p = os.path.join(OUT, f"{tag}_recon.txt")
    if not os.path.exists(p): print(f"{tag:9} (no recon)"); continue
    t = open(p).read()
    print(f"{tag:9} {grab(t, r'TTFT ([0-9.]+)ms'):>7.0f} {grab(t, r'queue +([0-9.]+)'):>7.0f} "
          f"{grab(t, r'residual +([0-9.]+)'):>9.0f} {grab(t, r'frontend_prep +([0-9.]+)'):>11.0f} "
          f"{grab(t, r'TPOT ([0-9.]+)ms'):>6.1f} {grab(t, r'mixed fraction +([0-9.]+)'):>6.1f}%")
PYEOF
echo "FRONTEND_SWEEP_DONE -> $OUT"
