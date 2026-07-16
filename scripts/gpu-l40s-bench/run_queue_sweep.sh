#!/bin/bash
# (c) SWEEP QUEUE-LEVER — dinh luong don bay QUEUE (cung 1 lan thue, cung pipeline).
# Giu model=fp8 co dinh, chi doi knob lien quan queue:
#   base   : kv=fp8   budget=2048(default) seqs=24   (= v28)
#   b3216  : kv=fp8   budget=3216           seqs=24   (2-block prefill)
#   kvauto : kv=auto  budget=2048           seqs=24   (bf16-KV -> block 544, 2-lane)
#   s32    : kv=fp8   budget=2048           seqs=32   (headroom seq-slot / round-overlap)
# Moi combo 1 rep. In bang so sanh queue/prefill/residual/ttft/tpot cuoi.
set -u
PY=/venv/main/bin/python
OUT=/root/queue_sweep
mkdir -p "$OUT"

# combos: TAG|KV|BUDGET|SEQS   (BUDGET rong = default 2048)
COMBOS=(
  "base|fp8|@|24"
  "b3216|fp8|3216|24"
  "kvauto|auto|@|24"
  "s32|fp8|@|32"
)

kill_server() {
  pkill -15 -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 3
  pkill -9  -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  pkill -9  -f "multiprocessing.spawn" 2>/dev/null
  pkill -9  -f "resource_tracker" 2>/dev/null; sleep 2
}

# patches (idempotent)
$PY /root/patch_loggers.py >/dev/null 2>&1
$PY /root/patch_residual_ts.py apply >/dev/null 2>&1
$PY /root/patch_sched_trace.py apply >/dev/null 2>&1

run_combo() {
  local tag=$1 kv=$2 budget=$3 seqs=$4
  echo "=================== combo $tag (kv=$kv budget=$budget seqs=$seqs) ==================="
  kill_server
  export VLLM_LOGGING_LEVEL=INFO
  export RESIDUAL_TRACE=$OUT/${tag}_rests.jsonl
  export SCHED_TRACE=$OUT/${tag}_sched.jsonl
  : > "$RESIDUAL_TRACE"; : > "$SCHED_TRACE"
  local BUDGET_FLAG=""
  [ "$budget" != "@" ] && BUDGET_FLAG="--max-num-batched-tokens=$budget"
  nohup $PY -m vllm.entrypoints.openai.api_server \
    --model=/root/model --served-model-name=Qwen3.5-2B --host=0.0.0.0 --port=8000 \
    --max-model-len=32768 --gpu-memory-utilization=0.40 --tensor-parallel-size=1 \
    --enable-prefix-caching --language-model-only \
    --default-chat-template-kwargs='{"enable_thinking": false}' \
    --quantization=fp8 --kv-cache-dtype="$kv" \
    --no-enable-log-requests --async-scheduling \
    --max-num-seqs="$seqs" $BUDGET_FLAG \
    > "$OUT/${tag}_serve.log" 2>&1 &
  echo $! > /root/vllm.pid
  local up=0
  for i in $(seq 1 400); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)" = "200" ] \
      && { echo "[health] UP ${i}s"; up=1; break; }
    kill -0 "$(cat /root/vllm.pid 2>/dev/null)" 2>/dev/null || { echo "[FAIL proc chet]"; grep -iE "error|oom|traceback" "$OUT/${tag}_serve.log" | tail -8; return 1; }
    sleep 1
  done
  [ "$up" = 1 ] || { echo "[TIMEOUT]"; return 1; }
  # xac nhan block-size that (fp8=1072, auto/bf16=544)
  grep -m1 "attention block size" "$OUT/${tag}_serve.log" | sed 's/^.*INFO/  [blk]/'

  $PY /root/warmup_stub.py > "$OUT/${tag}_warmup.txt" 2>&1
  sleep 2
  $PY /root/replay_trace_detailed.py > "$OUT/${tag}_replay.txt" 2>&1
  cp /root/replay_detailed_requests.json "$OUT/${tag}_requests.json" 2>/dev/null
  cp /root/replay_detailed_samples.json  "$OUT/${tag}_samples.json"  2>/dev/null
  local npre; npre=$(grep -c "PREEMPT\|preempted" "$OUT/${tag}_serve.log" 2>/dev/null)
  echo "  preemptions=$npre"

  $PY /root/merge_request_metrics.py --requests /root/replay_detailed_requests.json \
    --log "$OUT/${tag}_serve.log" --out "$OUT/${tag}_full.json" > "$OUT/${tag}_merge.txt" 2>&1
  $PY /root/reconcile_trace.py --full "$OUT/${tag}_full.json" \
    --rests "$RESIDUAL_TRACE" --sched "$SCHED_TRACE" \
    --out "$OUT/${tag}_recon.json" > "$OUT/${tag}_recon.txt" 2>&1
  echo "  -> $OUT/${tag}_recon.txt"
}

echo "########## QUEUE SWEEP (4 combo x 1 rep) ##########"
for c in "${COMBOS[@]}"; do
  IFS='|' read -r tag kv budget seqs <<< "$c"
  run_combo "$tag" "$kv" "$budget" "$seqs"
done
kill_server

echo ""
echo "########## BANG SO SANH ##########"
$PY - "$OUT" "${COMBOS[@]}" <<'PYEOF'
import sys, re, os
OUT = sys.argv[1]; combos = sys.argv[2:]
def grab(txt, pat):
    m = re.search(pat, txt); return float(m.group(1)) if m else float('nan')
print(f"{'combo':8} {'TTFT':>7} {'queue':>7} {'prefill':>7} {'residual':>8} {'front_prep':>10} {'TPOT':>6} {'pure':>6} {'mixed%':>7}")
for c in combos:
    tag = c.split('|')[0]
    p = os.path.join(OUT, f"{tag}_recon.txt")
    if not os.path.exists(p):
        print(f"{tag:8} (no recon)"); continue
    t = open(p).read()
    ttft = grab(t, r"TTFT\s+([\d.]+)ms")
    queue = grab(t, r"queue\s+([\d.]+)")
    prefill = grab(t, r"prefill\s+([\d.]+)")
    resid = grab(t, r"residual\s+([\d.]+)")
    fp = grab(t, r"frontend_prep\s+([\d.]+)")
    tpot = grab(t, r"TPOT\s+([\d.]+)ms")
    pure = grab(t, r"pure-decode step\s+([\d.]+)")
    mixed = grab(t, r"mixed fraction\s+([\d.]+)")
    print(f"{tag:8} {ttft:>7.0f} {queue:>7.0f} {prefill:>7.0f} {resid:>8.0f} {fp:>10.0f} {tpot:>6.1f} {pure:>6.1f} {mixed:>6.1f}%")
PYEOF
echo "QUEUE_SWEEP_DONE -> $OUT"
