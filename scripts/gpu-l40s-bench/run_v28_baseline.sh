#!/bin/bash
# BASELINE v28 (single-config, fully-instrumented) tren L40S thue.
# = docker-compose.submit-v28.yml NHUNG:
#   - gpu-memory-utilization = 0.40  (L40S 48GB full VRAM -> ~19GB ~ khop grader H200-MIG 18GB)
#   - max-num-seqs           = 24    (20 user/burst + margin; khong nghen seq-slot)
# Muc tieu: lay CAY PHAN RA TTFT/TPOT day du va DOI SOAT (Sigma con ~ cha).
#
# Yeu cau co san tren box:
#   /root/model                 (hf download Qwen/Qwen3.5-2B  -- KHONG scp)
#   /root/trace-round1.jsonl    (scp tu data/)
#   /root/*.py                  (scp toan bo scripts/gpu-l40s-bench/*.py)
# Interpreter = /venv/main/bin/python (co vllm 0.24 + httpx). KHONG dung /usr/bin/python3.
set -u
PY=/venv/main/bin/python
OUT=/root/v28_baseline
mkdir -p "$OUT"
source /root/env_pins.sh 2>/dev/null || true
SRV_PIN=${SRV_PIN:-}; CLI_PIN=${CLI_PIN:-}
REPS="${REPS:-3}"

kill_server() {
  pkill -15 -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 3
  pkill -9  -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  pkill -9  -f "multiprocessing.spawn" 2>/dev/null
  pkill -9  -f "resource_tracker" 2>/dev/null; sleep 2
}

# --- apply 3 patch instrument (idempotent) ---
$PY /root/patch_loggers.py            # REQSTAT per-request vao log
$PY /root/patch_residual_ts.py apply  # arrival/ftl/*_ts (residual)
$PY /root/patch_sched_trace.py apply  # per-step exec_gap/tokens/ids (queue/prefill/tpot)
$PY /root/patch_residual_ts.py status
$PY /root/patch_sched_trace.py status

run_one() {
  local rep=$1 tag="rep${1}"
  echo "=================== v28 $tag ==================="
  kill_server
  export VLLM_LOGGING_LEVEL=INFO
  export RESIDUAL_TRACE=$OUT/${tag}_rests.jsonl
  export SCHED_TRACE=$OUT/${tag}_sched.jsonl
  : > "$RESIDUAL_TRACE"; : > "$SCHED_TRACE"
  # === v28 flags (giu nguyen) + 2 override ===
  nohup $SRV_PIN $PY -m vllm.entrypoints.openai.api_server \
    --model=/root/model --served-model-name=Qwen3.5-2B --host=0.0.0.0 --port=8000 \
    --max-model-len=32768 \
    --gpu-memory-utilization=0.40 \
    --tensor-parallel-size=1 \
    --enable-prefix-caching \
    --language-model-only \
    --default-chat-template-kwargs='{"enable_thinking": false}' \
    --quantization=fp8 \
    --kv-cache-dtype=fp8 \
    --no-enable-log-requests \
    --async-scheduling \
    --max-num-seqs=24 \
    > "$OUT/${tag}_serve.log" 2>&1 &
  echo $! > /root/vllm.pid
  local up=0
  for i in $(seq 1 400); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)" = "200" ] \
      && { echo "[health] UP ${i}s"; up=1; break; }
    kill -0 "$(cat /root/vllm.pid 2>/dev/null)" 2>/dev/null || { echo "[FAIL proc chet]"; grep -iE "error|oom|traceback" "$OUT/${tag}_serve.log" | head -6; return 1; }
    sleep 1
  done
  [ "$up" = 1 ] || { echo "[TIMEOUT]"; return 1; }

  $CLI_PIN $PY /root/warmup_stub.py > "$OUT/${tag}_warmup.txt" 2>&1
  sleep 2
  $CLI_PIN $PY /root/replay_trace_detailed.py > "$OUT/${tag}_replay.txt" 2>&1
  cp /root/replay_detailed_samples.json  "$OUT/${tag}_samples.json"  2>/dev/null
  cp /root/replay_detailed_requests.json "$OUT/${tag}_requests.json" 2>/dev/null

  local npre; npre=$(grep -c "PREEMPT\|preempted" "$OUT/${tag}_serve.log" 2>/dev/null)
  echo "  preemptions=$npre"
  $PY /root/env_gate.py --samples "$OUT/${tag}_samples.json" || echo "  GATE FAIL $tag"

  $PY /root/merge_request_metrics.py --requests /root/replay_detailed_requests.json \
    --log "$OUT/${tag}_serve.log" --out "$OUT/${tag}_full.json" > "$OUT/${tag}_merge.txt" 2>&1
  $PY /root/reconcile_trace.py --full "$OUT/${tag}_full.json" \
    --rests "$RESIDUAL_TRACE" --sched "$SCHED_TRACE" \
    --out "$OUT/${tag}_recon.json" | tee "$OUT/${tag}_recon.txt"
}

echo "########## v28 baseline x $REPS rep ##########"
for rep in $(seq 1 $REPS); do run_one "$rep"; done
kill_server
echo "V28_BASELINE_DONE -> $OUT"
echo "Xem: $OUT/rep*_recon.txt  (cay phan ra + err reconcile moi tang)"
