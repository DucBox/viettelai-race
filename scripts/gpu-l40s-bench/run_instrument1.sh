#!/bin/bash
# BƯỚC 1: 1 rep fully-instrumented (fp8-all, seqs20, batch3216 = v23 best) để
# VALIDATE pipeline reconcile + trả lời residual = frontend_prep vs client_transport.
set -u
PY=/venv/main/bin/python
OUT=/root/inst1
mkdir -p "$OUT"
source /root/env_pins.sh 2>/dev/null || true
SRV_PIN=${SRV_PIN:-}; CLI_PIN=${CLI_PIN:-}
echo "[pin] SRV='$SRV_PIN'  CLI='$CLI_PIN'"

kill_server() {
  pkill -15 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  sleep 3
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  pkill -9 -f "multiprocessing.spawn" 2>/dev/null
  pkill -9 -f "resource_tracker" 2>/dev/null
  sleep 2
}

# ---- patch runtime (import vllm -> phải PY của venv) ----
$PY /root/patch_loggers.py apply
$PY /root/patch_residual_ts.py apply
$PY /root/patch_sched_trace.py apply

kill_server
export VLLM_LOGGING_LEVEL=INFO
export RESIDUAL_TRACE=$OUT/rests.jsonl
export SCHED_TRACE=$OUT/sched.jsonl
: > "$RESIDUAL_TRACE"; : > "$SCHED_TRACE"

echo "[serve] fp8-all seqs20 batch3216 ..."
nohup $SRV_PIN $PY -m vllm.entrypoints.openai.api_server \
  --model=/root/model --served-model-name=Qwen3.5-2B --host=0.0.0.0 --port=8000 \
  --max-model-len=48000 --gpu-memory-utilization=0.37 --tensor-parallel-size=1 \
  --enable-prefix-caching --language-model-only \
  --kv-cache-dtype=fp8 --calculate-kv-scales --quantization=fp8 \
  --max-num-seqs=20 --max-num-batched-tokens=3216 --gdn-prefill-backend=flashinfer \
  > "$OUT/serve.log" 2>&1 &
echo $! > /root/vllm.pid

for i in $(seq 1 400); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)" = "200" ] \
    && { echo "[health] UP ${i}s"; break; }
  kill -0 "$(cat /root/vllm.pid 2>/dev/null)" 2>/dev/null || { echo "[FAIL] proc chet"; tail -40 "$OUT/serve.log"; exit 1; }
  sleep 1
done

echo "[warmup]"; $CLI_PIN $PY /root/warmup_stub.py > "$OUT/warmup.txt" 2>&1; tail -1 "$OUT/warmup.txt"
sleep 2
echo "[replay 120 req]"; $CLI_PIN $PY /root/replay_trace_detailed.py > "$OUT/replay.txt" 2>&1
cp /root/replay_detailed_samples.json "$OUT/samples.json" 2>/dev/null
cp /root/replay_detailed_requests.json "$OUT/requests.json" 2>/dev/null

echo "[env gate]"; $PY /root/env_gate.py --samples "$OUT/samples.json" || echo "  ⚠️ GATE FAIL — rep nhiễu"
echo "[merge]"; $PY /root/merge_request_metrics.py --requests /root/replay_detailed_requests.json \
  --log "$OUT/serve.log" --out "$OUT/full.json" > "$OUT/merge.txt" 2>&1; tail -1 "$OUT/merge.txt"
echo "[reconcile]"; $PY /root/reconcile_trace.py --full "$OUT/full.json" \
  --rests "$OUT/rests.jsonl" --sched "$OUT/sched.jsonl" --out "$OUT/recon.json"

kill_server
echo "INSTRUMENT1_DONE -> $OUT"
