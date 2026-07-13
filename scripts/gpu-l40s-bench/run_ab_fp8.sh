#!/bin/bash
# A/B: noquant (bf16 weight + KV mặc định) vs fp8 (weight FP8 + KV fp8).
# Khác nhau DUY NHẤT: --quantization=fp8 --kv-cache-dtype=fp8 --calculate-kv-scales.
# Chung: 48k ctx, gmu=0.37, tp=1, prefix-cache, language-model-only, max-num-seqs=20.
# median-of-3, INTERLEAVED cold: noquant-fp8-noquant-fp8-noquant-fp8.
# Mỗi rep: kill sạch -> snapshot GPU -> serve -> WARMUP (synthetic) -> health ->
#          replay 120-req (+sampler 0.5s) -> merge -> ERS+stats.
set -u
OUT=/root/ab_fp8
REPS=${REPS:-3}
mkdir -p "$OUT"

kill_server() {
  pkill -15 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  for i in $(seq 1 90); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${m:-9999}" -lt 1000 ] && break
    sleep 1
  done
  # fallback nếu vẫn còn (CUDA context sót)
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${m:-9999}" -ge 1000 ]; then
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
    fuser -k /dev/nvidia* 2>/dev/null
    sleep 5
  fi
  sleep 2
}

snap_gpu() {
  nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw \
    --format=csv,noheader | head -1
}

start_server() {
  local cfg=$1 log=$2 EXTRA=""
  [ "$cfg" = "fp8" ] && EXTRA="--kv-cache-dtype=fp8 --calculate-kv-scales --quantization=fp8"
  VLLM_LOGGING_LEVEL=INFO nohup /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model=/root/model --served-model-name=Qwen3.5-2B --host=0.0.0.0 --port=8000 \
    --max-model-len=48000 --gpu-memory-utilization=0.37 --tensor-parallel-size=1 \
    --enable-prefix-caching --language-model-only --max-num-seqs=20 $EXTRA \
    > "$log" 2>&1 &
  echo $! > /root/vllm.pid
}

wait_health() {
  local log=$1
  for i in $(seq 1 300); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)" = "200" ] \
      && { echo "[health] UP ${i}s"; return 0; }
    kill -0 "$(cat /root/vllm.pid 2>/dev/null)" 2>/dev/null || { echo "[FAIL] proc chết"; tail -25 "$log"; return 1; }
    sleep 1
  done
  echo "[health] TIMEOUT"; return 1
}

run_one() {
  local cfg=$1 rep=$2
  local log=$OUT/${cfg}_rep${rep}.log
  echo "=================== $cfg rep$rep ==================="
  kill_server
  echo "[gpu pre-load ] $(snap_gpu)"
  start_server "$cfg" "$log"
  if ! wait_health "$log"; then echo "[SKIP] $cfg rep$rep"; return 1; fi
  echo "[gpu post-load] $(snap_gpu)"
  # WARMUP ổn định (synthetic, non-trace)
  /usr/bin/python3 /root/warmup_stub.py > "$OUT/${cfg}_rep${rep}_warmup.txt" 2>&1
  echo "[warmup] $(tail -1 $OUT/${cfg}_rep${rep}_warmup.txt)"
  sleep 2
  # BENCH: replay 120-req + sampler 0.5s
  /usr/bin/python3 /root/replay_trace_detailed.py > "$OUT/${cfg}_rep${rep}_replay.txt" 2>&1
  cp /root/replay_detailed_samples.json "$OUT/${cfg}_rep${rep}_samples.json" 2>/dev/null
  cp /root/replay_detailed_requests.json "$OUT/${cfg}_rep${rep}_requests.json" 2>/dev/null
  # MERGE server REQSTAT (queue/prefill/decode/tpot/cached)
  /usr/bin/python3 /root/merge_request_metrics.py --requests /root/replay_detailed_requests.json \
    --log "$log" --out "$OUT/${cfg}_rep${rep}_full.json" > "$OUT/${cfg}_rep${rep}_table.txt" 2>&1
  # ERS + stats
  /usr/bin/python3 /root/ers_and_stats.py --full "$OUT/${cfg}_rep${rep}_full.json" \
    --samples "$OUT/${cfg}_rep${rep}_samples.json" --out "$OUT/${cfg}_rep${rep}_summary.json" \
    > "$OUT/${cfg}_rep${rep}_ers.txt" 2>&1
  echo "[done] $(head -1 $OUT/${cfg}_rep${rep}_ers.txt)"
}

echo "########## A/B noquant vs fp8 — $REPS reps interleaved ##########"
for rep in $(seq 1 $REPS); do
  run_one noquant "$rep"
  run_one fp8 "$rep"
done
kill_server
echo "ALL_DONE  ->  $OUT"