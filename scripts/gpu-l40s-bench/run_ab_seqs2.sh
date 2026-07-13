#!/bin/bash
# A/B: fp8 (v12-like) với --max-num-seqs = 20 / 10 / 6. Khác nhau DUY NHẤT max-num-seqs.
# Chung: 48k ctx, gmu=0.37, tp=1, prefix-cache, language-model-only,
#        --quantization=fp8 --kv-cache-dtype=fp8 --calculate-kv-scales.
# Mục tiêu: đo lever "over-admit hại queue" — ít seq đồng thời => nhiều budget/step
#           => prefill xả nhanh => queue/TTFT thấp hơn? Đánh đổi decode throughput.
# 3 exp x 3 rep = 9 rep, INTERLEAVED cold: s20-s10-s6 x3. s20 để check reproducibility
# so với phiên ab_fp8 trước.
set -u
OUT=/root/ab_seqs2
REPS=${REPS:-3}
mkdir -p "$OUT"

kill_server() {
  pkill -15 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  for i in $(seq 1 90); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${m:-9999}" -lt 1000 ] && break
    sleep 1
  done
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${m:-9999}" -ge 1000 ]; then
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null; fuser -k /dev/nvidia* 2>/dev/null; sleep 5
  fi
  sleep 2
}

snap_gpu() {
  nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw \
    --format=csv,noheader | head -1
}

start_server() {
  local cfg=$1 log=$2 NSEQ
  case "$cfg" in
    s32) NSEQ=32 ;; s20) NSEQ=20 ;; s10) NSEQ=10 ;;
    *) echo "bad cfg $cfg"; return 1 ;;
  esac
  VLLM_LOGGING_LEVEL=INFO nohup /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model=/root/model --served-model-name=Qwen3.5-2B --host=0.0.0.0 --port=8000 \
    --max-model-len=48000 --gpu-memory-utilization=0.37 --tensor-parallel-size=1 \
    --enable-prefix-caching --language-model-only \
    --kv-cache-dtype=fp8 --calculate-kv-scales --quantization=fp8 \
    --max-num-seqs=$NSEQ --max-num-batched-tokens=3216 \
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
  /usr/bin/python3 /root/warmup_stub.py > "$OUT/${cfg}_rep${rep}_warmup.txt" 2>&1
  echo "[warmup] $(tail -1 $OUT/${cfg}_rep${rep}_warmup.txt)"
  sleep 2
  /usr/bin/python3 /root/replay_trace_detailed.py > "$OUT/${cfg}_rep${rep}_replay.txt" 2>&1
  cp /root/replay_detailed_samples.json "$OUT/${cfg}_rep${rep}_samples.json" 2>/dev/null
  cp /root/replay_detailed_requests.json "$OUT/${cfg}_rep${rep}_requests.json" 2>/dev/null
  /usr/bin/python3 /root/merge_request_metrics.py --requests /root/replay_detailed_requests.json \
    --log "$log" --out "$OUT/${cfg}_rep${rep}_full.json" > "$OUT/${cfg}_rep${rep}_table.txt" 2>&1
  /usr/bin/python3 /root/ers_and_stats.py --full "$OUT/${cfg}_rep${rep}_full.json" \
    --samples "$OUT/${cfg}_rep${rep}_samples.json" --out "$OUT/${cfg}_rep${rep}_summary.json" \
    > "$OUT/${cfg}_rep${rep}_ers.txt" 2>&1
  echo "[done] $(head -1 $OUT/${cfg}_rep${rep}_ers.txt)"
}

echo "########## A/B max-num-seqs 10/20/32 @ batch3216 (fp8) — $REPS reps interleaved ##########"
for rep in $(seq 1 $REPS); do
  run_one s10 "$rep"
  run_one s20 "$rep"
  run_one s32 "$rep"
done
kill_server
echo "ALL_DONE  ->  $OUT"