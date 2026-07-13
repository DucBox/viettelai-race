#!/bin/bash
# A/B: fp8 CÓ --calculate-kv-scales vs KHÔNG. Cô lập đúng 1 biến để kiểm chứng
# giả thuyết "calc-kv-scales là overhead làm queue fp8 cao hơn noquant".
# Chung: 48k, gmu=0.37, tp=1, prefix-cache, language-model-only,
#        --quantization=fp8 --kv-cache-dtype=fp8 --max-num-seqs=20.
# 2 exp x 3 rep = 6 rep, INTERLEAVED cold: calc-nocalc x3.
set -u
OUT=/root/ab_calc
REPS=${REPS:-3}
mkdir -p "$OUT"

kill_server() {
  pkill -15 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  for i in $(seq 1 90); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${m:-9999}" -lt 1000 ] && break; sleep 1
  done
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${m:-9999}" -ge 1000 ]; then
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null; fuser -k /dev/nvidia* 2>/dev/null; sleep 5
  fi
  sleep 2
}
snap_gpu() { nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw --format=csv,noheader | head -1; }

start_server() {
  local cfg=$1 log=$2 CALC=""
  [ "$cfg" = "calc" ] && CALC="--calculate-kv-scales"
  VLLM_LOGGING_LEVEL=INFO nohup /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model=/root/model --served-model-name=Qwen3.5-2B --host=0.0.0.0 --port=8000 \
    --max-model-len=48000 --gpu-memory-utilization=0.37 --tensor-parallel-size=1 \
    --enable-prefix-caching --language-model-only \
    --kv-cache-dtype=fp8 --quantization=fp8 --max-num-seqs=20 $CALC \
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
echo "########## A/B calc vs nocalc (fp8) — $REPS reps interleaved ##########"
for rep in $(seq 1 $REPS); do
  run_one calc "$rep"
  run_one nocalc "$rep"
done
kill_server
echo "ALL_DONE  ->  $OUT"