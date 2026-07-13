#!/bin/bash
# A/B: fp8 (v12-like) với max_num_batched_tokens = 3216/5360/6144/6432.
# fp8 block=1072 → 2048 chỉ nhét 1 block (phí 48%); 2144=2×1072, 3216=3×1072 khít.
# Kiểm chứng lever: budget khít block → prefill xả nhanh → queue/TTFT giảm.
# Chung: 48k, gmu=0.37, tp=1, prefix-cache, language-model-only,
#        --quantization=fp8 --kv-cache-dtype=fp8 --max-num-seqs=20.
# 4 exp x 3 rep = 12 rep, INTERLEAVED cold.
set -u
OUT=/root/ab_batchtok2
REPS=${REPS:-3}
mkdir -p "$OUT"
kill_server() {
  pkill -15 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  for i in $(seq 1 90); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "${m:-9999}" -lt 1000 ]&&break; sleep 1; done
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1)
  [ "${m:-9999}" -ge 1000 ] && { pkill -9 -f "vllm.entrypoints.openai.api_server"; fuser -k /dev/nvidia* 2>/dev/null; sleep 5; }
  sleep 2
}
snap_gpu() { nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw --format=csv,noheader | head -1; }
start_server() {
  local cfg=$1 log=$2
  local N=${cfg#b}
  VLLM_LOGGING_LEVEL=INFO nohup /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model=/root/model --served-model-name=Qwen3.5-2B --host=0.0.0.0 --port=8000 \
    --max-model-len=48000 --gpu-memory-utilization=0.37 --tensor-parallel-size=1 \
    --enable-prefix-caching --language-model-only \
    --kv-cache-dtype=fp8 --quantization=fp8 --max-num-seqs=20 \
    --max-num-batched-tokens=$N \
    > "$log" 2>&1 &
  echo $! > /root/vllm.pid
}
wait_health() { local log=$1; for i in $(seq 1 300); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health)" = "200" ]&&{ echo "UP ${i}s"; return 0; }; kill -0 $(cat /root/vllm.pid) 2>/dev/null||{ echo FAIL; tail -20 "$log"; return 1; }; sleep 1; done; return 1; }
run_one() {
  local cfg=$1 rep=$2
  local log=$OUT/${cfg}_rep${rep}.log
  echo "=================== $cfg rep$rep ==================="
  kill_server
  echo "[gpu pre] $(snap_gpu)"
  start_server "$cfg" "$log"
  if ! wait_health "$log"; then echo "[SKIP] $cfg rep$rep"; return 1; fi
  echo "[gpu post] $(snap_gpu)"
  /usr/bin/python3 /root/warmup_stub.py > "$OUT/${cfg}_rep${rep}_warmup.txt" 2>&1
  sleep 2
  /usr/bin/python3 /root/replay_trace_detailed.py > "$OUT/${cfg}_rep${rep}_replay.txt" 2>&1
  cp /root/replay_detailed_samples.json "$OUT/${cfg}_rep${rep}_samples.json" 2>/dev/null
  /usr/bin/python3 /root/merge_request_metrics.py --requests /root/replay_detailed_requests.json \
    --log "$log" --out "$OUT/${cfg}_rep${rep}_full.json" > "$OUT/${cfg}_rep${rep}_table.txt" 2>&1
  /usr/bin/python3 /root/ers_and_stats.py --full "$OUT/${cfg}_rep${rep}_full.json" \
    --samples "$OUT/${cfg}_rep${rep}_samples.json" --out "$OUT/${cfg}_rep${rep}_summary.json" \
    > "$OUT/${cfg}_rep${rep}_ers.txt" 2>&1
  echo "[done] $(head -1 $OUT/${cfg}_rep${rep}_ers.txt)"
}
echo "########## A/B max-num-batched-tokens 3216/5360/6144/6432 (fp8) — $REPS reps ##########"
for rep in $(seq 1 $REPS); do
  run_one b3216 "$rep"; run_one b5360 "$rep"; run_one b6144 "$rep"; run_one b6432 "$rep"
done
kill_server
echo "ALL_DONE  ->  $OUT"