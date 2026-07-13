#!/bin/bash
# A/B sach: baseline (FCFS) vs spf (priority policy + patch priority=len).
# Moi rep = kill + restart server SACH (cold-start dung nhu bai thi, khong cache
# carryover). Chi khac DUNG 1 bien = scheduling policy + patch serving.py.
# Moi flag serve khac deu GIONG HET (gpu-mem 0.37, khong DP, khong batched-tokens).
#
# YEU CAU truoc khi chay (lam trong setup):
#   python3 /root/patch_loggers.py            # bat REQSTAT per-request logging
#   (patch_serving_priority.py se duoc script nay goi tu dong moi run)
#
# Luu y GPU-leak: neu GPU co san memory leak tu process cu (host-PID khong kill
# duoc), kill_server KHONG cho VRAM ve 0 -- chi cho process vLLM cua minh chet.
# Tot nhat chay tren instance SACH.
set -u
OUT=/root/ab
REPS=${REPS:-3}
mkdir -p $OUT

kill_server() {
  # QUAN TRONG: dung SIGTERM (graceful) truoc, de vLLM tu dong dep CUDA context
  # va tra VRAM. SIGKILL (-9) de lai context mo coi -> leak VRAM -> server sau
  # chong len -> OOM. Chi -9 khi graceful that bai.
  pkill -15 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  # doi VRAM that su tra ve (khong chi doi process chet) -- day moi la dieu quan trong
  local ok=0
  for i in $(seq 1 60); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${m:-9999}" -lt 1000 ]; then ok=1; break; fi
    sleep 1
  done
  if [ "$ok" != "1" ]; then
    echo "[kill] graceful chua sach sau 60s -> SIGKILL + fuser"
    pkill -9 -f "vllm" 2>/dev/null
    fuser -k /dev/nvidia* 2>/dev/null
    for i in $(seq 1 30); do
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      if [ "${m:-9999}" -lt 1000 ]; then break; fi
      sleep 1
    done
  fi
  sleep 2
  echo "[gpu] mem_used sau kill: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB (can <1000)"
}

start_server() {
  local cfg=$1 log=$2
  if [ "$cfg" = "baseline" ]; then
    /usr/bin/python3 /root/patch_serving_priority.py revert
    POLICY=""
  else
    /usr/bin/python3 /root/patch_serving_priority.py apply
    POLICY="--scheduling-policy=priority"
  fi
  VLLM_LOGGING_LEVEL=INFO nohup /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model=/root/model --served-model-name=Qwen3.5-2B --host=0.0.0.0 --port=8000 \
    --max-model-len=48000 --gpu-memory-utilization=0.37 --tensor-parallel-size=1 \
    --enable-prefix-caching --language-model-only --kv-cache-dtype=fp8 \
    --calculate-kv-scales --max-num-seqs=32 --quantization=fp8 \
    --gdn-prefill-backend=flashinfer $POLICY \
    > "$log" 2>&1 &
  echo $! > /root/vllm.pid
}

wait_health() {
  for i in $(seq 1 240); do
    code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)
    if [ "$code" = "200" ]; then echo "[health] UP sau ${i}s"; return 0; fi
    sleep 1
  done
  echo "[health] TIMEOUT"; return 1
}

run_one() {
  local cfg=$1 rep=$2
  local log=$OUT/${cfg}_rep${rep}.log
  echo "=================== $cfg rep$rep ==================="
  kill_server
  start_server "$cfg" "$log"
  if ! wait_health; then echo "[FAIL] $cfg rep$rep health timeout"; tail -5 "$log"; return 1; fi
  echo "[kv] $(grep -iE 'GPU KV cache size|Maximum concurrency|KV cache' "$log" | tail -1)"
  echo "[gpu] mem_used sau load: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB"
  sleep 3
  echo "[bench] chay replay_trace_detailed.py ..."
  /usr/bin/python3 /root/replay_trace_detailed.py > $OUT/${cfg}_rep${rep}_replay.txt 2>&1
  cp /root/replay_detailed_requests.json $OUT/${cfg}_rep${rep}_requests.json
  cp /root/replay_detailed_samples.json  $OUT/${cfg}_rep${rep}_samples.json
  /usr/bin/python3 /root/merge_request_metrics.py \
    --requests /root/replay_detailed_requests.json \
    --log "$log" \
    --out $OUT/${cfg}_rep${rep}_full.json > $OUT/${cfg}_rep${rep}_table.txt 2>&1
  echo "[done] $cfg rep$rep -> $(head -1 $OUT/${cfg}_rep${rep}_table.txt)"
}

echo "[setup] loggers patch: $(grep -c 'REQSTAT' /usr/local/lib/python3.12/dist-packages/vllm/v1/metrics/loggers.py) (can >=1)"
for rep in $(seq 1 $REPS); do
  run_one baseline $rep
  run_one spf $rep
done
kill_server
echo "ALL_DONE"
