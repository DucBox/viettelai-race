#!/bin/bash
# Chẩn đoán per-STEP: serve noquant & fp8 (1 lần mỗi cái) với SCHED_TRACE bật,
# replay trace, thu sched_*.jsonl. Không phải A/B điểm — để bóc tách queue.
set -u
OUT=/root/sched_diag
mkdir -p "$OUT"
kill_server() {
  pkill -15 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  for i in $(seq 1 90); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "${m:-9999}" -lt 1000 ]&&break; sleep 1; done
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1)
  [ "${m:-9999}" -ge 1000 ] && { pkill -9 -f "vllm.entrypoints.openai.api_server"; fuser -k /dev/nvidia* 2>/dev/null; sleep 5; }
  sleep 2
}
wait_health() { for i in $(seq 1 300); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health)" = "200" ]&&{ echo "UP ${i}s"; return 0; }; kill -0 $(cat /root/vllm.pid) 2>/dev/null||{ tail -20 "$1"; return 1; }; sleep 1; done; return 1; }

run_cfg() {
  local cfg=$1 EXTRA=""
  [ "$cfg" = "fp8" ] && EXTRA="--kv-cache-dtype=fp8 --quantization=fp8"
  kill_server
  echo "=== serve $cfg (SCHED_TRACE on) ==="
  SCHED_TRACE=$OUT/sched_${cfg}.jsonl VLLM_LOGGING_LEVEL=INFO \
    nohup /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model=/root/model --served-model-name=Qwen3.5-2B --host=0.0.0.0 --port=8000 \
    --max-model-len=48000 --gpu-memory-utilization=0.37 --tensor-parallel-size=1 \
    --enable-prefix-caching --language-model-only --max-num-seqs=20 $EXTRA \
    > $OUT/${cfg}_serve.log 2>&1 &
  echo $! > /root/vllm.pid
  wait_health $OUT/${cfg}_serve.log || { echo "FAIL $cfg"; return 1; }
  grep -oE "max_num_batched_tokens=[0-9]+|scheduler.*max_num_batched" $OUT/${cfg}_serve.log | head -1
  /usr/bin/python3 /root/warmup_stub.py > /dev/null 2>&1
  # (không truncate: file mở 1 lần lúc import; warmup chỉ 2 req nhỏ, analyzer lọc burst)
  sleep 1
  echo "=== replay $cfg ==="
  /usr/bin/python3 /root/replay_trace_detailed.py > $OUT/${cfg}_replay.txt 2>&1
  cp /root/replay_detailed_requests.json $OUT/${cfg}_requests.json
  /usr/bin/python3 /root/merge_request_metrics.py --requests /root/replay_detailed_requests.json \
    --log $OUT/${cfg}_serve.log --out $OUT/${cfg}_full.json > /dev/null 2>&1
  echo "steps ghi: $(wc -l < $OUT/sched_${cfg}.jsonl)"
}

/usr/bin/python3 /root/patch_sched_trace.py apply
run_cfg noquant
run_cfg fp8
kill_server
/usr/bin/python3 /root/patch_sched_trace.py revert
echo "SCHED_DIAG_DONE -> $OUT"