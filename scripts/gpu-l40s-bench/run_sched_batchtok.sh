#!/bin/bash
# Instrument per-STEP cho 4 cấu hình max_num_batched_tokens (fp8, block=1072).
# Đo THỰC: tokens_sched mỗi step prefill => xác nhận block-tiling & margin.
set -u
OUT=/root/sched_batchtok
mkdir -p "$OUT"
kill_server() {
  pkill -15 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  for i in $(seq 1 90); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "${m:-9999}" -lt 1000 ]&&break; sleep 1; done
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1)
  [ "${m:-9999}" -ge 1000 ] && { pkill -9 -f "vllm.entrypoints.openai.api_server"; fuser -k /dev/nvidia* 2>/dev/null; sleep 5; }
  sleep 2
}
wait_health() { local log=$1; for i in $(seq 1 300); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health)" = "200" ]&&{ echo "UP ${i}s"; return 0; }; kill -0 $(cat /root/vllm.pid) 2>/dev/null||{ tail -20 "$log"; return 1; }; sleep 1; done; return 1; }
run_cfg() {
  local N=$1
  kill_server
  echo "=== serve b$N (SCHED_TRACE) ==="
  SCHED_TRACE=$OUT/sched_b${N}.jsonl VLLM_LOGGING_LEVEL=INFO \
    nohup /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model=/root/model --served-model-name=Qwen3.5-2B --host=0.0.0.0 --port=8000 \
    --max-model-len=48000 --gpu-memory-utilization=0.37 --tensor-parallel-size=1 \
    --enable-prefix-caching --language-model-only \
    --kv-cache-dtype=fp8 --quantization=fp8 --max-num-seqs=20 --max-num-batched-tokens=$N \
    > $OUT/b${N}_serve.log 2>&1 &
  echo $! > /root/vllm.pid
  wait_health $OUT/b${N}_serve.log || { echo "FAIL b$N"; return 1; }
  /usr/bin/python3 /root/warmup_stub.py > /dev/null 2>&1
  sleep 1
  /usr/bin/python3 /root/replay_trace_detailed.py > $OUT/b${N}_replay.txt 2>&1
  /usr/bin/python3 /root/merge_request_metrics.py --requests /root/replay_detailed_requests.json \
    --log $OUT/b${N}_serve.log --out $OUT/b${N}_full.json > /dev/null 2>&1
  echo "b$N steps=$(wc -l < $OUT/sched_b${N}.jsonl)"
}
/usr/bin/python3 /root/patch_sched_trace.py apply
for N in 2048 2144 3216 4096; do run_cfg $N; done
kill_server
/usr/bin/python3 /root/patch_sched_trace.py revert
echo "SCHED_BATCHTOK_DONE -> $OUT"