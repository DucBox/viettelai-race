#!/bin/bash
# A/B chuan: baseline (v12) vs spec (v12 + MTP num_speculative_tokens=1).
# Chi khac DUNG 1 bien = --speculative-config. serving.py giu ORIGINAL ca 2 ben
# (spec-decode khong lien quan patch priority). median-of-3, interleaved, cold moi rep.
# Voi rep spec: snapshot /metrics spec-decode counters truoc/sau de tinh acceptance.
set -u
OUT=/root/ab_spec
REPS=${REPS:-3}
mkdir -p $OUT

kill_server() {
  pkill -15 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  for i in $(seq 1 60); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${m:-9999}" -lt 1000 ] && break
    sleep 1
  done
  sleep 2
  echo "[gpu] mem_used sau kill: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB"
}

start_server() {
  local cfg=$1 log=$2 SPEC=""
  [ "$cfg" = "spec" ] && SPEC="--speculative-config {\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":1}"
  VLLM_LOGGING_LEVEL=INFO nohup /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model=/root/model --served-model-name=Qwen3.5-2B --host=0.0.0.0 --port=8000 \
    --max-model-len=48000 --gpu-memory-utilization=0.37 --tensor-parallel-size=1 \
    --enable-prefix-caching --language-model-only --kv-cache-dtype=fp8 \
    --calculate-kv-scales --max-num-seqs=32 --quantization=fp8 \
    --gdn-prefill-backend=flashinfer $SPEC \
    > "$log" 2>&1 &
  echo $! > /root/vllm.pid
}

wait_health() {
  for i in $(seq 1 300); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)" = "200" ] && { echo "[health] UP sau ${i}s"; return 0; }
    sleep 1
  done
  echo "[health] TIMEOUT"; return 1
}

spec_counter() {  # tong accepted / draft tu /metrics
  curl -s http://localhost:8000/metrics | awk '
    /vllm:spec_decode_num_draft_tokens_total/ {d=$2}
    /vllm:spec_decode_num_accepted_tokens_total/ {a=$2}
    END{printf "%d %d", d, a}'
}

run_one() {
  local cfg=$1 rep=$2
  local log=$OUT/${cfg}_rep${rep}.log
  echo "=================== $cfg rep$rep ==================="
  kill_server
  start_server "$cfg" "$log"
  if ! wait_health; then echo "[FAIL] $cfg rep$rep"; tail -5 "$log"; return 1; fi
  echo "[gpu] mem_used sau load: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB"
  sleep 3
  local db="0 0"; [ "$cfg" = "spec" ] && db=$(spec_counter)
  /usr/bin/python3 /root/replay_trace_detailed.py > $OUT/${cfg}_rep${rep}_replay.txt 2>&1
  /usr/bin/python3 /root/merge_request_metrics.py --requests /root/replay_detailed_requests.json \
    --log "$log" --out $OUT/${cfg}_rep${rep}_full.json > $OUT/${cfg}_rep${rep}_table.txt 2>&1
  if [ "$cfg" = "spec" ]; then
    local da=$(spec_counter)
    echo "$db | $da" > $OUT/${cfg}_rep${rep}_spec.txt
    echo "[spec] draft/accept before=[$db] after=[$da]"
  fi
  echo "[done] $cfg rep$rep -> $(head -1 $OUT/${cfg}_rep${rep}_table.txt)"
}

# serving.py phai ORIGINAL (spec khong dung priority patch)
/usr/bin/python3 /root/patch_serving_priority.py revert 2>&1 | tail -1
for rep in $(seq 1 $REPS); do
  run_one baseline $rep
  run_one spec $rep
done
kill_server
echo "ALL_DONE"
