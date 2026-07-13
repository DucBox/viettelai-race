#!/bin/bash
# Wrapper cho replay_trace_detailed.py -- khac voi bench_with_diag.sh (chi
# chup before/after), script nay sampling CPU/GPU LIEN TUC trong suot qua
# trinh bench (moi 0.5s, xem sampler_task trong replay_trace_detailed.py)
# va log CHI TIET TUNG REQUEST thay vi chi mean/median theo round.
#
# Dung khi can biet: request nao cham, cham vao dung luc nao, va tai thoi
# diem do GPU/CPU dang o trang thai gi (util, sm_clock, throttle%).
set -e
cd /root
VLLM_LOG_PATH="${VLLM_LOG_PATH:-/root/vllm_serve.log}"
echo "=== Chay replay chi tiet (per-request + continuous system sampling) ==="
/usr/bin/python3 replay_trace_detailed.py | tee /root/bench_detailed_output.txt
if [ -f "$VLLM_LOG_PATH" ]; then
  echo
  echo "=== Join them REQSTAT tu vLLM log: $VLLM_LOG_PATH ==="
  /usr/bin/python3 merge_request_metrics.py --log "$VLLM_LOG_PATH" \
    | tee /root/bench_request_metrics_output.txt
else
  echo
  echo "Khong tim thay VLLM_LOG_PATH=$VLLM_LOG_PATH, bo qua buoc join REQSTAT."
fi
echo
echo "=== File output ==="
echo "  /root/replay_detailed_requests.json  (tung request)"
echo "  /root/replay_detailed_samples.json   (timeline he thong, moi 0.5s)"
echo "  /root/bench_detailed_output.txt      (bang in ra console)"
echo "  /root/replay_request_metrics_full.json (join client TTFT + server queue/prefill/tpot)"
echo "  /root/bench_request_metrics_output.txt (bang full metrics theo request)"
