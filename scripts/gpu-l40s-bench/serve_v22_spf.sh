#!/bin/bash
# v12 + --scheduling-policy=priority, KET HOP voi patch runtime tai
# chat_completion/serving.py (priority = len(prompt_token_ids) thay vi
# request.priority mac dinh) de thuc hien Shortest-Prefill-First (SPF).
# Patch phai duoc ap dung TRUOC khi chay script nay (xem lenh sed/python
# patch trong lich su ban trao doi / docs muc thi nghiem SPF).
export VLLM_LOGGING_LEVEL=INFO
nohup /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model=/root/model \
  --served-model-name=Qwen3.5-2B \
  --host=0.0.0.0 \
  --port=8000 \
  --max-model-len=48000 \
  --gpu-memory-utilization=0.37 \
  --tensor-parallel-size=1 \
  --enable-prefix-caching \
  --language-model-only \
  --kv-cache-dtype=fp8 \
  --calculate-kv-scales \
  --max-num-seqs=32 \
  --quantization=fp8 \
  --gdn-prefill-backend=flashinfer \
  --scheduling-policy=priority \
  > /root/vllm_serve_v22_spf.log 2>&1 &
echo $! > /root/vllm.pid
echo "Started, PID=$(cat /root/vllm.pid)"
