#!/bin/bash
export VLLM_LOGGING_LEVEL=INFO
nohup /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model=/root/model --served-model-name=Qwen3.5-2B --host=0.0.0.0 --port=8000 \
  --max-model-len=48000 --gpu-memory-utilization=0.37 --tensor-parallel-size=1 \
  --enable-prefix-caching --language-model-only --kv-cache-dtype=fp8 \
  --calculate-kv-scales --max-num-seqs=32 --quantization=fp8 \
  --gdn-prefill-backend=flashinfer --max-num-batched-tokens=4096 \
  > /root/vllm_serve_v20.log 2>&1 &
echo $! > /root/vllm.pid
echo "Started, PID=$(cat /root/vllm.pid)"
