#!/bin/bash
# Baseline config (v12) + MTP speculative decode, num_speculative_tokens=1.
# mamba-cache-mode=align la default cho model nay (bat buoc cho Qwen3_5MTP).
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
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}' \
  > /root/specdecode_serve.log 2>&1 &
echo $! > /root/vllm.pid
echo "Started spec-decode server, PID=$(cat /root/vllm.pid)"
