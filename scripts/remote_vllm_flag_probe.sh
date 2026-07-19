#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-18080}"
SESSION="${SESSION:-vllm_probe}"
MODEL_PATH="${MODEL_PATH:-/root/viettelai-race/serve/models/lfm2.5-1.2b}"
PATCH_ROOT="${PATCH_ROOT:-/root}"
SITE_PACKAGES="${SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages/vllm/v1/worker}"
LOG_DIR="${LOG_DIR:-/root/viettelai-race/tmp/flag-probe}"
FLAG_NAME="${1:-}"

mkdir -p "${LOG_DIR}"

cp "${PATCH_ROOT}/backport_flags.py" "${SITE_PACKAGES}/backport_flags.py"
cp "${PATCH_ROOT}/block_table.py" "${SITE_PACKAGES}/block_table.py"
cp "${PATCH_ROOT}/gpu_input_batch.py" "${SITE_PACKAGES}/gpu_input_batch.py"
cp "${PATCH_ROOT}/gpu_model_runner.py" "${SITE_PACKAGES}/gpu_model_runner.py"

/venv/main/bin/python -m py_compile \
  "${SITE_PACKAGES}/backport_flags.py" \
  "${SITE_PACKAGES}/block_table.py" \
  "${SITE_PACKAGES}/gpu_input_batch.py" \
  "${SITE_PACKAGES}/gpu_model_runner.py"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
fuser -k "${PORT}/tcp" 2>/dev/null || true

ENV_PREFIX=""
if [[ -n "${FLAG_NAME}" && "${FLAG_NAME}" != "baseline" ]]; then
  ENV_PREFIX="${FLAG_NAME}=1"
fi

LOG_FILE="${LOG_DIR}/${FLAG_NAME:-baseline}.log"
RESP_FILE="${LOG_DIR}/${FLAG_NAME:-baseline}.response.json"

CMD="${ENV_PREFIX} /venv/main/bin/python -m vllm.entrypoints.openai.api_server \
  --model=${MODEL_PATH} \
  --served-model-name=LFM2.5-1.2B-Instruct \
  --host=127.0.0.1 \
  --port=${PORT} \
  --max-model-len=32768 \
  --gpu-memory-utilization=0.95 \
  --tensor-parallel-size=1 \
  --enable-prefix-caching"

tmux new-session -d -s "${SESSION}" "bash -lc '${CMD} > ${LOG_FILE} 2>&1'"

READY=0
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "${READY}" != "1" ]]; then
  echo "SERVER_FAILED"
  tail -n 80 "${LOG_FILE}" || true
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  exit 2
fi

curl -sf "http://127.0.0.1:${PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "LFM2.5-1.2B-Instruct",
    "prompt": "Hello from vLLM backport test.",
    "max_tokens": 8,
    "temperature": 0.0
  }' > "${RESP_FILE}"

python3 - <<'PY' "${RESP_FILE}" "${FLAG_NAME:-baseline}"
import json, sys
resp_path, flag_name = sys.argv[1], sys.argv[2]
with open(resp_path) as f:
    data = json.load(f)
text = data["choices"][0]["text"]
usage = data.get("usage", {})
print(json.dumps({
    "flag": flag_name,
    "ok": True,
    "text_preview": text[:120],
    "usage": usage,
}, ensure_ascii=False))
PY

tmux kill-session -t "${SESSION}" 2>/dev/null || true
