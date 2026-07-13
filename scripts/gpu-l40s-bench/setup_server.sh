#!/bin/bash
# Chay TREN instance moi (sau khi da scp *.sh *.py + trace-round1.jsonl len /root).
# Tai model + apply patch_loggers. Sau do dung run_ab.sh de bench.
set -e
cd /root

echo "=== 1. Tai model Qwen3.5-2B (neu chua co) ==="
if [ ! -f /root/model/config.json ]; then
  /usr/bin/python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Qwen/Qwen3.5-2B", local_dir="/root/model")'
else
  echo "model da co, bo qua"
fi

echo "=== 2. Patch loggers (REQSTAT per-request) ==="
/usr/bin/python3 /root/patch_loggers.py

echo "=== 3. Verify patch serving.py (status) ==="
/usr/bin/python3 /root/patch_serving_priority.py status

echo "=== 4. GPU sach? ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || echo "(khong co process nao giu GPU -- tot)"

echo
echo "SAN SANG. Chay bench trong tmux:"
echo "  tmux new-window -t bench -n abrun 'bash /root/run_ab.sh 2>&1 | tee /root/ab/driver.log'"
echo "  # theo doi: tail -f /root/ab/driver.log"
echo "  # sau khi ALL_DONE: /usr/bin/python3 /root/compare_ab.py --dir /root/ab"
