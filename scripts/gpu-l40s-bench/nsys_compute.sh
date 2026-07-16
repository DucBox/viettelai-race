#!/bin/bash
# TANG 5 (tuy chon) — nsys KHONG ROOT. Chi CUDA+NVTX, TAT cpu-sampling/ctxsw
# (nhung cai do moi can perf_event_paranoid/root). Cho launch-gap + per-nvtx sach.
# Skip an toan neu khong co nsys -> van du cong cu (tang 4 da du X-vs-Y qua wall-busy).
set -u
PY=${PY:-/venv/main/bin/python}
OUT=${OUT:-/root/compute_bd}
HERE="$(cd "$(dirname "$0")" && pwd)"

if ! command -v nsys >/dev/null 2>&1; then
  echo "[nsys] KHONG CO -> skip (dung wall-busy tang 4 la du de tach X-vs-Y)"; exit 0
fi
mkdir -p "$OUT"
for cfg in noquant fp8; do
  echo "===== nsys $cfg (no-root: --sample=none) ====="
  nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx \
    --force-overwrite=true -o "$OUT/nsys_$cfg" \
    $PY "$HERE/profile_compute_ab.py" --config "$cfg" --out "$OUT" --reps 1 \
    > "$OUT/nsys_${cfg}.log" 2>&1 || echo "  [nsys $cfg] loi -> xem $OUT/nsys_${cfg}.log"
  # xuat stats gpu-kernel + nvtx (parse tay sau, hoac mo .nsys-rep tren may co GUI)
  nsys stats --report cuda_gpu_kern_sum,nvtx_sum "$OUT/nsys_$cfg".nsys-rep \
    > "$OUT/nsys_${cfg}_stats.txt" 2>/dev/null && echo "  stats -> $OUT/nsys_${cfg}_stats.txt"
done
