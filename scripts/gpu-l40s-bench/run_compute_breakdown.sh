#!/bin/bash
# ORCHESTRATOR compute-breakdown (TANG 4-5) — KHONG SUDO. Chay sau khi da co
# run_v28_baseline.sh (tang 1-3: cay TTFT/TPOT wall). Cai nay bo sung:
#   T3b layer-group | T3c token-tax | T3d overhead(wall-busy) | D1 decode compute
# Yeu cau tren box: /root/model , scripts/gpu-l40s-bench/*.py|*.sh o /root
set -u
PY=${PY:-/venv/main/bin/python}
OUT=${OUT:-/root/compute_bd}
HERE="$(cd "$(dirname "$0")" && pwd)"
NTOK=${NTOK:-12947}     # = turn0/user0 uncached; doi neu muon shape khac
REPS=${REPS:-5}
mkdir -p "$OUT"

echo "########## COMPUTE BREAKDOWN (no-sudo) ##########"
bash "$HERE/measure_preflight.sh" start          # bat sampler clock (log throttle)
trap 'bash "$HERE/measure_preflight.sh" stop' EXIT

for cfg in noquant fp8; do
  echo "=================== profile $cfg ==================="
  $PY "$HERE/profile_compute_ab.py" --config "$cfg" --out "$OUT" --ntok "$NTOK" --reps "$REPS" \
    2>&1 | tee "$OUT/profile_${cfg}.log"
done

echo "=================== PARSE (fp8 vs noquant) ==================="
$PY "$HERE/parse_compute.py" --dir "$OUT" | tee "$OUT/compute_breakdown.txt"

echo "=================== nsys (neu co, no-root) ==================="
OUT="$OUT" PY="$PY" bash "$HERE/nsys_compute.sh"

echo ""
echo "XEM:"
echo "  $OUT/compute_breakdown.txt   (op-cat + layer-group + WALL/OVERHEAD, prefill & decode)"
echo "  $OUT/clean_*.json            (wall sach per config)"
echo "  $OUT/gpuclock_samples.csv    (clock/nhiet/power — loai mau throttle)"
echo "  $OUT/nsys_*_stats.txt        (neu co nsys)"
echo "COMPUTE_BREAKDOWN_DONE"
