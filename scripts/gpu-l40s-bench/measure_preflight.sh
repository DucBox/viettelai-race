#!/bin/bash
# PREFLIGHT do luong — KHONG SUDO. Xac nhan du cong cu + BAT sampler clock/nhiet.
# Vi KHONG lock duoc clock (can root), ta LOG clock de sau loai mau bi throttle.
#   bash measure_preflight.sh start   -> in moi truong + bat sampler nen (ghi PID)
#   bash measure_preflight.sh stop    -> tat sampler
set -u
PY=${PY:-/venv/main/bin/python}
OUT=${OUT:-/root/compute_bd}
SAMP_CSV="$OUT/gpuclock_samples.csv"
SAMP_PIDF="$OUT/gpuclock.pid"
mkdir -p "$OUT"

start() {
  echo "===== PREFLIGHT (no-sudo) ====="
  echo "python: $PY"
  $PY - <<'EOF'
import importlib,sys
for m in ("vllm","torch"):
    try:
        mod=importlib.import_module(m); print(f"  {m}: {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m}: MISSING ({e})"); sys.exit(1)
import torch
print("  cuda avail:", torch.cuda.is_available(),
      "| dev:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
      "| cap:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "-")
EOF
  command -v nsys >/dev/null 2>&1 && echo "  nsys: $(nsys --version 2>/dev/null | head -1)" || echo "  nsys: KHONG CO (se skip tang 5, van du cong cu)"
  # perf_event_paranoid: chi de BIET nsys --sample=cpu co chay khong; ta luon dung --sample=none
  echo "  perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo '?') (ta dung --sample=none nen khong can)"
  echo "  LUU Y: khong lock clock -> chi LOG. So sanh A/B PHAI trong cung 1 lan thue."

  # sampler nen: clock SM/mem, temp, power, util moi 1s (userspace, khong root)
  ( echo "ts,sm_mhz,mem_mhz,temp_c,power_w,util_pct"
    while true; do
      nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | sed "s/^/$(date +%s),/"
      sleep 1
    done ) >> "$SAMP_CSV" &
  echo $! > "$SAMP_PIDF"
  echo "  sampler clock -> $SAMP_CSV (PID $(cat $SAMP_PIDF))"
}
stop() {
  [ -f "$SAMP_PIDF" ] && kill "$(cat $SAMP_PIDF)" 2>/dev/null && rm -f "$SAMP_PIDF" && echo "sampler stopped"
}
case "${1:-start}" in start) start;; stop) stop;; *) echo "usage: $0 start|stop";; esac
