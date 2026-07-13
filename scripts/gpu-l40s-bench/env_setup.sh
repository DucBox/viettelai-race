#!/usr/bin/env bash
# Preflight ỔN ĐỊNH ENV cho bench — chạy 1 LẦN sau khi thuê GPU, TRƯỚC mọi A/B.
# Mục tiêu: mọi rep chạy trong cùng 1 env make-sense (GPU clock cố định, CPU
# governor performance, GPU độc quyền, tách core server<->client). Best-effort:
# thiếu quyền thì CẢNH BÁO chứ không chết — phần không khóa được thì env_gate.py
# canh per-rep.
#
#   source env_setup.sh        # source để lấy $SRV_PIN / $CLI_PIN
# Sau đó:
#   $SRV_PIN <lệnh serve vLLM>          # server chạy trên core riêng
#   $CLI_PIN python3 replay_...         # client+sampler chạy trên core KHÁC
set -u
echo "############ ENV PREFLIGHT ############"

# ---------- 1. GPU độc quyền? ----------
NPROC_GPU=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || echo 0)
MIG=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null | head -1)
echo "[GPU] process đang chạy: $NPROC_GPU   MIG mode: ${MIG:-N/A}"
[ "$NPROC_GPU" -gt 0 ] && echo "  ⚠️  CÓ process GPU khác — kill sạch trước khi bench (bench sẽ vô nghĩa nếu share)."

# ---------- 2. Persistence + KHÓA GPU CLOCK ----------
sudo -n nvidia-smi -pm 1 >/dev/null 2>&1 && echo "[GPU] persistence mode ON" || echo "[GPU] ⚠️ không set được persistence (thiếu quyền) — bỏ qua"
# clock tối đa hỗ trợ
MAXC=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "${MAXC:-}" ] && nvidia-smi -lgc "${MAXC},${MAXC}" >/dev/null 2>&1; then
  echo "[GPU] ĐÃ KHÓA sm clock = ${MAXC}MHz (cố định, hết boost drift)"
elif [ -n "${MAXC:-}" ] && sudo -n nvidia-smi -lgc "${MAXC},${MAXC}" >/dev/null 2>&1; then
  echo "[GPU] ĐÃ KHÓA sm clock = ${MAXC}MHz (qua sudo)"
else
  echo "[GPU] ⚠️ KHÔNG khóa được clock (rented thường cấm) → dựa vào: interleaved A/B + env_gate loại rep bị throttle"
fi

# ---------- 3. CPU governor = performance ----------
GOVOK=0
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [ -w "$g" ] && echo performance > "$g" 2>/dev/null && GOVOK=1
done
if [ "$GOVOK" = 1 ]; then echo "[CPU] governor = performance"
else sudo -n bash -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done' 2>/dev/null \
     && echo "[CPU] governor = performance (sudo)" || echo "[CPU] ⚠️ không set được governor — env_gate canh cpu_throttle"
fi

# ---------- 4. Chia CORE server <-> client (chống client giành CPU lúc tokenize) ----------
NCPU=$(nproc)
if [ "$NCPU" -ge 6 ]; then
  CLI_N=2
elif [ "$NCPU" -ge 4 ]; then
  CLI_N=1
else
  CLI_N=0
fi
if [ "$CLI_N" -gt 0 ]; then
  SRV_LAST=$((NCPU - CLI_N - 1)); CLI_FIRST=$((NCPU - CLI_N)); CLI_LAST=$((NCPU - 1))
  export SRV_PIN="taskset -c 0-${SRV_LAST}"
  export CLI_PIN="taskset -c ${CLI_FIRST}-${CLI_LAST}"
  echo "[CPU] $NCPU core → server=0-${SRV_LAST}  client=${CLI_FIRST}-${CLI_LAST}"
  echo "      dùng:  \$SRV_PIN <serve>   và   \$CLI_PIN <replay>"
else
  export SRV_PIN=""; export CLI_PIN=""
  echo "[CPU] ⚠️ chỉ $NCPU core — KHÔNG tách được server/client. residual (tokenize) SẼ nhiễu; A/B TTFT kém tin cậy. Nên thuê box ≥6 core."
fi
# ghi ra file để driver source lại
echo "export SRV_PIN='${SRV_PIN}'; export CLI_PIN='${CLI_PIN}'" > /root/env_pins.sh 2>/dev/null || true

echo "[env] baseline: $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null | head -1) | load:$(cut -d' ' -f1-3 /proc/loadavg)"
echo "######################################"
