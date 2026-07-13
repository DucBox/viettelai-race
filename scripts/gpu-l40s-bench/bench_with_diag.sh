#!/bin/bash
# Chup trang thai host TRUOC va SAU khi bench, tinh delta trong dung cua so
# thoi gian chay replay -- de biet TTFT cao/thap co tuong quan voi throttle
# / load tai THOI DIEM do khong, thay vi doan mo.

snapshot() {
  echo "LOADAVG:$(cut -d' ' -f1-3 /proc/loadavg)"
  echo "CPUSTAT:$(cat /sys/fs/cgroup/cpu.stat | tr '\n' ' ')"
  nvidia-smi --query-gpu=utilization.gpu,clocks.sm,temperature.gpu,power.draw --format=csv,noheader
}

echo "=== BEFORE ==="
snapshot > /tmp/before.snap
cat /tmp/before.snap

echo "=== RUNNING REPLAY ==="
cd /root && /usr/bin/python3 replay_trace.py

echo "=== AFTER ==="
snapshot > /tmp/after.snap
cat /tmp/after.snap

echo "=== DELTA (trong dung cua so replay nay) ==="
python3 - << 'PYEOF'
def parse_cpustat(line):
    parts = line.replace("CPUSTAT:", "").split()
    d = {}
    for i in range(0, len(parts) - 1, 2):
        try:
            d[parts[i]] = int(parts[i + 1])
        except ValueError:
            pass
    return d

before = open("/tmp/before.snap").read().splitlines()
after = open("/tmp/after.snap").read().splitlines()

cb = parse_cpustat([l for l in before if l.startswith("CPUSTAT")][0])
ca = parse_cpustat([l for l in after if l.startswith("CPUSTAT")][0])

d_periods = ca["nr_periods"] - cb["nr_periods"]
d_throttled = ca["nr_throttled"] - cb["nr_throttled"]
d_throttled_usec = ca["throttled_usec"] - cb["throttled_usec"]
d_usage_usec = ca["usage_usec"] - cb["usage_usec"]

print(f"So chu ky (periods) trong luc bench : {d_periods}")
print(f"So chu ky BI THROTTLE               : {d_throttled} ({d_throttled/d_periods*100 if d_periods else 0:.1f}%)")
print(f"Thoi gian bi throttle (ms)           : {d_throttled_usec/1000:.1f}")
print(f"Thoi gian thuc su duoc chay (ms)     : {d_usage_usec/1000:.1f}")
print(f"Ty le throttle/usage                : {d_throttled_usec/d_usage_usec*100 if d_usage_usec else 0:.1f}%")
PYEOF
