#!/usr/bin/env python3
"""Per-rep ENV GATE — quyết định 1 rep có chạy trong env ỔN ĐỊNH không.

Đọc *_samples.json (sampler 0.5s của replay). Nếu env nhiễu (GPU throttle / clock
tụt / CPU throttle / neighbor ồn) -> FAIL -> driver nên CHẠY LẠI rep đó, không
đưa vào median. Mục đích: đảm bảo mọi rep so sánh được với nhau (cùng env).

exit 0 = PASS, exit 1 = FAIL (in lý do).

    python3 env_gate.py --samples X_samples.json
Ngưỡng chỉnh bằng cờ; mặc định hợp cho L40S/H200 rented.
"""
import argparse
import json
import statistics as st
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cpu-throttle-max", type=float, default=1.0,
                    help="%% throttle CPU tối đa cho phép trong 1 window")
    ap.add_argument("--sm-clock-drop", type=float, default=0.05,
                    help="sm_clock_min được phép tụt tối đa (tỉ lệ) so với mean")
    ap.add_argument("--load-jump", type=float, default=1.5,
                    help="load1 cuối / load1 đầu vượt ngưỡng => nghi neighbor ồn")
    args = ap.parse_args()

    sm = json.load(open(args.samples))
    if len(sm) < 3:
        print("GATE ? quá ít sample, bỏ qua"); sys.exit(0)

    def col(k):
        return [s[k] for s in sm if isinstance(s.get(k), (int, float))]

    fails = []
    warns = []

    # 1. CPU throttle (steal / cgroup cap) — hạng nhất vì residual=tokenize CPU-bound
    thr = col("cpu_throttled_pct_of_window")
    if thr and max(thr) > args.cpu_throttle_max:
        fails.append(f"CPU throttle max={max(thr):.2f}% > {args.cpu_throttle_max}% "
                     f"(residual/TTFT sẽ nhiễu)")

    # 2. GPU clock tụt (thermal/power throttle giữa chừng)
    clk = col("sm_clock_mhz")
    if clk:
        mean_c = st.mean(clk); mn = min(clk)
        if mn < mean_c * (1 - args.sm_clock_drop):
            fails.append(f"sm_clock tụt: min={mn:.0f} < {(1-args.sm_clock_drop)*100:.0f}%×mean({mean_c:.0f})")

    # 3. Neighbor ồn: load nhảy trong lúc bench
    ld = col("load1")
    if len(ld) >= 4:
        head = st.mean(ld[:2]); tail = st.mean(ld[-2:])
        if head > 0.2 and tail > head * args.load_jump:
            warns.append(f"load1 {head:.1f}->{tail:.1f} (nghi neighbor/ồn)")

    # 4. GPU util quá thấp = có thể bị nghẽn phía client, không phải server
    util = col("gpu_util_pct")
    if util and st.mean(util) < 20:
        warns.append(f"gpu_util mean={st.mean(util):.0f}% thấp (nghi nghẽn CPU/client)")

    tag = f"[thr_max={max(thr) if thr else 0:.2f}% clk_min={min(clk) if clk else 0:.0f} " \
          f"cpu_use={st.mean(col('cpu_usage_pct_of_window') or [0]):.0f}%]"
    if fails:
        print(f"GATE FAIL {tag} :: " + " | ".join(fails) + ("  warn: " + " | ".join(warns) if warns else ""))
        sys.exit(1)
    print(f"GATE PASS {tag}" + ("  warn: " + " | ".join(warns) if warns else ""))
    sys.exit(0)


if __name__ == "__main__":
    main()
