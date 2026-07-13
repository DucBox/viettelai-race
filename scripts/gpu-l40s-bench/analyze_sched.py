#!/usr/bin/env python3
"""Bóc tách queue từ sched_trace.jsonl (per-step). So sánh nhiều config.
Trả lời: mỗi iteration mất bao lâu, budget có nghẽn không, step prefill vs
decode chênh bao nhiêu, và burst (n_waiting>0) ngốn bao nhiêu wall-clock."""
import json, sys, statistics as st

def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs)-1, int(p*len(xs)))] if xs else 0

def analyze(path, label):
    rows = load(path)
    if not rows:
        print(f"{label}: rỗng"); return
    # bỏ step đầu (gap=0). burst = các step có n_waiting>0 hoặc n_prefilling>0
    steps = rows[1:]
    burst = [r for r in steps if r["n_waiting"] > 0 or r["n_prefilling"] > 0]
    decode_only = [r for r in steps if r["n_prefilling"] == 0 and r["n_waiting"] == 0]
    has_prefill = [r for r in steps if r["n_prefilling"] > 0]

    print(f"\n{'='*70}\n{label}  ({len(rows)} steps)\n{'='*70}")
    if burst:
        t_span = burst[-1]["t"] - burst[0]["t"]
        exec_burst = [r["exec_gap_ms"] for r in burst if r["exec_gap_ms"] > 0]
        print(f"BURST (n_waiting>0 hoặc còn prefill): {len(burst)} steps, "
              f"wall-clock {t_span*1000:.0f}ms")
        print(f"  step exec time  : mean {st.mean(exec_burst):.1f}ms  median {st.median(exec_burst):.1f}  "
              f"p90 {pct(exec_burst,.9):.1f}  max {max(exec_burst):.1f}")
        print(f"  tokens_sched    : mean {st.mean([r['tokens_sched'] for r in burst]):.0f}  "
              f"max {max(r['tokens_sched'] for r in burst)}  "
              f"(so max_num_batched_tokens để biết nghẽn budget?)")
        print(f"  n_running       : mean {st.mean([r['n_running'] for r in burst]):.1f}  max {max(r['n_running'] for r in burst)}")
        print(f"  n_waiting       : max {max(r['n_waiting'] for r in burst)}")
        print(f"  n_prefilling    : mean {st.mean([r['n_prefilling'] for r in burst]):.1f}  max {max(r['n_prefilling'] for r in burst)}")
        admits = [r for r in burst if r["n_new_admit"] > 0]
        print(f"  admission events: {len(admits)} step có admit mới  "
              f"(tổng admit {sum(r['n_new_admit'] for r in admits)})")

    # step time: prefill-mixed vs decode-only
    if has_prefill and decode_only:
        ep = [r["exec_gap_ms"] for r in has_prefill if r["exec_gap_ms"] > 0]
        ed = [r["exec_gap_ms"] for r in decode_only if r["exec_gap_ms"] > 0]
        print(f"  STEP TIME theo loại:")
        print(f"    mixed-prefill steps (n={len(ep)}): mean {st.mean(ep):.1f}ms")
        print(f"    decode-only  steps (n={len(ed)}): mean {st.mean(ed):.1f}ms")

    # step time vs n_running bins (test: batch to => step lâu hơn?)
    print(f"  STEP TIME vs n_running (chỉ step decode-only, cô lập batch decode):")
    bins = {}
    for r in decode_only:
        if r["exec_gap_ms"] <= 0: continue
        b = r["n_running"]
        bins.setdefault(b, []).append(r["exec_gap_ms"])
    for b in sorted(bins):
        if len(bins[b]) >= 3:
            print(f"    n_running={b:2d}: step {st.mean(bins[b]):.1f}ms  (n={len(bins[b])})")

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        analyze(path, label)
