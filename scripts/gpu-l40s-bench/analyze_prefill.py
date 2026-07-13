#!/usr/bin/env python3
"""Bóc chunk prefill thực mỗi step từ sched trace. Xác nhận block-tiling 1072
và giả thuyết 'margin'. prefill_chunk ≈ tokens_sched − decode_tokens, với
decode_tokens ≈ (n_running − n_prefilling) (mỗi decoder ~1 token/step)."""
import json, sys, statistics as st
from collections import Counter

def load(p): return [json.loads(l) for l in open(p) if l.strip()]

def analyze(path, label, block=1072):
    rows = load(path)[1:]
    pre = [r for r in rows if r["n_prefilling"] > 0]
    if not pre:
        print(f"{label}: không có step prefill"); return
    chunks = []
    for r in pre:
        dec = max(0, r["n_running"] - r["n_prefilling"])
        chunk = r["tokens_sched"] - dec
        chunks.append(chunk)
    # số block tương đương
    nblk = Counter(round(c / block) for c in chunks)
    print(f"\n{'='*64}\n{label}  block={block}, {len(pre)} prefill-steps\n{'='*64}")
    print(f"  prefill_chunk: mean {st.mean(chunks):.0f}  median {st.median(chunks):.0f}  "
          f"max {max(chunks)}  min {min(chunks)}")
    print(f"  ~số block/step (chunk/{block}): " +
          "  ".join(f"{k}blk×{v}" for k, v in sorted(nblk.items())))
    # tokens_sched thô (gồm decode) để so trần budget
    ts = [r["tokens_sched"] for r in pre]
    print(f"  tokens_sched(gồm decode): mean {st.mean(ts):.0f}  max {max(ts)}")
    # exec time của step prefill
    et = [r["exec_gap_ms"] for r in pre if r["exec_gap_ms"] > 0]
    print(f"  step prefill exec: mean {st.mean(et):.1f}ms")
    # burst wall-clock của pha prefill
    span = pre[-1]["t"] - pre[0]["t"]
    print(f"  #prefill-steps={len(pre)}  span_prefill={span*1000:.0f}ms")

if __name__ == "__main__":
    for a in sys.argv[1:]:
        lbl, p = a.split("=", 1)
        analyze(p, lbl)
