#!/usr/bin/env python3
"""FORENSIC: rút ra RULE chi phối chunk prefill mỗi step.
Giả thuyết cần kiểm: chunk = floor((budget − decode_tokens) / block) × block
  budget = max_num_batched_tokens (biết theo config)
  decode_tokens ≈ n_running − n_prefilling  (mỗi decoder 1 token/step)
  block = 1072 (fp8)
So chunk THỰC (tokens_sched − decode_tokens) với chunk DỰ ĐOÁN của rule.
"""
import json, sys, statistics as st
from collections import Counter, defaultdict

BLOCK = 1072

def load(p): return [json.loads(l) for l in open(p) if l.strip()]

def forensic(path, label, budget):
    rows = load(path)[1:]
    pre = [r for r in rows if r["n_prefilling"] > 0]
    if not pre:
        print(f"{label}: 0 prefill steps"); return

    # phân bố chunk thực + kiểm rule
    exact = 0
    chunk_hist = Counter()
    by_decode = defaultdict(list)   # decode_tokens -> list(chunk)
    for r in pre:
        dec = max(0, r["n_running"] - r["n_prefilling"])
        chunk = r["tokens_sched"] - dec
        pred = ((budget - dec) // BLOCK) * BLOCK
        if chunk == pred:
            exact += 1
        chunk_hist[chunk] += 1
        by_decode[dec].append(chunk)

    print(f"\n{'='*66}\n{label}  budget={budget}  block={BLOCK}  ({len(pre)} prefill-steps)\n{'='*66}")
    print(f"  RULE khớp (chunk == floor((budget−dec)/block)×block): {exact}/{len(pre)} "
          f"= {100*exact/len(pre):.0f}%")
    print(f"  chunk_hist (chunk_thuc × số lần):")
    for c, n in sorted(chunk_hist.items(), key=lambda x: -x[1])[:6]:
        print(f"      {c:5d} tok = {c/BLOCK:.2f} block  × {n}")
    print(f"  mean prefill_chunk = {st.mean([c for r in pre for c in [r['tokens_sched']-max(0,r['n_running']-r['n_prefilling'])]]):.0f} tok")
    print(f"  #prefill-steps = {len(pre)}   (ít hơn = xả prefill nhanh hơn)")
    # decode load ảnh hưởng chunk thế nào (chọn vài mức decode)
    print(f"  chunk theo decode_load (n_running−n_prefill):")
    for dec in sorted(by_decode)[:8]:
        cs = by_decode[dec]
        if len(cs) >= 2:
            pred = ((budget - dec) // BLOCK) * BLOCK
            print(f"      dec={dec:2d}: chunk_mean={st.mean(cs):5.0f}  ({st.mean(cs)/BLOCK:.1f}blk)  "
                  f"rule_pred={pred} ({pred//BLOCK}blk)  n={len(cs)}")

if __name__ == "__main__":
    for a in sys.argv[1:]:
        lbl, rest = a.split("=", 1)
        budget, path = rest.split(":", 1)
        forensic(path, lbl, int(budget))
