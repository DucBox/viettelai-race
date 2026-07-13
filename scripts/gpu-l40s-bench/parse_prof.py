#!/usr/bin/env python3
"""Parse trace theo prefix (PREFILL_/DECODE_) -> bảng per-op. So được nhiều config."""
import re, json, gzip, glob, sys
from collections import defaultdict
CAT=[("GEMM fp8",       r"cutlass_scaled_mm|cutlass::Kernel2|aten::mm|gemvx|internal::gemv|scaled_mm"),
     ("GDN conv1d",     r"causal_conv1d|post_conv"),
     ("GDN scan/state", r"ChunkGatedDelta|chunk_fwd|chunk_gated_delta|recompute_w_u|chunk_scaled_dot|merge_16x16|qwen_gdn|cumsum|fused_recurrent|gdn"),
     ("Attention",      r"unified_attention|flashinfer|BatchPrefill|BatchDecode|paged"),
     ("KV write",       r"reshape_and_cache|slot_mapping"),
     ("Copy/mem/misc",  r"copy_|Memcpy|cudaDevice|elementwise|poi_fused|red_fused|vectorized|gather")]
def parse(prefix):
    fs=[f for f in glob.glob("/root/prof/*.json.gz") if prefix in f]
    if not fs: return None
    f=sorted(fs)[-1]
    ev=[e for e in json.loads(gzip.open(f).read())["traceEvents"] if e.get("ph")=="X" and e.get("cat")=="kernel"]
    g=defaultdict(float); tot=0.0
    for e in ev:
        dur=e.get("dur",0); tot+=dur; nm=e.get("name","")
        for c,p in CAT:
            if re.search(p,nm,re.I): g[c]+=dur; break
    return tot,g,len(ev)
for phase in ("PREFILL","DECODE"):
    r=parse(phase)
    if not r: continue
    tot,g,n=r
    print(f"\n=== {phase}  tong {tot/1000:.1f}ms ({n} kernel) ===")
    for c,_ in CAT:
        if g[c]: print(f"  {c:16}{g[c]/1000:>8.2f}ms{100*g[c]/tot:>7.1f}%")
