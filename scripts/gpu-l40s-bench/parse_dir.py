import re, json, gzip, glob, os, sys
from collections import defaultdict
D=sys.argv[1]
CAT=[("GEMM (matmul)",  r"cutlass_scaled_mm|cutlass::Kernel2|scaled_mm|aten::mm|gemvx|internal::gemv|gemm|ampere_|sm80_|sm89_|cublas|cutlass.*Kernel"),
     ("GDN conv1d",     r"causal_conv1d|post_conv"),
     ("GDN scan/state", r"ChunkGatedDelta|chunk_fwd|chunk_gated_delta|recompute_w_u|chunk_scaled_dot|merge_16x16|qwen_gdn|cumsum|fused_recurrent|_gdn|solve_tril|wy_fast"),
     ("Attention",      r"unified_attention|flashinfer|BatchPrefill|BatchDecode|paged|_attn|flash"),
     ("KV write",       r"reshape_and_cache|slot_mapping"),
     ("Copy/mem/misc",  r"copy_|Memcpy|cudaDevice|elementwise|poi_fused|red_fused|vectorized|gather|silu|act")]
fs=sorted(glob.glob(f"{D}/*.json.gz"), key=os.path.getmtime)
for lab,f in zip(["PREFILL","DECODE"],fs):
    ev=[e for e in json.loads(gzip.open(f).read())["traceEvents"] if e.get("ph")=="X" and e.get("cat")=="kernel"]
    g=defaultdict(float); tot=0.0; oth=defaultdict(float)
    for e in ev:
        dur=e.get("dur",0); tot+=dur; nm=e.get("name","")
        for c,p in CAT:
            if re.search(p,nm,re.I): g[c]+=dur; break
        else: oth[nm]+=dur
    print(f"=== {lab} [{os.path.basename(D)}]  {tot/1000:.1f}ms ({len(ev)}k) ===")
    for c,_ in CAT:
        if g[c]: print(f"  {c:15}{g[c]/1000:>8.1f}ms{100*g[c]/tot:>6.1f}%")
    oo=sum(oth.values())
    if oo>tot*0.03: print(f"  {'OTHER':15}{oo/1000:>8.1f}ms{100*oo/tot:>6.1f}%  " + ", ".join(f"{n[:22]}={v/1000:.1f}" for n,v in sorted(oth.items(),key=lambda x:-x[1])[:3]))
