#!/usr/bin/env python3
"""Parse trace cua profile_compute_ab.py -> CAY COMPUTE fp8 vs noquant.

Tach:
  - PHA prefill vs decode  (theo user_annotation execute_context: T>0 = prefill, T==0 = decode)
  - OP-CATEGORY gpu_busy    (GEMM / FP8_quant_tax / GDN-core / conv / FullAttn / KV_write)
  - LAYER-GROUP (neu co NVTX GRP:*)  -> GEMM chia GDN/ATTN/MLP
  - OVERHEAD = wall(clean json) - gpu_busy   [T3d]

  PY parse_compute.py --dir /root/compute_bd
"""
import gzip, json, glob, os, re, bisect, argparse
from collections import defaultdict

ANN = re.compile(r"execute_context_\d+\((\d+)\)_generation_\d+\((\d+)\)")
GRP = re.compile(r"GRP:(\w+)")

def bucket(n):
    n = n.lower()
    if "triton_red_fused" in n and ("scaled_mm" in n or "clamp" in n or "abs" in n):
        return "FP8_quant_tax"
    if "cutlass" in n or "aten::mm" in n or "gemvx" in n or "tensorop" in n or "scaled_mm" in n or "gemm" in n:
        return "GEMM"
    if any(k in n for k in ("qwen_gdn_attention_core","gated_delta","chunkgated","chunk_fwd",
                            "chunk_gated","merge_16x16","recompute","recurrent")):
        return "GDN_core"
    if "conv1d" in n: return "GDN_conv1d"
    if "unified_attention" in n or "flashinfer" in n or "batchprefill" in n or "mergestate" in n:
        return "FullAttn"
    if "reshape_and_cache" in n or "cache_flash" in n or "zero_kv" in n or "slot_mapping" in n:
        return "KV_write"
    return "other"

def load_events(path):
    d = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    return d["traceEvents"] if isinstance(d, dict) else d

def _pick(prof_dir, tag):
    """Chon file trace: uu tien co _{tag}. (PF/DEC), fallback file lon nhat."""
    fs = glob.glob(os.path.join(prof_dir, f"*_{tag}.*pt.trace.json*"))
    if not fs:
        fs = glob.glob(os.path.join(prof_dir, "*.pt.trace.json*"))
    return sorted(fs, key=os.path.getsize)[-1] if fs else None

def analyze(prof_dir, clean):
    # PF file -> phase prefill ; DEC file -> phase decode (tach session, khong nhiem)
    pf_path, dec_path = _pick(prof_dir, "PF"), _pick(prof_dir, "DEC")
    r_pf = _analyze_file(pf_path, "prefill") if pf_path else None
    r_dec = _analyze_file(dec_path, "decode") if dec_path else None
    if not r_pf and not r_dec: return None
    agg = {"prefill": (r_pf or {}).get("agg", {}), "decode": (r_dec or {}).get("agg", {})}
    busy = {"prefill": (r_pf or {}).get("busy", 0.0), "decode": (r_dec or {}).get("busy", 0.0)}
    grp = {"prefill": (r_pf or {}).get("grp", {}), "decode": (r_dec or {}).get("grp", {})}
    return {"agg": agg, "busy": busy, "grp_gemm": grp, "n_dec": (r_dec or {}).get("n_win", 1)}

def _analyze_file(path, keep_phase):
    if not path: return None
    ev = load_events(path)
    # phase windows + grp windows
    pw, gw = [], []
    for e in ev:
        if e.get("cat") == "user_annotation":
            m = ANN.search(str(e.get("name","")))
            if m:
                T, S = int(m.group(1)), int(m.group(2))
                pw.append((e["ts"], e["ts"]+e["dur"], "prefill" if T>0 else "decode")); continue
        if e.get("cat") in ("user_annotation","gpu_user_annotation"):
            g = GRP.search(str(e.get("name","")))
            if g: gw.append((e["ts"], e["ts"]+e["dur"], g.group(1)))
    pw.sort(); gw.sort()
    ps = [w[0] for w in pw]; gs = [w[0] for w in gw]
    n_dec = max(1, sum(1 for w in pw if w[2] == "decode"))  # so step decode -> chia ve per-step
    def find(wins, starts, ts):
        c = bisect.bisect_right(starts, ts) - 1
        for i in (c, c-1):
            if 0 <= i < len(wins) and wins[i][0] <= ts <= wins[i][1]:
                return wins[i][2]
        return None
    n_win = max(1, sum(1 for w in pw if w[2] == keep_phase))
    agg = defaultdict(float); busy = [0.0]; grp = defaultdict(float)
    for e in ev:
        if e.get("ph") != "X": continue
        if str(e.get("cat","")).lower() not in ("kernel","gpu_memcpy","gpu_memset"): continue
        ts, dur = e.get("ts"), e.get("dur",0)/1000.0
        if find(pw, ps, ts) != keep_phase: continue   # CHI giu phase can (bo prefill cua batch-decode)
        b = bucket(e.get("name",""))
        agg[b] += dur; busy[0] += dur
        if b == "GEMM":
            g = find(gw, gs, ts)
            if g: grp[g] += dur
    # decode: chia ve PER-STEP; prefill giu tong (1 request)
    if keep_phase == "decode":
        for k in list(agg): agg[k] /= n_win
        for k in list(grp): grp[k] /= n_win
        busy[0] /= n_win
    return {"agg": agg, "busy": busy[0], "grp": grp, "n_win": n_win}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="/root/compute_bd")
    a = ap.parse_args()
    res = {}
    for cfg in ("fp8", "noquant"):
        cj = os.path.join(a.dir, f"clean_{cfg}.json")
        clean = json.load(open(cj)) if os.path.exists(cj) else {}
        r = analyze(os.path.join(a.dir, f"prof_{cfg}"), clean)
        if r: res[cfg] = (r, clean)

    order = ["GEMM","FP8_quant_tax","GDN_core","GDN_conv1d","FullAttn","KV_write","other"]
    for ph in ("prefill","decode"):
        print("\n" + "="*72); print(f"  {ph.upper()}  — gpu_busy theo op (ms) + WALL/OVERHEAD")
        print("="*72)
        hdr = f"{'component':16}"+"".join(f"{c:>12}" for c in res); print(hdr)
        for c in order:
            row = f"{c:16}"
            for cfg in res:
                row += f"{res[cfg][0]['agg'][ph].get(c,0):>12.1f}"
            print(row)
        # busy / wall / overhead
        print("-"*72)
        for lab, key in [("gpu_busy_tot", None)]:
            row = f"{'gpu_busy_tot':16}"
            for cfg in res: row += f"{res[cfg][0]['busy'][ph]:>12.1f}"
            print(row)
        wl = f"{'WALL(clean)':16}"; ov = f"{'OVERHEAD=w-busy':16}"
        for cfg in res:
            _, clean = res[cfg]
            w = clean.get("prefill_wall_median_ms") if ph=="prefill" else clean.get("decode_wall_per_step_ms")
            b = res[cfg][0]['busy'][ph]
            wl += f"{(w or 0):>12.1f}"; ov += f"{((w or 0)-b):>12.1f}"
        print(wl); print(ov)
        # layer-group GEMM (neu co nvtx)
        gg = {cfg: res[cfg][0]['grp_gemm'][ph] for cfg in res}
        if any(gg[c] for c in gg):
            print(f"  --- GEMM chia layer-group (NVTX) ---")
            for g in ("MLP","GDN","ATTN"):
                row = f"  GEMM:{g:11}"
                for cfg in res: row += f"{gg[cfg].get(g,0):>12.1f}"
                print(row)
    print("\nLuu y: WALL do KHONG profiler (sach); gpu_busy do TU trace. "
          "OVERHEAD lon = launch/dispatch (Ada khong fuse quant).")

if __name__ == "__main__":
    main()
