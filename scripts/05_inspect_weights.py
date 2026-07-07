#!/usr/bin/env python3
"""Read the ACTUAL weight tensors from the safetensors header — the ground truth
of what is in the model — with zero heavy deps (no torch/transformers). It parses
the safetensors JSON header (tensor name, shape, dtype) and groups by layer.

Purpose: see exactly which tensors exist per layer and their dtype, so we can plan
FP8 quantization precisely — WHICH tensors are plain Linear GEMMs (good FP8 targets)
vs conv1d / norms / SSM state (must stay high precision).

Usage:
    ./.venv/bin/python scripts/05_inspect_weights.py
"""
import glob
import json
import re
import struct
import sys
from collections import defaultdict, Counter

GLOB = "serve/models/qwen3.5-2b/model*.safetensors"


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]     # header length (uint64 LE)
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def classify(name):
    """Bucket a tensor by role — drives the FP8 plan."""
    n = name
    if n.startswith("visual") or "vision" in n:
        return "VISION (unused for text trace)"
    if "mtp" in n:
        return "MTP head (speculative decode)"
    if "conv1d" in n:
        return "conv1d  (KEEP high precision)"
    if re.search(r"(norm|layernorm|\.ln)", n):
        return "norm    (KEEP)"
    if "embed" in n or "lm_head" in n:
        return "embed/lm_head"
    if re.search(r"(A_log|dt_bias|\.dt|D\b|ssm|beta|gate_proj_bias)", n):
        return "SSM/gate params (KEEP)"
    if n.endswith(".weight") and re.search(
        r"(q_proj|k_proj|v_proj|o_proj|qkv|gate_proj|up_proj|down_proj|"
        r"in_proj|out_proj|proj\.weight|fc|mlp)", n):
        return "LINEAR GEMM  <-- FP8 target"
    return "other"


def main():
    hits = sorted(glob.glob(GLOB))
    if not hits:
        sys.exit(f"No safetensors under {GLOB} — run scripts/01_check_model.sh to see what's missing")

    header = {}
    for p in hits:
        header.update(read_header(p))
    print(f"shards: {len(hits)}  |  total tensors: {len(header)}")

    # Global dtype histogram
    dtypes = Counter(v["dtype"] for v in header.values())
    print(f"dtypes: {dict(dtypes)}")

    # Per-layer view of the text decoder (language_model.layers.N or model.layers.N)
    layer_re = re.compile(r"(?:language_model|model)\.layers\.(\d+)\.(.*)")
    per_layer = defaultdict(dict)
    non_layer = {}
    for name, meta in header.items():
        m = layer_re.search(name)
        if m:
            per_layer[int(m.group(1))][m.group(2)] = meta
        else:
            non_layer[name] = meta

    # Print two representative decoder layers: one linear (0) and one full-attn (3)
    for idx in sorted(per_layer)[:8]:
        subs = per_layer[idx]
        kind = "FULL-ATTN" if any("self_attn" in k and "conv1d" not in k for k in subs) \
                             and not any("conv1d" in k for k in subs) else "LINEAR-ATTN(GDN)"
        # Better heuristic: conv1d present => linear/GDN layer
        kind = "LINEAR-ATTN (Gated DeltaNet)" if any("conv1d" in k for k in subs) else "FULL-ATTENTION"
        print(f"\n── decoder layer {idx}  [{kind}] ── {len(subs)} tensors")
        for sub, meta in sorted(subs.items()):
            print(f"    {meta['dtype']:>6}  {str(meta['shape']):<20} {sub}")

    print("\n── non-decoder tensors (embeddings / norms / vision / mtp / lm_head) ──")
    buckets = defaultdict(list)
    for name, meta in non_layer.items():
        buckets[classify(name)].append((name, meta))
    for b in sorted(buckets):
        items = buckets[b]
        print(f"  [{b}] — {len(items)} tensors, e.g.:")
        for name, meta in items[:3]:
            print(f"      {meta['dtype']:>6}  {str(meta['shape']):<18} {name}")

    # FP8 target accounting: how many params sit in plain Linear GEMMs?
    def numel(shape):
        n = 1
        for s in shape:
            n *= s
        return n

    role_params = Counter()
    for name, meta in header.items():
        role_params[classify(name)] += numel(meta["shape"])
    total = sum(role_params.values())
    print(f"\n── PARAMETER BUDGET (total {total/1e9:.3f} B params) ──")
    for role, p in sorted(role_params.items(), key=lambda x: -x[1]):
        print(f"  {p/1e9:7.3f} B  {100*p/total:5.1f}%  {role}")
    fp8 = role_params.get("LINEAR GEMM  <-- FP8 target", 0)
    print(f"\n  >> ~{100*fp8/total:.0f}% of params live in plain Linear GEMMs = the FP8-quantizable")
    print("     compute-bound core (prefill bottleneck). conv1d/norms/SSM/embeddings stay high-precision.")


if __name__ == "__main__":
    main()
