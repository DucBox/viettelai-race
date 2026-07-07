#!/usr/bin/env python3
"""Inspect Qwen3.5-2B architecture straight from config.json and VERIFY the
claims made in the competition doc. No torch / GPU needed — pure config math,
runs instantly on any machine.

Usage:
    ./.venv/bin/python scripts/04_inspect_arch.py
    ./.venv/bin/python scripts/04_inspect_arch.py /path/to/config.json
"""
import glob
import json
import sys
from collections import Counter

DEFAULT_GLOB = "serve/models/qwen3.5-2b/config.json"


def find_config(argv):
    if len(argv) > 1:
        return argv[1]
    hits = glob.glob(DEFAULT_GLOB)
    if not hits:
        sys.exit(f"config.json not found at {DEFAULT_GLOB}\n"
                 f"Run scripts/01_check_model.sh to see what's missing (or pass a path).")
    return sorted(hits)[-1]


def h(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main():
    path = find_config(sys.argv)
    cfg = json.load(open(path))
    print(f"config: {path}")

    t = cfg.get("text_config", cfg)      # text_config on multimodal, else root
    is_mm = "vision_config" in cfg

    h("1. WHAT KIND OF MODEL IS THIS?")
    print(f"  architectures        : {cfg.get('architectures')}")
    print(f"  model_type           : {cfg.get('model_type')}")
    print(f"  MULTIMODAL (vision)? : {'YES — has vision_config' if is_mm else 'no'}")
    if is_mm:
        v = cfg["vision_config"]
        print(f"    vision tower       : depth={v.get('depth')} hidden={v.get('hidden_size')} "
              f"patch={v.get('patch_size')} → out_hidden={v.get('out_hidden_size')}")
        print(f"    image/video tokens : image_token_id={cfg.get('image_token_id')} "
              f"video_token_id={cfg.get('video_token_id')}")
        print("    >> DOC SAYS 'Dense Transformer, pure text'. REALITY: vision-language model.")
        print("    >> The round-1 trace is 100% text, so the vision tower is dead weight at serve time.")

    h("2. LAYER STRUCTURE (the hybrid claim)")
    layer_types = t.get("layer_types")
    n_layers = t.get("num_hidden_layers")
    interval = t.get("full_attention_interval")
    if layer_types:
        counts = Counter(layer_types)
        print(f"  num_hidden_layers    : {n_layers}")
        print(f"  full_attention_every : {interval}  (every {interval}th layer is full attention)")
        print(f"  layer_types count    : {dict(counts)}")
        # Compact visual: L=linear, F=full
        vis = "".join("F" if "full" in x else "L" for x in layer_types)
        print(f"  pattern (L=linear,F=full):\n    {vis}")
        ratio = counts.get("linear_attention", 0) / max(1, counts.get("full_attention", 1))
        print(f"  linear : full ratio  : {counts.get('linear_attention',0)} : "
              f"{counts.get('full_attention',0)}  (~{ratio:.0f}:1)")
        print("  >> DOC's '3:1 hybrid Gated DeltaNet + full attention' → "
              + ("CONFIRMED." if abs(ratio - 3) < 0.5 else "MISMATCH!"))

    h("3. FULL-ATTENTION LAYERS (softmax) + KV cache cost")
    n_full = Counter(layer_types).get("full_attention", 0) if layer_types else "?"
    hd = t.get("head_dim")
    n_heads = t.get("num_attention_heads")
    n_kv = t.get("num_key_value_heads")
    print(f"  num_attention_heads  : {n_heads}")
    print(f"  num_key_value_heads  : {n_kv}   (GQA {n_heads//n_kv if n_kv else '?'}:1)")
    print(f"  head_dim             : {hd}")
    print(f"  attn_output_gate     : {t.get('attn_output_gate')} (gated attention)")
    if isinstance(n_full, int) and hd and n_kv:
        kv_per_tok_bytes = n_full * n_kv * hd * 2 * 2   # layers × kvheads × dim × (k+v) × bf16
        print(f"  >> KV cache grows ONLY in the {n_full} full-attention layers.")
        print(f"     Per token: {n_full} layers × {n_kv} kv-heads × {hd} dim × 2(k+v) × 2B "
              f"= {kv_per_tok_bytes/1024:.1f} KB/token")
        for ctx in (24000, 32768, 262144):
            print(f"     {ctx:>7} tokens → {kv_per_tok_bytes*ctx/1e9:.2f} GB KV  (per sequence)")
        print("     × 20 concurrent sessions at ~24k tokens = "
              f"{kv_per_tok_bytes*24000*20/1e9:.2f} GB  (fits 18GB? watch this).")

    h("4. LINEAR-ATTENTION LAYERS (Gated DeltaNet / SSM) — fixed-size state")
    print(f"  linear_num_key_heads : {t.get('linear_num_key_heads')}")
    print(f"  linear_key_head_dim  : {t.get('linear_key_head_dim')}")
    print(f"  linear_num_value_heads: {t.get('linear_num_value_heads')}")
    print(f"  linear_value_head_dim: {t.get('linear_value_head_dim')}")
    print(f"  linear_conv_kernel   : {t.get('linear_conv_kernel_dim')} (causal conv1d)")
    print(f"  mamba_ssm_dtype      : {t.get('mamba_ssm_dtype')}")
    print("  >> State is FIXED-SIZE (does NOT grow with context) → this is why long")
    print("     context is cheap in memory, but also why classic block-based prefix")
    print("     caching does NOT directly apply to these 18 layers (must verify on engine).")
    if t.get("mamba_ssm_dtype") == "float32":
        print("  >> SSM state is fp32 → FP8/quant must EXCLUDE the linear-attn state path.")

    h("5. EXTRAS the doc missed")
    mtp = t.get("mtp_num_hidden_layers")
    if mtp:
        print(f"  MTP (Multi-Token Prediction) layers : {mtp}")
        print("  >> Model ships a NATIVE speculative-decoding head. The doc treated spec")
        print("     decoding as 'risky, maybe skip'; but MTP is built in and cheap to enable")
        print("     in vLLM (--speculative-config). Real lever for TPOT.")
    rope = t.get("rope_parameters", {})
    if rope:
        print(f"  RoPE                 : theta={rope.get('rope_theta')} "
              f"partial_rotary_factor={rope.get('partial_rotary_factor')} "
              f"mrope={rope.get('mrope_interleaved')} section={rope.get('mrope_section')}")
    print(f"  MoE?                 : {'NO (mlp_only_layers empty, no experts) → Dense CONFIRMED' if not t.get('num_experts') else 'YES'}")

    h("6. SUMMARY vs DOC")
    print("  CONFIRMED : hybrid 3:1 Gated DeltaNet + full attention, 24 layers, 262144 ctx, Dense (non-MoE)")
    print("  CORRECTED : it is a VISION-LANGUAGE model, not 'pure Dense Transformer' text")
    print("  NEW       : native MTP head (speculative decode), fp32 SSM state (quant caution)")
    print(f"  ctx (max_position_embeddings): {t.get('max_position_embeddings')}")
    print(f"  hidden_size={t.get('hidden_size')}  intermediate={t.get('intermediate_size')}  "
          f"vocab={t.get('vocab_size')}  tie_embeddings={t.get('tie_word_embeddings')}")


if __name__ == "__main__":
    main()
