#!/usr/bin/env python3
"""Quantize Qwen3.5-2B to FP8 W8A8 (compressed-tensors) via llm-compressor.

    source .venv-quantize/bin/activate      # repo-root venv with llmcompressor +
                                             # compressed-tensors + transformers-from-source
                                             # already installed (see
                                             # docs/llm-compressor-quantization-guide.md
                                             # §B for the exact pins this needs)
    python3 scripts/14_quantize_fp8.py
    python3 scripts/14_quantize_fp8.py --model-dir serve/models/qwen3.5-2b \
        --out serve/models/qwen3.5-2b-fp8-dynamic

RECIPE — NOT a guess, copied from a REAL published checkpoint of the SAME
model family: RedHatAI/Qwen3.5-4B-FP8-dynamic (config.json read directly,
see docs/llm-compressor-quantization-guide.md). That checkpoint's
quantization_config.ignore, once you collapse the per-layer expansion back to
patterns, is exactly:

    ignore = ["lm_head", "re:.*embed_tokens$", "re:^mtp.*", "re:.*visual.*", "re:.*linear_attn.*"]

i.e. the ENTIRE Gated DeltaNet mixer (in_proj_qkv/in_proj_z/out_proj/in_proj_a/
in_proj_b — not just the tiny gates) stays BF16, along with the tied lm_head,
the MTP speculative head, and the vision tower. Only the full-attention
self_attn.{q,k,v,o}_proj and every mlp.{gate,up,down}_proj (both layer types)
get FP8'd. Real published accuracy for that recipe on the 4B/9B siblings://
99.5-100.3% recovery vs BF16 across GSM8k-Platinum/MMLU-Pro/Math500/AIME —
see docs/llm-compressor-quantization-guide.md for the numbers and sourcing.

This script does NOT hand-list layer indices (unlike the published
checkpoint's expanded config.json) — a blanket "re:.*linear_attn.*" matches
every GDN layer regardless of how many there are (24 here vs 32 on the 4B),
so it doesn't need to know num_hidden_layers or which indices are GDN vs
full-attention. It DOES loop over config.text_config.layer_types to PRINT a
verification table (real param counts ignored vs quantized) before running —
so a config mismatch (wrong model, wrong arch) fails loud before burning GPU
time, instead of silently quantizing the wrong thing.

Env:
    MODEL_DIR   default serve/models/qwen3.5-2b
    OUT_DIR     default <MODEL_DIR>-fp8-dynamic
"""
import argparse
import json
import os
import re
import shutil
import sys


def h(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# Patterns straight from the RedHatAI/Qwen3.5-*-FP8-dynamic recipe (see module
# docstring). Order doesn't matter — QuantizationModifier ORs them.
IGNORE = [
    "lm_head",           # tied to embed_tokens (nn.Embedding — never matches
                         # targets="Linear" anyway, listed for clarity/safety)
    "re:.*embed_tokens$",  # same tied tensor, listed for parity with the real
                           # RedHatAI recipe.yaml (belt-and-suspenders: also
                           # nn.Embedding, so this never matches targets="Linear"
                           # either — harmless either way)
    "re:^mtp.*",         # speculative-decode draft head, 2.7% of params
    "re:.*visual.*",     # vision tower, dead weight for the text-only trace
    "re:.*linear_attn.*",  # ENTIRE Gated DeltaNet mixer — see module docstring
]

# Files transformers' save_pretrained() calls don't reliably re-emit (they
# belong to the processor/chat layer, untouched by quantization) — copy
# verbatim from the source dir so 01_check_model.sh / vLLM see a complete dir.
EXTRA_FILES = [
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
]

SANITY_PROMPT = "In one sentence, what is prefix caching in LLM serving?"


def matches_ignore(name, patterns):
    """Approximate compressed-tensors' own ignore matching, for the PRE-FLIGHT
    report only (the actual oneshot() call does the real matching itself —
    this is just so the printed table isn't lying to you)."""
    for p in patterns:
        if p.startswith("re:"):
            if re.search(p[3:], name):
                return True
        elif name == p or name.endswith("." + p):
            return True
    return False


def load_model(model_dir):
    import torch
    import transformers

    for loader_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
        try:
            Loader = getattr(transformers, loader_name)
            model = Loader.from_pretrained(model_dir, dtype="auto", device_map="auto")
            print(f">> loaded via {loader_name}")
            return model
        except Exception as e:  # noqa: BLE001
            print(f"   {loader_name} failed: {e}")
    sys.exit("!! could not load model with any Auto* class — check transformers version "
             "in .venv-quantize (needs from-source main branch for qwen3_5)")


def preflight_report(model, cfg):
    """Real param accounting (not hand math) — ignored vs quantized, and a
    sanity check against the hybrid layer_types the config claims."""
    import torch.nn as nn

    text_cfg = getattr(cfg, "text_config", cfg)
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types:
        counts = {}
        for t in layer_types:
            counts[t] = counts.get(t, 0) + 1
        print(f"config layer_types: {counts}  (expect 18 linear_attention : 6 full_attention for the 2B)")

    h("PARAM ACCOUNTING (real, from the loaded model — not hand math)")
    ignored_params, quantized_params, other_params = 0, 0, 0
    n_linear_attn_ignored = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        n = sum(p.numel() for p in mod.parameters())
        if matches_ignore(name, IGNORE):
            ignored_params += n
            if "linear_attn" in name:
                n_linear_attn_ignored += 1
        else:
            quantized_params += n
    total = ignored_params + quantized_params
    print(f"  Linear params quantized (FP8) : {quantized_params/1e9:8.3f} B  ({100*quantized_params/total:5.1f}%)")
    print(f"  Linear params ignored (BF16)  : {ignored_params/1e9:8.3f} B  ({100*ignored_params/total:5.1f}%)")
    print(f"  linear_attn Linear modules ignored: {n_linear_attn_ignored}  "
          f"(expect 18 layers x 3 = 54: in_proj_qkv/in_proj_z/out_proj)")
    if n_linear_attn_ignored == 0:
        sys.exit("!! 0 linear_attn modules matched — module names don't look like expected "
                 "(model.language_model.layers.N.linear_attn.*). Wrong model or wrong loader class. Aborting.")
    return quantized_params, ignored_params


def sanity_generate(model, tokenizer, label):
    import torch

    h(f"SANITY-CHECK GENERATION ({label})")
    msgs = [{"role": "user", "content": SANITY_PROMPT}]
    input_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=64, do_sample=False)
    text = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"  prompt: {SANITY_PROMPT}")
    print(f"  output: {text!r}")
    if len(text.strip()) < 5:
        print("  !! WARNING: output looks empty/degenerate — do NOT trust this checkpoint, inspect before benching.")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "serve/models/qwen3.5-2b"))
    ap.add_argument("--out", default=os.environ.get("OUT_DIR"))
    ap.add_argument("--skip-baseline-generate", action="store_true",
                     help="skip the pre-quantize generation (saves ~10s, less to compare against)")
    args = ap.parse_args()

    model_dir = args.model_dir
    out_dir = args.out or (model_dir.rstrip("/") + "-fp8-dynamic")

    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        sys.exit(f"!! {model_dir}/config.json not found — run scripts/01_check_model.sh first.")

    from transformers import AutoConfig, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    h(f"LOAD  {model_dir}")
    cfg = AutoConfig.from_pretrained(model_dir)
    print(f"architectures={cfg.architectures}  model_type={cfg.model_type}")
    model = load_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    preflight_report(model, cfg)

    if not args.skip_baseline_generate:
        sanity_generate(model, tokenizer, "BEFORE quantize, BF16 baseline")

    h("QUANTIZE  FP8_DYNAMIC (RTN, data-free)  targets=Linear")
    print(f"ignore = {IGNORE}")
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE)
    oneshot(model=model, recipe=recipe)

    try:
        from compressed_tensors.offload import dispatch_model
        dispatch_model(model)
    except Exception as e:  # noqa: BLE001
        print(f"   (dispatch_model skipped: {e})")

    text = sanity_generate(model, tokenizer, "AFTER quantize, FP8")
    if len(text.strip()) < 5:
        ans = input("Degenerate output — save anyway? [y/N] ")
        if ans.strip().lower() != "y":
            sys.exit("!! aborted before save — inspect the model before re-running.")

    h(f"SAVE  ->  {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)
    for f in EXTRA_FILES:
        src, dst = os.path.join(model_dir, f), os.path.join(out_dir, f)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)
            print(f"  copied {f} (not re-emitted by save_pretrained)")

    def du(path):
        return sum(os.path.getsize(os.path.join(dp, fn))
                   for dp, _, fns in os.walk(path) for fn in fns)

    h("DONE")
    print(f"  {model_dir}  ->  {du(model_dir)/1e9:.2f} GB (BF16)")
    print(f"  {out_dir}  ->  {du(out_dir)/1e9:.2f} GB (FP8)")
    # serve_up.sh / 01_check_model.sh read MODEL_DIR relative to serve/ (e.g.
    # "./models/x"), not relative to the repo root this script runs from.
    out_dir_for_env = "./" + os.path.relpath(out_dir, "serve") if out_dir.startswith("serve/") else out_dir

    print()
    print("Next: point MODEL_DIR at the new checkpoint and bench it with the existing pipeline:")
    print(f"  MODEL_DIR={out_dir_for_env} ./scripts/10_bench_e2e.sh")
    print("  # or add a row to scripts/11_multi_bench.sh's EXPERIMENTS to compare vs baseline in one table")


if __name__ == "__main__":
    main()
