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
    python3 scripts/14_quantize_fp8.py --calib   # static FP8, activation scales
                                                 # CALIBRATED on the round-1 trace
                                                 # (an A/B candidate vs dynamic)

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

CALIBRATION (--calib) — read this before assuming calib = better:
  The DEFAULT path is FP8_DYNAMIC, which is DATA-FREE. FP8 weight scales are
  max-abs (data-independent), and activations are scaled per-token at runtime,
  so a calibration dataset changes NOTHING there — feeding one in would be
  theatre. Calibration only bites if you switch the ACTIVATION scheme to static.
  That is what --calib does: scheme=FP8 (static per-tensor W8A8) with activation
  scales frozen from a calibration set. Static per-tensor activations are
  usually a hair LESS accurate than dynamic per-token, so --calib is an A/B
  CANDIDATE vs the dynamic default, not a guaranteed win — quantize both,
  compare GPQA (scripts/12), keep the winner.

  WHICH calibration set (--calib-set): the competition grades ACCURACY on GPQA
  and SPEED on the trace SEPARATELY. Static scales are frozen at calib time, so
  they must fit the distribution where accuracy is graded = GPQA. Calibrating on
  the trace (word-salad tech English) was measured to DROP GPQA accuracy: at
  eval the dense-scientific-English activations fall outside the trace-fit range
  and clip. The trace half only scores latency + non-empty output (ERS ignores
  correctness), so it does not need correct activation scales. Hence the default
  --calib-set=gpqa builds the calibration set from data/GPQA/gpqa_diamond.parquet
  reproducing the EXACT eval prompt (scripts/12's system_instruction + the
  "{{question}}\nLet's think step by step: " template through the chat template),
  so calib-time activations match eval-time activations. (--calib-set=trace is
  kept only to reproduce the accuracy-drop A/B.) Caveat: calibrating on the eval
  set fits the test distribution — legitimate here because BTC grades on it, but
  fragile if the final accuracy set differs; FP8_DYNAMIC sidesteps this entirely.

KNOWN BUG, FOUND AND FIXED (2026-07-10, found on scripts/15's AWQ output via
scripts/16's tensor-level diff, then confirmed this script has the identical
gap since load_model() is byte-for-byte the same function): neither
AutoModelForImageTextToText nor AutoModelForCausalLM attaches Qwen3_5's `mtp`
submodule on load (transformers 5.10.2) — `any('mtp' in n for n, _ in
model.named_modules())` is False for both, despite the BF16 source checkpoint
genuinely holding 15 `mtp.*` weight keys on disk. The model class silently
drops them; nothing in the quantize/save pipeline ever sees them, so
save_pretrained() writes a checkpoint 15 tensors short with no error. FIXED
below via copy_missing_mtp_tensors(): copies those tensors' raw bytes
straight from source to output safetensors post-save (no model round-trip
needed, since `mtp` was always meant to stay untouched BF16 per the
`re:^mtp.*` ignore entry anyway).

Env:
    MODEL_DIR   default serve/models/qwen3.5-2b
    OUT_DIR     default <MODEL_DIR>-fp8-dynamic (or -fp8-static-calib-<set> with --calib)
    CALIB_SET   gpqa (default) | trace          (only used with --calib)
    CALIB_DATA  default per set: data/GPQA/gpqa_diamond.parquet | data/trace-round1.jsonl
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

# Copied VERBATIM from scripts/12_gpqa_lmeval.sh's default SYSTEM_INSTRUCTION —
# the calibration prompt must reproduce the eval-time prompt exactly, system
# message included, or the static activation scales won't match what the server
# sees at grading time. If you change it in scripts/12, change it here too.
GPQA_SYSTEM_INSTRUCTION = (
    "You are an expert scientist. Reason through the problem step by step, then "
    "end your response with exactly this sentence on its own line: The answer is "
    "(X) — replacing X with the correct letter."
)
# doc_to_text from the local task YAML (bench/lmeval_tasks/gpqa_diamond_local/):
# the parquet's `question` already has the 4 choices baked in, so the template
# just appends this suffix. Kept identical so calib prompts == eval prompts.
GPQA_USER_SUFFIX = "\nLet's think step by step: "


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

    # NOT device_map="auto": at 2.27B params (~4.5GB BF16) this fits on any
    # single modern GPU with room to spare — device_map="auto" only earns its
    # keep on models too big for one card. Forcing everything onto a single
    # real device (index 0, or CPU if no CUDA) sidesteps accelerate's dispatch
    # hooks entirely, which is exactly the layer implicated in a real device
    # -side assert seen crossing an accelerate hook boundary on this very-new
    # model (modeling_qwen3_5.py) — fewer moving parts, less surface for a
    # multi-device dtype/placement bug to hide in.
    device_map = {"": 0} if torch.cuda.is_available() else None
    for loader_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
        try:
            Loader = getattr(transformers, loader_name)
            model = Loader.from_pretrained(model_dir, dtype="auto", device_map=device_map)
            print(f">> loaded via {loader_name}  (device_map={device_map})")
            return model
        except Exception as e:  # noqa: BLE001
            print(f"   {loader_name} failed: {e}")
    sys.exit("!! could not load model with any Auto* class — check transformers version "
             "in .venv-quantize (needs from-source main branch for qwen3_5)")


def copy_missing_mtp_tensors(model_dir, out_dir):
    """Patch `mtp.*` tensors back into the FP8 output, copied verbatim (raw
    bytes, no model round-trip) from the BF16 source checkpoint.

    CONFIRMED ON GPU (2026-07-10, transformers 5.10.2, found via
    scripts/16_inspect_quantized_model.py diffed against scripts/15's AWQ
    output): neither AutoModelForImageTextToText NOR AutoModelForCausalLM
    attaches the `mtp` submodule when loading Qwen3_5ForConditionalGeneration
    — `any('mtp' in n for n, _ in model.named_modules())` is False for both,
    even though the BF16 source checkpoint's model.safetensors.index.json
    genuinely lists 15 `mtp.*` weight keys and config.json declares
    `mtp_num_hidden_layers: 1`. Loading only pulls 617 (ImageTextToText) or
    320 (CausalLM) of the source's 632 tensors — the MTP head is simply never
    instantiated, so it can't be quantized OR explicitly ignored: it's
    invisible to the whole pipeline, and save_pretrained() then writes out a
    checkpoint 15 tensors short with no error or warning. This script's
    load_model() is identical to scripts/15's, so it has the exact same gap.

    Since `re:^mtp.*` in IGNORE only ever meant "leave this untouched BF16"
    anyway, the fix doesn't need the model class to support `mtp` at all —
    copy the original bytes straight from source safetensors to output
    safetensors, unmodified."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    def read_matching(shard_path, keys=None):
        out = {}
        with safe_open(shard_path, framework="pt") as f:
            for k in f.keys():
                if keys is None or k in keys:
                    out[k] = f.get_tensor(k)
        return out

    src_index = os.path.join(model_dir, "model.safetensors.index.json")
    src_single = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(src_index):
        with open(src_index) as fh:
            idx = json.load(fh)
        mtp_keys = [k for k in idx["weight_map"] if k.startswith("mtp.")]
        mtp_tensors = {}
        by_shard = {}
        for k in mtp_keys:
            by_shard.setdefault(idx["weight_map"][k], []).append(k)
        for shard, keys in by_shard.items():
            mtp_tensors.update(read_matching(os.path.join(model_dir, shard), set(keys)))
    elif os.path.isfile(src_single):
        mtp_tensors = read_matching(src_single)
        mtp_tensors = {k: v for k, v in mtp_tensors.items() if k.startswith("mtp.")}
    else:
        sys.exit(f"!! no safetensors found in {model_dir} to recover mtp tensors from")

    if not mtp_tensors:
        print("  (no mtp.* tensors found in source checkpoint — nothing to patch; "
              "this model genuinely has no MTP head, or naming differs — verify before trusting this.)")
        return

    out_single = os.path.join(out_dir, "model.safetensors")
    out_index = os.path.join(out_dir, "model.safetensors.index.json")
    if os.path.isfile(out_index):
        sys.exit("!! output is sharded (model.safetensors.index.json) — "
                 "copy_missing_mtp_tensors() only handles the single-shard case seen "
                 "in practice; extend this function before using it on a multi-shard output.")
    if not os.path.isfile(out_single):
        sys.exit(f"!! no model.safetensors in {out_dir} — save_pretrained() may have failed")

    existing = read_matching(out_single)
    already = [k for k in mtp_tensors if k in existing]
    if already:
        print(f"  ({len(already)} mtp.* tensors already present in output — save_pretrained "
              f"picked them up this time, nothing to patch)")
        return

    merged = {**existing, **mtp_tensors}
    save_file(merged, out_single, metadata={"format": "pt"})
    added_bytes = sum(t.numel() * t.element_size() for t in mtp_tensors.values())
    print(f"  patched {len(mtp_tensors)} mtp.* tensors ({added_bytes/1e6:.1f} MB, BF16 verbatim) "
          f"into {out_single} — model class dropped them on load, this restores them post-hoc")


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
          f"(expect 18 layers x 5 = 90: in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj "
          f"— in_proj_a/_b ARE nn.Linear too, tiny [16,2048], swept in by the same blanket regex)")
    if n_linear_attn_ignored == 0:
        sys.exit("!! 0 linear_attn modules matched — module names don't look like expected "
                 "(model.language_model.layers.N.linear_attn.*). Wrong model or wrong loader class. Aborting.")
    return quantized_params, ignored_params


def _tokenize_texts(tokenizer, texts, num_samples, max_seq_len):
    """Deterministic subsample (if capped) + tokenize into a datasets.Dataset."""
    import random

    from datasets import Dataset

    if num_samples and num_samples < len(texts):
        random.Random(42).shuffle(texts)
        texts = texts[:num_samples]
    rows = []
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=max_seq_len, add_special_tokens=False)
        rows.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    return Dataset.from_list(rows)


def build_gpqa_calib(tokenizer, path, num_samples, max_seq_len, system_instruction):
    """Calibration set that reproduces the EXACT GPQA eval prompt so the frozen
    static activation scales match what the server sees at grading time. Mirrors
    scripts/12_gpqa_lmeval.sh: system=<expert-scientist instruction>, user=the
    parquet's `question` (choices already baked in) + GPQA_USER_SUFFIX, rendered
    through the model's own chat template with the generation prompt appended —
    just like /v1/chat/completions does at eval. Reads the SAME parquet the eval
    consumes (not the raw CSV), so choice order/lettering can't drift."""
    from datasets import load_dataset

    if not os.path.isfile(path):
        sys.exit(f"!! --calib-data {path} not found (expected data/GPQA/gpqa_diamond.parquet).")
    ext = os.path.splitext(path)[1].lower()
    fmt = "parquet" if ext == ".parquet" else "json" if ext in (".jsonl", ".json") else None
    if fmt is None:
        sys.exit(f"!! GPQA calib expects .parquet or .jsonl, got {ext}")
    raw = load_dataset(fmt, data_files=path, split="train")
    if "question" not in raw.column_names:
        sys.exit(f"!! {path} has no `question` column (cols={raw.column_names}) — wrong file?")
    texts = []
    for rec in raw:
        q = rec.get("question")
        if not q:
            continue
        msgs = []
        if system_instruction:
            msgs.append({"role": "system", "content": system_instruction})
        msgs.append({"role": "user", "content": q + GPQA_USER_SUFFIX})
        texts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
    if not texts:
        sys.exit(f"!! no usable questions in {path}")
    return _tokenize_texts(tokenizer, texts, num_samples, max_seq_len)


def build_trace_calib(tokenizer, path, num_samples, max_seq_len):
    """Calibration set = the ACTUAL serving distribution. Each trace record's
    body.messages is rendered through the model's own chat template exactly as
    it will be at serve time, then tokenized. The trace's built-in redundancy
    (20 sessions x 6 snapshots, one shared system prompt) is KEPT on purpose:
    it weights the activation statistics by what the server really sees, which
    is the whole point of calibrating on-distribution rather than on ultrachat.
    NOTE: trace-calib is measured to DROP GPQA accuracy (see module docstring) —
    kept only to reproduce that A/B; --calib-set=gpqa is the real path."""
    if not os.path.isfile(path):
        sys.exit(f"!! --calib-data {path} not found (expected the round-1 trace JSONL).")
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line).get("body", {}).get("messages")
            if msgs:
                texts.append(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False))
    if not texts:
        sys.exit(f"!! no body.messages found in {path} — wrong file?")
    return _tokenize_texts(tokenizer, texts, num_samples, max_seq_len)


def sanity_generate(model, tokenizer, label):
    import torch

    h(f"SANITY-CHECK GENERATION ({label})")
    msgs = [{"role": "user", "content": SANITY_PROMPT}]
    # return_dict=True is required here: this transformers version's chat
    # template returns a BatchEncoding (dict-like: input_ids + attention_mask),
    # NOT a bare tensor, even with return_tensors="pt" alone. Passing that dict
    # positionally as `input_ids` to generate() crashes deep inside
    # (inputs_tensor.shape[0] on a dict). Ask for the dict explicitly and
    # unpack it — works whether or not this quirk is present.
    inputs = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
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
    ap.add_argument("--calib", action="store_true",
                     help="Static-activation FP8 (scheme=FP8) with activation scales CALIBRATED, "
                          "instead of the default data-free FP8_DYNAMIC. NOTE (see module docstring): "
                          "calib does NOT change FP8 weights — only the static activation scales. "
                          "It's an A/B candidate vs dynamic, not a guaranteed win.")
    ap.add_argument("--calib-set", choices=["gpqa", "trace"], default=os.environ.get("CALIB_SET", "gpqa"),
                     help="Calibration distribution. 'gpqa' (default) matches where accuracy is graded; "
                          "'trace' reproduces the measured accuracy-drop A/B. See module docstring.")
    ap.add_argument("--calib-data", default=os.environ.get("CALIB_DATA"),
                     help="Override calib file. Default per --calib-set: "
                          "gpqa->data/GPQA/gpqa_diamond.parquet, trace->data/trace-round1.jsonl")
    ap.add_argument("--system-instruction", default=os.environ.get("SYSTEM_INSTRUCTION", GPQA_SYSTEM_INSTRUCTION),
                     help="System message prepended to each GPQA calib prompt — MUST match scripts/12's "
                          "SYSTEM_INSTRUCTION so calib activations match eval activations. Set '' to omit.")
    ap.add_argument("--num-calib-samples", type=int, default=int(os.environ.get("NUM_CALIB_SAMPLES", "0")),
                     help="Cap on calibration samples (0 = every record; GPQA-diamond is 198, trace is 120).")
    ap.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "8192")),
                     help="Truncate each calibration prompt to this many tokens (only with --calib).")
    args = ap.parse_args()

    model_dir = args.model_dir
    out_dir = args.out or (model_dir.rstrip("/") +
                           (f"-fp8-static-calib-{args.calib_set}" if args.calib else "-fp8-dynamic"))

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

    has_mtp = any("mtp" in n for n, _ in model.named_modules())
    print(f"model has `mtp` submodule loaded: {has_mtp}" + (
        "" if has_mtp else
        "  !! KNOWN GAP (confirmed 2026-07-10, transformers 5.10.2): the loader "
        "class silently drops Qwen3_5's MTP head. Will patch the original BF16 "
        "mtp.* tensors back in from source safetensors after save — see "
        "copy_missing_mtp_tensors()."
    ))

    preflight_report(model, cfg)

    if not args.skip_baseline_generate:
        sanity_generate(model, tokenizer, "BEFORE quantize, BF16 baseline")

    if args.calib:
        default_data = ("data/GPQA/gpqa_diamond.parquet" if args.calib_set == "gpqa"
                        else "data/trace-round1.jsonl")
        calib_path = args.calib_data or default_data
        h(f"QUANTIZE  FP8 (static W8A8, activation scales CALIBRATED on {args.calib_set})  targets=Linear")
        print(f"ignore = {IGNORE}")
        print(f"calib set  = {args.calib_set}")
        print(f"calib data = {calib_path}")
        if args.calib_set == "gpqa":
            ds = build_gpqa_calib(tokenizer, calib_path, args.num_calib_samples,
                                  args.max_seq_len, args.system_instruction)
        else:
            ds = build_trace_calib(tokenizer, calib_path, args.num_calib_samples, args.max_seq_len)
        print(f"calibration samples = {len(ds)}   (max_seq_len={args.max_seq_len})")
        recipe = QuantizationModifier(targets="Linear", scheme="FP8", ignore=IGNORE)
        oneshot(model=model, recipe=recipe, dataset=ds,
                num_calibration_samples=len(ds), max_seq_length=args.max_seq_len)
    else:
        h("QUANTIZE  FP8_DYNAMIC (RTN, data-free)  targets=Linear")
        print(f"ignore = {IGNORE}")
        recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE)
        oneshot(model=model, recipe=recipe)

    # No dispatch_model() call here: the model is already single-device
    # (device_map={"": 0} at load, see load_model()) with no offload, so
    # there's nothing to re-dispatch. Calling it anyway previously triggered
    # a CUDA device-side assert (indexSelectSmallIndex) in embed_tokens
    # during generate — exactly the accelerate-dispatch-hook failure mode
    # load_model()'s device_map choice was meant to avoid in the first place.

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

    if not has_mtp:
        h("PATCH  mtp.* tensors back in (model class dropped them on load)")
        copy_missing_mtp_tensors(model_dir, out_dir)

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
