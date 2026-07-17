#!/usr/bin/env python3
"""Quantize Qwen3.5-2B to INT4 AWQ (compressed-tensors) via llm-compressor.

    source .venv-quantize/bin/activate
    python3 scripts/15_quantize_awq.py
    python3 scripts/15_quantize_awq.py --model-dir serve/models/qwen3.5-2b \
        --out serve/models/qwen3.5-2b-awq-int4

RECIPE — NOT a guess, copied field-for-field from a REAL published INT4-AWQ
checkpoint of the SAME model: cyankiwi/Qwen3.5-2B-AWQ-4bit (recipe.yaml +
config.json quantization_config read directly from the HF repo). That
recipe's AWQModifier block is:

    weights: num_bits=4, type=int, symmetric=true, group_size=32,
             strategy=group, observer=mse, dynamic=false
    targets: [Linear]
    ignore:  ['re:.*embed_tokens', 're:.*linear_attn.*', 're:model[.]visual.*',
              're:mtp.*', 'lm_head']
    mappings: 3 custom AWQMapping entries (see build_mappings() below) —
              NOT llm-compressor's built-in `default_mappings`, because this
              hybrid GDN/full-attention arch (Qwen3_5) has no entry in
              AWQ_MAPPING_REGISTRY (verified against the real
              llm-compressor main-branch source,
              src/llmcompressor/modifiers/transform/awq/mappings.py — no
              "Qwen3_5ForConditionalGeneration" / "Qwen3NextForCausalLM" key).
              cyankiwi hand-wrote mappings restricted to the 6 full-attention
              layers only (indices 3,7,11,15,19,23 of 24 — every 4th layer,
              matching text_config.full_attention_interval=4); the 18 GDN
              (linear_attention) layers have no self_attn.{q,k,v,o}_proj to
              smooth at all, so they're absent from the mapping (and already
              covered by the blanket `re:.*linear_attn.*` ignore).
    duo_scaling: true, n_grid: 20 — BOTH are llm-compressor's own class
              defaults (AWQModifier.duo_scaling=True, .n_grid=20 — verified
              against transform/awq/base.py). cyankiwi did not tune these;
              they're listed here only because oneshot() always serializes
              the full recipe, defaults included.
    offload_device: cpu — NOT the class default (default is None for a
              non-MoE model; llm-compressor only auto-picks cpu for detected
              MoE models). This IS a deliberate cyankiwi choice, kept as-is.

Compare with scripts/14_quantize_fp8.py's FP8_DYNAMIC recipe: that one is
data-free RTN. AWQ is fundamentally different — it is NOT data-free. The
scale-search (grid search over `n_grid` ratios per mapping, picking the scale
that minimizes reconstruction error) requires real forward passes with real
activations flowing through the smooth_layer -> balance_layers pairs. There
is no calibration-free AWQ path; skipping --calib is not an option here
(unlike FP8_DYNAMIC in script 14, which has no calibration flag at all).

TWO THINGS THAT CANNOT BE REPRODUCED EXACTLY (verified, not assumed):

1. Calibration dataset. Neither recipe.yaml nor config.json records which
   dataset cyankiwi used — oneshot()'s `dataset=` argument is caller-side,
   never part of the saved recipe. So exact bit-for-bit scale reproduction
   is not possible without guessing their calibration set. This script
   defaults calibration to the SAME GPQA-matching prompts scripts/12
   evaluates on and scripts/14 --calib already established the rationale
   for: frozen scales (here: weights themselves, not just activation
   scales) must fit the distribution graded for accuracy. See scripts/14's
   module docstring for the fuller argument; the short version is unchanged
   for AWQ.

2. `balance_exponent: 1`, present on EVERY mapping entry in the real
   recipe.yaml. The current, real `AWQMapping` dataclass (fetched directly
   from llm-compressor's live GitHub source,
   src/llmcompressor/modifiers/transform/awq/mappings.py) has only 3
   fields: smooth_layer, balance_layers, activation_hook_target — no
   `balance_exponent`. A GitHub code/commit search for that name in the
   real repo turned up nothing either. So this is not "an older default we
   can still set" — no pip-installable llm-compressor build we found
   accepts this kwarg at all. build_mappings() below omits it; there is no
   substitute.

NOT a gap, despite looking like one at first — the calibration PIPELINE:
oneshot()'s own real default is `pipeline="independent"` (verified against
src/llmcompressor/entrypoints/oneshot.py's signature). Like the dataset,
recipe.yaml doesn't record which pipeline cyankiwi used either — but unlike
the dataset, this one doesn't need guessing: "independent" re-infers a
pipeline PER modifier (pipelines/independent/pipeline.py), giving the
AWQModifier half real calibration data ("sequential", since it needs actual
activations for its smoothing grid search) and the paired
QuantizationModifier half a data-free pass ("datafree", since weight-only
int4 with an mse observer is computed straight from the weight tensor, no
forward pass needed). That's the objectively correct split for this exact
modifier pair, not merely "whatever happens by default" — so relying on it
(passed explicitly below, not left implicit) carries no reproduction risk.

KNOWN BUG, FOUND AND FIXED (2026-07-10, via scripts/16's tensor-level diff
against the real cyankiwi checkpoint): a real GPU run of this script produced
model.safetensors at 2.79GB vs the reference's 2.50GB. Root cause was NOT the
recipe — every quantized category (mlp/self_attn/embed_tokens/visual) matched
the reference byte-for-byte. Two things fully reconciled the +496MB delta:

1. `linear_attn` (+542.6MB): this script's output is 100% BF16 for the whole
   Gated DeltaNet mixer, matching the recipe's literal `re:.*linear_attn.*`
   ignore text. The REFERENCE checkpoint's own config.json ignore list makes
   the same claim but its actual saved tensors contradict it — `in_proj_qkv`,
   `in_proj_z`, and `out_proj` (3 of 5 linear_attn Linears per GDN layer) are
   genuinely INT4-packed in the real file, only `in_proj_a`/`in_proj_b` (the
   tiny [16,2048] gates) stayed BF16. This script is arguably MORE faithful
   to the recipe's stated intent than the reference checkpoint is.

2. `mtp` (-44.1MB, opposite direction, smaller magnitude): the reference has
   32 `mtp.*` tensors (partially INT4-quantized); a real run of THIS script
   had ZERO — confirmed on GPU that neither AutoModelForImageTextToText NOR
   AutoModelForCausalLM attaches Qwen3_5's `mtp` submodule when loading
   (transformers 5.10.2) — `any('mtp' in n for n, _ in model.named_modules())`
   is False for both, despite the BF16 source checkpoint genuinely holding 15
   `mtp.*` weight keys on disk. The model class silently drops them; nothing
   in the quantize/save pipeline ever sees them, so save_pretrained() writes
   a checkpoint 15 tensors short with no error. FIXED below via
   copy_missing_mtp_tensors(): copies those tensors' raw bytes straight from
   source to output safetensors post-save (no model round-trip needed, since
   `mtp` was always meant to stay untouched BF16 anyway).

Env:
    MODEL_DIR   default serve/models/qwen3.5-2b
    OUT_DIR     default <MODEL_DIR>-awq-int4
    CALIB_DATA  default data/GPQA/gpqa_diamond.parquet
"""
import argparse
import os
import re
import shutil
import sys


def h(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# Verbatim from cyankiwi/Qwen3.5-2B-AWQ-4bit's recipe.yaml `ignore:` list.
IGNORE = [
    "re:.*embed_tokens",
    "re:.*linear_attn.*",   # entire Gated DeltaNet mixer, all 18 GDN layers
    "re:model[.]visual.*",  # vision tower
    "re:mtp.*",              # speculative-decode draft head
    "lm_head",
]

# Verbatim from the recipe.yaml AWQModifier block (quantization side).
WEIGHTS_ARGS = dict(
    num_bits=4,
    type="int",
    symmetric=True,
    group_size=32,
    strategy="group",
    observer="mse",
    dynamic=False,
)

EXTRA_FILES = [
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
]

SANITY_PROMPT = "In one sentence, what is prefix caching in LLM serving?"

GPQA_SYSTEM_INSTRUCTION = (
    "You are an expert scientist. Reason through the problem step by step, then "
    "end your response with exactly this sentence on its own line: The answer is "
    "(X) — replacing X with the correct letter."
)
GPQA_USER_SUFFIX = "\nLet's think step by step: "


def matches_ignore(name, patterns):
    """Approximate compressed-tensors' own ignore matching, for the preflight
    report only — oneshot() does the real matching itself."""
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

    # Single real device, not device_map="auto" — see scripts/14's load_model()
    # for why (accelerate dispatch hooks were implicated in a real CUDA
    # device-side assert on this exact model class).
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


def full_attention_layer_indices(cfg):
    """Derive which decoder layer indices are full-attention (vs GDN/linear
    attention) FROM THE LOADED CONFIG, not a hand-typed list — layer_types is
    the ground truth (see scripts/14_quantize_fp8.py's preflight_report() for
    the same philosophy: fail loud on a mismatch rather than trust a constant
    that could silently go stale on a different checkpoint)."""
    text_cfg = getattr(cfg, "text_config", cfg)
    layer_types = getattr(text_cfg, "layer_types", None)
    if not layer_types:
        sys.exit("!! config has no text_config.layer_types — wrong model, can't build AWQ mappings.")
    idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    if not idx:
        sys.exit("!! 0 full_attention layers found in layer_types — can't build AWQ mappings.")
    return idx


def build_mappings(cfg):
    """3 AWQMapping entries, restricted to the full-attention layers only —
    mirrors cyankiwi's recipe.yaml exactly (see module docstring). The 4th
    mapping in llm-compressor's own `default_mappings`
    (up_proj -> down_proj smoothing) is deliberately ABSENT here too, matching
    the real recipe.yaml: down_proj is still quantized (targets=[Linear],
    not ignored) but gets no AWQ smoothing correction."""
    from llmcompressor.modifiers.awq import AWQMapping

    idx = full_attention_layer_indices(cfg)
    layer_alt = "(" + "|".join(str(i) for i in idx) + ")"
    print(f"AWQ mapping layer indices (full_attention): {idx}")

    return [
        AWQMapping(
            f"re:model.*layers[.]{layer_alt}[.]input_layernorm",
            [
                f"re:model.*layers[.]{layer_alt}[.]self_attn[.]q_proj",
                f"re:model.*layers[.]{layer_alt}[.]self_attn[.]k_proj",
                f"re:model.*layers[.]{layer_alt}[.]self_attn[.]v_proj",
            ],
        ),
        AWQMapping(
            f"re:model.*layers[.]{layer_alt}[.]self_attn[.]v_proj",
            [f"re:model.*layers[.]{layer_alt}[.]self_attn[.]o_proj"],
        ),
        AWQMapping(
            "re:model.*post_attention_layernorm",
            ["re:model.*mlp[.]gate_proj", "re:model.*mlp[.]up_proj"],
        ),
    ]


def copy_missing_mtp_tensors(model_dir, out_dir):
    """Patch `mtp.*` tensors back into the AWQ output, copied verbatim (raw
    bytes, no model round-trip) from the BF16 source checkpoint.

    CONFIRMED ON GPU (2026-07-10, transformers 5.10.2): neither
    AutoModelForImageTextToText NOR AutoModelForCausalLM attaches the `mtp`
    submodule when loading Qwen3_5ForConditionalGeneration —
    `any('mtp' in n for n, _ in model.named_modules())` is False for both,
    even though the BF16 source checkpoint's model.safetensors.index.json
    genuinely lists 15 `mtp.*` weight keys and config.json declares
    `mtp_num_hidden_layers: 1`. Loading only pulls 617 (ImageTextToText) or
    320 (CausalLM) of the source's 632 tensors — the MTP head is simply never
    instantiated, so it can't be quantized OR explicitly ignored: it's
    invisible to the whole pipeline, and save_pretrained() then writes out a
    checkpoint 15 tensors short with no error or warning.

    Since `re:mtp.*` in IGNORE only ever meant "leave this untouched BF16"
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
                 "in practice (AWQ INT4 output for this 2B model is ~3GB, one shard); "
                 "extend this function before using it on a multi-shard AWQ output.")
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
    import torch.nn as nn

    text_cfg = getattr(cfg, "text_config", cfg)
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types:
        counts = {}
        for t in layer_types:
            counts[t] = counts.get(t, 0) + 1
        print(f"config layer_types: {counts}  (expect 18 linear_attention : 6 full_attention for the 2B)")

    h("PARAM ACCOUNTING (real, from the loaded model — not hand math)")
    ignored_params, quantized_params = 0, 0
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
    print(f"  Linear params quantized (INT4)  : {quantized_params/1e9:8.3f} B  ({100*quantized_params/total:5.1f}%)")
    print(f"  Linear params ignored (BF16)     : {ignored_params/1e9:8.3f} B  ({100*ignored_params/total:5.1f}%)")
    print(f"  linear_attn Linear modules ignored: {n_linear_attn_ignored}")
    if n_linear_attn_ignored == 0:
        sys.exit("!! 0 linear_attn modules matched — module names don't look like expected. Aborting.")


def _tokenize_texts(tokenizer, texts, num_samples, max_seq_len):
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
    """Calibration set = the exact GPQA eval prompt (mirrors scripts/12 and
    scripts/14's --calib path) so the AWQ scale search sees activations from
    the same distribution accuracy is graded on. See module docstring for why
    this matters MORE for AWQ than for FP8-static: here it corrects the
    WEIGHTS themselves, not just a side activation scale."""
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


def sanity_generate(model, tokenizer, label):
    import torch

    h(f"SANITY-CHECK GENERATION ({label})")
    msgs = [{"role": "user", "content": SANITY_PROMPT}]
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
    ap.add_argument("--calib-data", default=os.environ.get("CALIB_DATA", "data/GPQA/gpqa_diamond.parquet"),
                     help="Calibration file (question column, parquet or jsonl). Default matches the "
                          "GPQA-diamond eval prompt so AWQ scales fit the accuracy-graded distribution.")
    ap.add_argument("--system-instruction", default=os.environ.get("SYSTEM_INSTRUCTION", GPQA_SYSTEM_INSTRUCTION),
                     help="Must match scripts/12's SYSTEM_INSTRUCTION so calib activations match eval "
                          "activations. Set '' to omit.")
    ap.add_argument("--num-calib-samples", type=int, default=int(os.environ.get("NUM_CALIB_SAMPLES", "0")),
                     help="Cap on calibration samples (0 = every record; GPQA-diamond is 198).")
    ap.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "8192")),
                     help="Truncate each calibration prompt to this many tokens.")
    args = ap.parse_args()

    model_dir = args.model_dir
    out_dir = args.out or (model_dir.rstrip("/") + "-awq-int4")

    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        sys.exit(f"!! {model_dir}/config.json not found — run scripts/01_check_model.sh first.")

    import torch
    from transformers import AutoConfig, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

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

    h("BUILD CALIBRATION SET (GPQA-matching prompts)")
    print(f"calib data = {args.calib_data}")
    ds = build_gpqa_calib(tokenizer, args.calib_data, args.num_calib_samples,
                          args.max_seq_len, args.system_instruction)
    print(f"calibration samples = {len(ds)}   (max_seq_len={args.max_seq_len})")

    h("QUANTIZE  AWQ INT4 (group_size=32, symmetric, observer=mse)  targets=Linear")
    print(f"ignore = {IGNORE}")
    print(f"weights = {WEIGHTS_ARGS}")
    mappings = build_mappings(cfg)
    # Single AWQModifier() call, kwargs split internally into an
    # AWQTransformModifier (mappings/offload_device/duo_scaling/n_grid) +
    # QuantizationModifier (config_groups/targets/ignore) — matches the
    # single top-level "AWQModifier:" block in cyankiwi's recipe.yaml (see
    # module docstring: this is llm-compressor's own back-compat shim over
    # the newer split API, not a home-grown shortcut).
    recipe = AWQModifier(
        config_groups={
            "group_1": {
                "targets": ["Linear"],
                "weights": WEIGHTS_ARGS,
            }
        },
        targets=["Linear"],
        ignore=IGNORE,
        mappings=mappings,
        offload_device=torch.device("cpu"),  # torch.device instance required, not a
                                              # bare str — matches the field's
                                              # `torch.device | None | Sentinel`
                                              # annotation (no str coercion in
                                              # llm-compressor's own validator).
        duo_scaling=True,
        n_grid=20,
    )
    # pipeline="independent" is llm-compressor's OWN default for oneshot()
    # (verified against the real signature,
    # src/llmcompressor/entrypoints/oneshot.py: `pipeline: str | None =
    # "independent"`), spelled out here instead of left implicit. It is also
    # the CORRECT choice for this exact modifier pair, not just "whatever
    # happens by default": CalibrationPipeline (pipelines/independent/
    # pipeline.py) re-infers a pipeline PER MODIFIER — AWQModifier (the
    # transform half) requires real calibration data for its smoothing-scale
    # grid search, so it gets "sequential"; the paired QuantizationModifier
    # here is weight-only (no input/output_activations), so its int4 group
    # scale/zero-point (mse observer) is computed straight from the weight
    # tensor with no forward pass needed, so it gets "datafree". Passing
    # pipeline="sequential" directly would force BOTH through one data
    # pipeline, wasting a pass and NOT matching how llm-compressor's own
    # default would run this recipe.
    oneshot(model=model, recipe=recipe, dataset=ds,
            num_calibration_samples=len(ds), max_seq_length=args.max_seq_len,
            pipeline="independent")

    text = sanity_generate(model, tokenizer, "AFTER quantize, AWQ INT4")
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
    print(f"  {out_dir}  ->  {du(out_dir)/1e9:.2f} GB (AWQ INT4)")
    out_dir_for_env = "./" + os.path.relpath(out_dir, "serve") if out_dir.startswith("serve/") else out_dir

    print()
    print("Next: point MODEL_DIR at the new checkpoint and bench it with the existing pipeline:")
    print(f"  MODEL_DIR={out_dir_for_env} ./scripts/10_bench_e2e.sh")
    print("  # or add a row to scripts/11_multi_bench.sh's EXPERIMENTS to compare vs FP8/baseline")


if __name__ == "__main__":
    main()
