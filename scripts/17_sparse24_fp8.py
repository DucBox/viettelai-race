#!/usr/bin/env python3
"""Compress Qwen3.5-2B to **2:4 sparse + W8A8-FP8 static** (compressed-tensors)
via llm-compressor, in ONE oneshot() recipe:

    recipe = [ SparseGPTModifier(sparsity=0.5, mask_structure="2:4", ...),   # prune
               QuantizationModifier(targets="Linear", scheme="FP8", ...) ]   # then FP8

    python3 scripts/17_sparse24_fp8.py \
        --model-dir serve/models/qwen3.5-2b \
        --out serve/models/qwen3.5-2b-sparse24-fp8

WHY THIS IS A SEPARATE SCRIPT FROM 14 (not a flag on it):
  * It needs llm-compressor **0.10.0.2** — the LAST release that still writes the
    2:4 `sparse-24-bitmask` compressed format (0.11.0 removed sparsity). Script
    14's image ships 0.12.0, which cannot produce this.
  * The output ONLY loads/accelerates on vLLM **<= 0.18.1**. From v0.19.0 the
    CompressedTensors24 scheme is a stub that raises
    "Sparse24 models are no longer supported by vLLM"; v0.22.0 deletes it. So the
    serving image for this checkpoint is vllm/vllm-openai:v0.18.1, NOT 0.24.0.
    (Verified: Qwen3_5ForConditionalGeneration + MTP + --gdn-prefill-backend all
    exist at 0.18.1, and its compressed_tensors_24.py supports FP8 STATIC —
    input_quant.dynamic=False -> GroupShape.PER_TENSOR, strategy=TENSOR.)

  ⚠️ STRATEGIC COST (measured, read before trusting this wins): serving on 0.18.1
     forfeits vLLM 0.24's kernel/flashinfer TTFT edge (0.24 was ~5x faster TTFT
     than 0.22 in our A/B; 0.18.1 is older still). 2:4 only speeds the DECODE
     GEMMs (TPOT). Net score is an OPEN QUESTION this run measures, not a
     guaranteed gain. Bench 0.18.1-stock as the baseline BEFORE crediting 2:4.

CALIBRATION IS MANDATORY HERE (unlike script 14's data-free default):
  SparseGPT is a DATA-DEPENDENT pruner (it needs a Hessian from real activations
  to choose which weights to zero), and --calib static FP8 also needs activation
  scales. Both consume the SAME calibration set. We reuse script 14's exact GPQA
  calib builder so the frozen static scales match where accuracy is graded.
  (Default --fp8-scheme=FP8 = static per-tensor. --fp8-scheme=FP8_DYNAMIC is
  offered for an A/B, but note sparse+dynamic still needs calib for the pruning.)

SPARSE SET == FP8 SET (deliberate): both modifiers use the SAME `ignore` list.
  vLLM disabled the "w16a16 2:4" path (PR #12417) — a layer that is 2:4-sparse
  but NOT fp8-quantized has no fast kernel. Keeping the two sets identical means
  every pruned layer is also FP8 (fast sparse path), and every ignored layer
  stays plain dense BF16 (normal path). Never a sparse-but-unquantized layer.

Reuses script 14's helpers verbatim (load_model, GPQA/trace calib builders,
copy_missing_mtp_tensors, preflight_report, sanity_generate, IGNORE_PRESETS) so
the two stay in lock-step — see scripts/14_quantize_fp8.py for their rationale.
"""
import argparse
import importlib
import os
import shutil
import sys

# scripts/ isn't a package and the module name starts with a digit, so import it
# by path via importlib rather than a normal `import`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
q14 = importlib.import_module("14_quantize_fp8")

h = q14.h
IGNORE_PRESETS = q14.IGNORE_PRESETS
EXTRA_FILES = q14.EXTRA_FILES


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "serve/models/qwen3.5-2b"))
    ap.add_argument("--out", default=os.environ.get("OUT_DIR"))
    ap.add_argument("--skip-baseline-generate", action="store_true",
                     help="skip the pre-compress BF16 generation (saves ~10s)")
    ap.add_argument("--sparsity", type=float, default=float(os.environ.get("SPARSITY", "0.5")),
                     help="target sparsity for SparseGPT (0.5 = the only ratio 2:4 supports).")
    ap.add_argument("--mask-structure", default=os.environ.get("MASK_STRUCTURE", "2:4"),
                     help="SparseGPT mask structure. Keep 2:4 — that is the structure vLLM's "
                          "sparse cutlass kernels accelerate.")
    ap.add_argument("--fp8-scheme", choices=["FP8", "FP8_DYNAMIC"],
                     default=os.environ.get("FP8_SCHEME", "FP8"),
                     help="Activation scheme for the FP8 step. FP8 (default) = static per-tensor "
                          "(scales frozen from calib; the perf-meaningful, sparse24-supported path). "
                          "FP8_DYNAMIC = per-token (A/B; still needs calib for the SparseGPT pruning).")
    ap.add_argument("--calib-set", choices=["gpqa", "trace"], default=os.environ.get("CALIB_SET", "gpqa"),
                     help="Calibration distribution (feeds BOTH SparseGPT + static FP8). 'gpqa' "
                          "matches where accuracy is graded; 'trace' reproduces the accuracy-drop A/B.")
    ap.add_argument("--calib-data", default=os.environ.get("CALIB_DATA"),
                     help="Override calib file. Default per --calib-set: "
                          "gpqa->data/GPQA/gpqa_diamond.parquet, trace->data/trace-round1.jsonl")
    ap.add_argument("--system-instruction",
                     default=os.environ.get("SYSTEM_INSTRUCTION", q14.GPQA_SYSTEM_INSTRUCTION),
                     help="System message prepended to each GPQA calib prompt (must match scripts/12).")
    ap.add_argument("--num-calib-samples", type=int, default=int(os.environ.get("NUM_CALIB_SAMPLES", "0")),
                     help="Cap on calibration samples (0 = all; GPQA-diamond is 198). Lower = faster "
                          "startup, weaker Hessian/scales.")
    ap.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "8192")),
                     help="Truncate each calibration prompt to this many tokens.")
    ap.add_argument("--ignore-preset", choices=list(IGNORE_PRESETS),
                     default=os.environ.get("IGNORE_PRESET", "default"),
                     help="IGNORE list shared by BOTH SparseGPT and FP8 (see scripts/14 IGNORE_PRESETS). "
                          "'default' keeps lm_head/GDN gates/visual/mtp dense BF16.")
    ap.add_argument("--dampening-frac", type=float,
                     default=float(os.environ.get("DAMPENING_FRAC", "0.01")),
                     help="SparseGPT Hessian dampening (numerical stability of the pruning solve).")
    args = ap.parse_args()

    ignore = IGNORE_PRESETS[args.ignore_preset]
    # Keep script 14's module-level IGNORE in sync so its preflight_report /
    # copy path see the same list we compress with.
    q14.IGNORE = ignore
    print(f">> ignore-preset = {args.ignore_preset}  ({len(ignore)} entries) — shared by SparseGPT + FP8")
    print(f">> sparsity={args.sparsity}  mask={args.mask_structure}  fp8-scheme={args.fp8_scheme}")
    if args.mask_structure != "2:4":
        print("!! mask_structure != 2:4 — vLLM's sparse cutlass kernels only accelerate 2:4; "
              "any other structure serves dense. Continue only if you know why.")

    model_dir = args.model_dir
    out_dir = args.out or (model_dir.rstrip("/") +
                           ("-sparse24-fp8-static" if args.fp8_scheme == "FP8"
                            else "-sparse24-fp8-dynamic"))

    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        sys.exit(f"!! {model_dir}/config.json not found — is the BF16 model at {model_dir}?")

    from transformers import AutoConfig, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.obcq import SparseGPTModifier

    h(f"LOAD  {model_dir}")
    cfg = AutoConfig.from_pretrained(model_dir)
    print(f"architectures={cfg.architectures}  model_type={cfg.model_type}")
    model = q14.load_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    has_mtp = any("mtp" in n for n, _ in model.named_modules())
    print(f"model has `mtp` submodule loaded: {has_mtp}" + (
        "" if has_mtp else
        "  !! KNOWN GAP (see scripts/14 copy_missing_mtp_tensors): the loader drops "
        "Qwen3_5's MTP head; will patch the BF16 mtp.* tensors back in post-save."))

    q14.preflight_report(model, cfg)

    if not args.skip_baseline_generate:
        q14.sanity_generate(model, tokenizer, "BEFORE compress, BF16 baseline")

    # ---- calibration set (mandatory: SparseGPT is data-dependent) ----
    default_data = ("data/GPQA/gpqa_diamond.parquet" if args.calib_set == "gpqa"
                    else "data/trace-round1.jsonl")
    calib_path = args.calib_data or default_data
    h(f"CALIB  ({args.calib_set})  feeds SparseGPT pruning + static FP8 scales")
    print(f"calib data = {calib_path}")
    if args.calib_set == "gpqa":
        ds = q14.build_gpqa_calib(tokenizer, calib_path, args.num_calib_samples,
                                  args.max_seq_len, args.system_instruction)
    else:
        ds = q14.build_trace_calib(tokenizer, calib_path, args.num_calib_samples, args.max_seq_len)
    print(f"calibration samples = {len(ds)}   (max_seq_len={args.max_seq_len})")

    # ---- recipe: prune to 2:4 FIRST, then FP8 the surviving weights ----
    h(f"COMPRESS  2:4 sparse ({args.mask_structure}, sparsity={args.sparsity}) + {args.fp8_scheme}  targets=Linear")
    print(f"ignore (both steps) = {ignore}")
    recipe = [
        SparseGPTModifier(
            sparsity=args.sparsity,
            mask_structure=args.mask_structure,
            targets=["Linear"],
            ignore=ignore,
            dampening_frac=args.dampening_frac,
        ),
        QuantizationModifier(
            targets="Linear",
            scheme=args.fp8_scheme,
            ignore=ignore,
        ),
    ]
    oneshot(model=model, recipe=recipe, dataset=ds,
            num_calibration_samples=len(ds), max_seq_length=args.max_seq_len)

    text = q14.sanity_generate(model, tokenizer, f"AFTER 2:4+{args.fp8_scheme}")
    if len(text.strip()) < 5:
        ans = input("Degenerate output — save anyway? [y/N] ")
        if ans.strip().lower() != "y":
            sys.exit("!! aborted before save — inspect the model before re-running.")

    h(f"SAVE  ->  {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    # save_compressed=True emits BOTH a sparsity_config (sparse-24-bitmask) and the
    # quantization_config into config.json — that is exactly what vLLM<=0.18.1's
    # CompressedTensors24 path reads.
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)
    for f in EXTRA_FILES:
        src, dst = os.path.join(model_dir, f), os.path.join(out_dir, f)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)
            print(f"  copied {f} (not re-emitted by save_pretrained)")

    if not has_mtp:
        h("PATCH  mtp.* tensors back in (model class dropped them on load)")
        q14.copy_missing_mtp_tensors(model_dir, out_dir)

    def du(path):
        return sum(os.path.getsize(os.path.join(dp, fn))
                   for dp, _, fns in os.walk(path) for fn in fns)

    h("DONE")
    print(f"  {model_dir}  ->  {du(model_dir)/1e9:.2f} GB (BF16)")
    print(f"  {out_dir}  ->  {du(out_dir)/1e9:.2f} GB (2:4-sparse + {args.fp8_scheme})")
    print()
    print("Serve ONLY on vllm/vllm-openai:v0.18.1 (see serve/Dockerfile.sparse24-fp8). "
          "On >=0.19.0 vLLM raises 'Sparse24 models are no longer supported'.")


if __name__ == "__main__":
    main()
