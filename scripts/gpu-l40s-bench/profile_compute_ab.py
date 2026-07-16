#!/usr/bin/env python3
"""TANG 4+5 (compute breakdown) — 1 config, offline LLM, KHONG SUDO.

Do 2 thu cho ca PREFILL (shape ~12947 tok, 1 seq, KHONG prefix-cache) va DECODE (20 seq):
  (A) WALL SACH  = time.perf_counter quanh generate(), KHONG gan profiler
                   -> tu so nay + gpu_busy(profiler) suy ra OVERHEAD = wall - gpu_busy (T3d).
  (B) TRACE      = torch profiler 1 lan -> gpu_busy per-op + tax-kernel (T3b/T3c).
Optional: NVTX group-range quanh module (GDN mixer / attention / MLP) de parse tach GEMM
          theo layer-group. Chi chay khi engine in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0).

  PY profile_compute_ab.py --config fp8|noquant --out /root/compute_bd --ntok 12947 --reps 5
"""
import os, sys, time, json, random, argparse

# LUU Y: KHONG dung NVTX module-hook cho layer-group — no goi torch.cuda.nvtx.range_push
# (tra ve int) -> dynamo vo torch.compile (VLLM_COMPILE). Va duoi compile+cudagraph hook
# cung khong fire luc inference that. -> Layer-group can 1 run enforce_eager rieng (sau).
# Core metrics (op-cat / tax / overhead / decode) KHONG can hook, chay o che do compile = giong serving.

def build_llm(config, ntok_max):
    from vllm import LLM
    from vllm.config.profiler import ProfilerConfig
    kw = dict(model="/root/model", max_model_len=max(ntok_max + 512, 16000),
              gpu_memory_utilization=0.40, tensor_parallel_size=1,
              enable_prefix_caching=False, max_num_seqs=24,
              gdn_prefill_backend="flashinfer",
              profiler_config=ProfilerConfig(profiler="torch", torch_profiler_dir=os.environ["PROF_DIR"]))
    if config == "fp8":
        kw.update(quantization="fp8", kv_cache_dtype="fp8")
    elif config == "noquant":
        pass  # bf16 W + bf16 KV
    else:
        sys.exit(f"config la {config}? chi fp8|noquant")
    return LLM(**kw)

# ---- optional NVTX group hooks (layer-group). Fail-safe: khong fire cung khong sao. ----
def install_group_nvtx():
    import torch
    PATS = {"GDN": ("gateddelta", "lineattn", "linear_attn", "deltanet"),
            "ATTN": ("attention",), "MLP": ("mlp",)}
    def grp(mod):
        n = type(mod).__name__.lower()
        for g, keys in PATS.items():
            if any(k in n for k in keys):
                return g
        return None
    def pre(mod, inp):
        g = grp(mod)
        if g:
            torch.cuda.nvtx.range_push(f"GRP:{g}")
            mod.__nvtx_on = True
    def post(mod, inp, out):
        if getattr(mod, "__nvtx_on", False):
            torch.cuda.nvtx.range_pop(); mod.__nvtx_on = False
    try:
        torch.nn.modules.module.register_module_forward_pre_hook(pre)
        torch.nn.modules.module.register_module_forward_hook(post)
        print("  [nvtx] global module hooks installed")
    except Exception as e:
        print(f"  [nvtx] skip ({e})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="/root/compute_bd")
    ap.add_argument("--ntok", type=int, default=12947)   # = turn0/user0 uncached
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.environ["PROF_DIR"] = os.path.join(a.out, f"prof_{a.config}")

    from vllm import SamplingParams
    llm = build_llm(a.config, a.ntok)
    rnd = random.Random(12947)
    pre_ids = [rnd.randint(1000, 30000) for _ in range(a.ntok)]          # exact token count, no cache
    dec_prompts = [{"prompt_token_ids": [rnd.randint(1000, 30000) for _ in range(2000)]} for _ in range(20)]
    sp1 = SamplingParams(max_tokens=1, temperature=0)
    spD = SamplingParams(max_tokens=40, temperature=0)

    print(f">> warmup x3 ({a.config})")
    for _ in range(3):
        llm.generate([{"prompt_token_ids": pre_ids}], sp1, use_tqdm=False)

    # (A) WALL SACH — KHONG profiler
    pf_wall = []
    for _ in range(a.reps):
        t = time.perf_counter()
        llm.generate([{"prompt_token_ids": pre_ids}], sp1, use_tqdm=False)
        pf_wall.append((time.perf_counter() - t) * 1000)
    # DECODE wall = phep TRU VI PHAN: (20 prompt, max_tokens=41) - (max_tokens=1) chia 40.
    # Ca 2 lan deu prefill 20 prompt -> tru di la khu het prefill -> con THUAN decode/step.
    sp41 = SamplingParams(max_tokens=41, temperature=0)
    t = time.perf_counter(); llm.generate(dec_prompts, sp1, use_tqdm=False); w1 = (time.perf_counter()-t)*1000
    t = time.perf_counter(); llm.generate(dec_prompts, sp41, use_tqdm=False); w41 = (time.perf_counter()-t)*1000
    dec_per_step = (w41 - w1) / 40.0
    clean = {"config": a.config, "ntok": a.ntok, "n_dec_seqs": len(dec_prompts),
             "prefill_wall_ms": sorted(pf_wall),
             "prefill_wall_median_ms": sorted(pf_wall)[len(pf_wall)//2],
             "decode_w1_ms": w1, "decode_w41_ms": w41,
             "decode_wall_per_step_ms": dec_per_step}
    json.dump(clean, open(os.path.join(a.out, f"clean_{a.config}.json"), "w"), indent=2)
    print(f">> WALL sach: prefill median={clean['prefill_wall_median_ms']:.1f}ms "
          f"| decode/step(vi phan)={dec_per_step:.2f}ms")

    # (B) TRACE — 2 SESSION RIENG de gpu_busy khong bi nhiem:
    #   PF : chi prefill 12947 (1 req)          -> prefill gpu_busy SACH
    #   DEC: 20-seq decode batch (co 20 prefill) -> chi dung DECODE-phase (prefill-phase bo)
    import glob as _glob
    def _mark(suffix):
        for f in _glob.glob(os.path.join(os.environ["PROF_DIR"], "*.pt.trace.json*")):
            if "_PF." in f or "_DEC." in f:
                continue
            os.rename(f, f.replace(".pt.trace.json", f"_{suffix}.pt.trace.json"))
    print(">> PROFILE session PF (prefill-only)")
    llm.start_profile(); llm.generate([{"prompt_token_ids": pre_ids}], sp1, use_tqdm=False)
    llm.stop_profile(); time.sleep(3); _mark("PF")
    print(">> PROFILE session DEC (decode batch)")
    try:
        llm.start_profile(); llm.generate(dec_prompts, spD, use_tqdm=False)
        llm.stop_profile(); time.sleep(3); _mark("DEC")
    except Exception as e:
        print(f"  [DEC profile skip] {e}")
    print(f">> DONE {a.config} -> trace {os.environ['PROF_DIR']} , clean_{a.config}.json")

if __name__ == "__main__":
    main()
