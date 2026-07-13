#!/usr/bin/env python3
"""Trace mỗi flag vLLM tới NƠI NÓ ĐƯỢC TIÊU THỤ trong source (không phải chỗ
argparse định nghĩa). Với mỗi flag 'dest', grep `\\bdest\\b` trong cây vllm/,
phân loại: [DEF] = định nghĩa config/argparse (bỏ), [USE] = nơi thực sự đọc giá
trị để đổi hành vi. In file:line + dòng, để đọc tiếp file consumer.

    python3 flag_trace.py /usr/local/lib/python3.12/dist-packages/vllm > trace.txt
"""
import os
import re
import sys

# dest (flag name -> underscore) cho các flag CÓ NGHĨA với Track 3
FLAGS = {
 "CACHE/KV": ["gpu_memory_utilization","kv_cache_dtype","calculate_kv_scales","block_size",
   "kv_cache_memory_bytes","num_gpu_blocks_override","enable_prefix_caching",
   "prefix_caching_hash_algo","kv_sharing_fast_prefill","kv_cache_dtype_skip_layers"],
 "SCHEDULER": ["max_num_batched_tokens","max_num_seqs","enable_chunked_prefill",
   "long_prefill_token_threshold","max_num_partial_prefills","max_long_partial_prefills",
   "scheduling_policy","disable_hybrid_kv_cache_manager","async_scheduling",
   "scheduler_reserve_full_isl","stream_interval"],
 "GRAPH/COMPILE": ["enforce_eager","cudagraph_capture_sizes","max_cudagraph_capture_size",
   "performance_mode","optimization_level"],
 "QUANT": ["quantization","linear_backend"],
 "MAMBA/GDN": ["mamba_cache_mode","gdn_prefill_backend","mamba_cache_dtype",
   "mamba_ssm_cache_dtype","mamba_block_size","mamba_backend"],
 "ATTENTION": ["attention_backend","enable_flashinfer_autotune","disable_cascade_attn"],
 "MODEL": ["max_model_len","dtype"],
 "FRONTEND/RENDER": ["renderer_num_workers","tokenizer_mode","skip_tokenizer_init",
   "enable_prompt_tokens_details","disable_log_stats","disable_uvicorn_access_log",
   "enable_log_requests"],
 "MM": ["language_model_only","skip_mm_profiling"],
 "LOAD": ["safetensors_load_strategy","safetensors_prefetch_num_threads"],
 "SPEC": ["spec_method","spec_tokens"],
}

DEF_HINTS = ("cli_args", "arg_utils", "/config/", "config.py", "EngineArgs", "add_argument")


def main():
    root = sys.argv[1]
    files = []
    for dp, _, fns in os.walk(root):
        if "/test" in dp:
            continue
        for fn in fns:
            if fn.endswith(".py"):
                files.append(os.path.join(dp, fn))
    # đọc 1 lần
    cache = {}
    for f in files:
        try:
            cache[f] = open(f, encoding="utf-8", errors="ignore").read().splitlines()
        except Exception:
            pass

    for group, dests in FLAGS.items():
        print(f"\n{'='*70}\n### {group}\n{'='*70}")
        for d in dests:
            pat = re.compile(rf"\b{re.escape(d)}\b")
            uses = []
            for f, lines in cache.items():
                short = f.replace(root, "vllm")
                is_def = any(h in f for h in DEF_HINTS)
                for i, ln in enumerate(lines, 1):
                    if pat.search(ln):
                        tagdef = "[DEF]" if is_def else "[USE]"
                        uses.append((tagdef, short, i, ln.strip()[:130]))
            n_use = sum(1 for u in uses if u[0] == "[USE]")
            print(f"\n--- {d}  ({n_use} USE hit) ---")
            # in USE truoc (quan trong), toi da 8; roi 2 DEF
            shown = 0
            for tag, sf, i, ln in uses:
                if tag == "[USE]":
                    print(f"  {tag} {sf}:{i}  {ln}")
                    shown += 1
                    if shown >= 8:
                        break
            defs = [u for u in uses if u[0] == "[DEF]"][:2]
            for tag, sf, i, ln in defs:
                print(f"  {tag} {sf}:{i}  {ln}")


if __name__ == "__main__":
    main()
