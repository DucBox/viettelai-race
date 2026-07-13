#!/usr/bin/env python3
"""Profile PREFILL + DECODE per-op cho 1 config (offline LLM, đúng kernel serving).
Dùng cho mọi A/B: chạy cho từng config -> so breakdown kernel để tìm root-cause nhỏ.
  python3 profile_config.py --tag v23 --seqs 20 --batch 3216 --decode-batch 20
"""
import argparse, random, time, os

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seqs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=3216)
    ap.add_argument("--quant", default="fp8")          # fp8 | none
    ap.add_argument("--kv", default="fp8")             # fp8 | auto
    ap.add_argument("--decode-batch", type=int, default=20)  # số seq decode đồng thời
    ap.add_argument("--outdir", default="/root/prof")
    a=ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.config.profiler import ProfilerConfig
    VOCAB="node cluster throughput latency batch kernel runtime memory compute scale vector context stream buffer process model layer gradient index queue system tensor cache prefill decode token".split()
    rnd=random.Random(1)
    long_prompt=" ".join(rnd.choice(VOCAB) for _ in range(7000))   # ~8k tok cho prefill
    short_prompt=" ".join(rnd.choice(VOCAB) for _ in range(50))    # ngắn -> decode chiếm phần lớn

    kw=dict(model="/root/model", max_model_len=48000, gpu_memory_utilization=0.90,
            tensor_parallel_size=1, enable_prefix_caching=False, max_num_seqs=a.seqs,
            max_num_batched_tokens=a.batch, gdn_prefill_backend="flashinfer",
            profiler_config=ProfilerConfig(profiler="torch", torch_profiler_dir=a.outdir))
    if a.quant=="fp8": kw["quantization"]="fp8"
    if a.kv=="fp8": kw["kv_cache_dtype"]="fp8"
    llm=LLM(**kw)
    for _ in range(3): llm.generate([long_prompt], SamplingParams(max_tokens=1,temperature=0), use_tqdm=False)

    print(f">> [{a.tag}] PROFILE PREFILL (1 seq, ~8k tok)")
    llm.start_profile("PREFILL_"+a.tag)
    o=llm.generate([long_prompt], SamplingParams(max_tokens=1,temperature=0), use_tqdm=False)
    llm.stop_profile(); time.sleep(2)
    print(f"   prefill prompt_tok={len(o[0].prompt_token_ids)}")

    print(f">> [{a.tag}] PROFILE DECODE ({a.decode_batch} seq song song, max_tokens=40)")
    llm.start_profile("DECODE_"+a.tag)
    llm.generate([short_prompt]*a.decode_batch, SamplingParams(max_tokens=40,temperature=0), use_tqdm=False)
    llm.stop_profile(); time.sleep(3)
    print(f">> [{a.tag}] DONE")

if __name__=="__main__": main()
