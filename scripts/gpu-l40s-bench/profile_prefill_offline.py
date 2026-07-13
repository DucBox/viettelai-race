#!/usr/bin/env python3
"""Profile 1 prefill bằng offline LLM (đúng kernel serving: fp8 + flashinfer GDN)."""
import random, time

def main():
    from vllm import LLM, SamplingParams
    from vllm.config.profiler import ProfilerConfig
    VOCAB=("node cluster throughput latency batch kernel runtime memory compute scale vector "
           "context stream buffer process model layer gradient index queue system tensor cache "
           "prefill decode token attention mamba delta rule conv scan proj norm quant").split()
    r=random.Random(1)
    prompt=" ".join(r.choice(VOCAB) for _ in range(7000))
    llm=LLM(model="/root/model", quantization="fp8", kv_cache_dtype="fp8", max_model_len=48000,
            gpu_memory_utilization=0.95, tensor_parallel_size=1, enable_prefix_caching=False,
            max_num_seqs=20, gdn_prefill_backend="flashinfer",
            profiler_config=ProfilerConfig(profiler="torch", torch_profiler_dir="/root/prof"))
    print(">> warmup x3")
    for _ in range(3):
        llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0), use_tqdm=False)
    print(">> START PROFILE + 1 prefill")
    llm.start_profile()
    t0=time.time()
    out=llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0), use_tqdm=False)
    print(f">> prefill done lat={(time.time()-t0)*1000:.0f}ms prompt_tok={len(out[0].prompt_token_ids)}")
    llm.stop_profile()
    time.sleep(3)
    print(">> DONE - trace o /root/prof")

if __name__=="__main__":
    main()
