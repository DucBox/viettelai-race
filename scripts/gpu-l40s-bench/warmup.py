#!/usr/bin/env python3
"""Warmup with SELF-GENERATED content only (not trace) to force Triton JIT
compilation of Mamba/GDN kernels (_zero_kv_blocks_kernel, _causal_conv1d_fwd_kernel,
etc.) to happen during startup instead of during the first graded requests.
Fires synthetic prompts spanning the same length range as the real trace
(12.9k-27.4k tokens) so every prefill-chunk shape gets JIT'd ahead of time.
"""
import asyncio
import random
import httpx

BASE_URL = "http://localhost:8000/v1/chat/completions"
VOCAB = ["node", "cluster", "throughput", "latency", "batch", "kernel", "runtime",
         "memory", "compute", "scale", "vector", "context", "stream", "buffer",
         "process", "model", "layer", "gradient", "index", "queue", "system"]


def gen_text(approx_tokens):
    # ~1.3 tokens/word rough estimate for this tokenizer; pad generously then trim later isn't needed for warmup purposes
    n_words = int(approx_tokens / 0.9)
    random.seed(approx_tokens)  # deterministic length, still not trace content
    return " ".join(random.choices(VOCAB, k=n_words))


async def fire(client, approx_tokens, n_messages):
    messages = [{"role": "system", "content": gen_text(200)}]
    remaining = approx_tokens
    for i in range(n_messages):
        chunk = remaining // (n_messages - i)
        messages.append({"role": "user" if i % 2 == 0 else "assistant",
                          "content": gen_text(chunk)})
        remaining -= chunk
    body = {
        "model": "Qwen3.5-2B",
        "messages": messages,
        "max_tokens": 5,
        "temperature": 0,
        "seed": 123,
    }
    try:
        r = await client.post(BASE_URL, json=body, timeout=60.0)
        print(f"  warmup req ~{approx_tokens}tok/{n_messages}msg -> HTTP {r.status_code}")
    except Exception as e:
        print(f"  warmup req ~{approx_tokens}tok/{n_messages}msg -> ERROR {e}")


async def main():
    # cover the same shape range as the real trace rounds (12.9k..27.4k, 2..12 msgs)
    targets = [(13000, 2), (16000, 4), (19000, 6), (22000, 8), (25000, 10), (27000, 12)]
    async with httpx.AsyncClient() as client:
        # fire a couple of concurrent bursts too, to also warm the multi-request path
        await asyncio.gather(*[fire(client, tok, n) for tok, n in targets])
    print("warmup done")


if __name__ == "__main__":
    asyncio.run(main())
