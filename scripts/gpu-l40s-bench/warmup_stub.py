#!/usr/bin/env python3
"""Warmup ON ĐỊNH (không phải cải thiện): bắn vài request NỘI DUNG TỰ SINH
(KHÔNG lấy từ trace) để endpoint/model ổn định trước khi bench, tránh rep đầu
lệch do JIT/cold. Hợp lệ: không dùng nội dung trace, chỉ chữ vô nghĩa.
2 request: 1 ngắn + 1 vừa (~2k token) để chạm vài shape prefill khác nhau.
"""
import sys
import time
import httpx

URL = "http://localhost:8000/v1/chat/completions"
VOCAB = ("node cluster throughput latency batch kernel runtime memory compute "
         "scale vector context stream buffer process model layer gradient index "
         "queue system tensor cache prefill decode token").split()


def gen(n_words):
    import random
    r = random.Random(0)
    return " ".join(r.choice(VOCAB) for _ in range(n_words))


def fire(text, max_tokens):
    body = {"model": "Qwen3.5-2B",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
    t0 = time.time()
    r = httpx.post(URL, json=body, timeout=120.0)
    dt = (time.time() - t0) * 1000
    ok = r.status_code == 200
    n = len(r.json().get("choices", [{}])[0].get("message", {}).get("content", "")) if ok else 0
    print(f"  warmup req: status={r.status_code} words_in={len(text.split())} "
          f"lat={dt:.0f}ms out_chars={n}")
    return ok


def main():
    print("=== warmup (synthetic, non-trace) ===")
    ok = True
    ok &= fire(gen(100), 16)      # ngắn
    ok &= fire(gen(2000), 16)     # vừa (~2k+ token) — chạm shape prefill dài hơn
    print("=== warmup OK ===" if ok else "!! warmup có request lỗi")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
