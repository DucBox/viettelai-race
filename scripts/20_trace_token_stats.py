#!/usr/bin/env python3
"""Thống kê token in/out THẬT của trace-round1.jsonl bằng tokenizer thật của
Qwen3.5-2B (KHÔNG dùng proxy cl100k_base như tài liệu BTC — số của BTC lệch
~6-7% vì tiktoken sinh ít token hơn tokenizer thật của Qwen cho cùng văn bản).

Chạy:
    .venv/bin/python3 scripts/20_trace_token_stats.py

Cần: transformers + tokenizers trong .venv, và serve/models/qwen3.5-2b/ (weights
+ tokenizer + chat_template.jinja) đã có sẵn tại chỗ.

Kết quả đã ghi lại (2026-07-10) trong docs/trace-round1-token-stats.md — file này
là nguồn tái tạo, chạy lại nếu trace hoặc tokenizer thay đổi.
"""
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
TRACE = REPO / "data" / "trace-round1.jsonl"
MODEL = REPO / "serve" / "models" / "qwen3.5-2b"
ROUND_GAP_MS = 300  # IAT lớn hơn ngưỡng này = ranh giới round (thực đo gap ~4525ms)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)

    rows = []
    with open(TRACE) as f:
        for line in f:
            rows.append(json.loads(line))

    # Prefill token count = kết quả apply_chat_template(add_generation_prompt=True),
    # đúng thứ vLLM thực sự đưa vào model (kèm special token + role tag + assistant
    # turn opening), không phải chỉ nối content.
    def prefill_len(messages):
        return len(
            tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )

    prefill = [prefill_len(r["body"]["messages"]) for r in rows]
    max_tokens = [r["body"]["max_tokens"] for r in rows]
    timestamps = [r["timestamp_ms"] for r in rows]

    print(f"Tổng request: {len(rows)}")
    print(f"\n=== PREFILL TOKEN (tokenizer + chat template THẬT) ===")
    print(
        f"min={min(prefill)} max={max(prefill)} "
        f"mean={stats.mean(prefill):.0f} median={stats.median(prefill):.0f}"
    )
    print(f"tổng prefill (120 req, KHÔNG cache) = {sum(prefill):,}")

    print(f"\n=== OUTPUT (max_tokens) ===")
    print(f"cố định = {max_tokens[0]} (đồng nhất: {len(set(max_tokens)) == 1})")
    print(f"tỷ lệ prefill:decode ≈ {sum(prefill) / sum(max_tokens):.1f} : 1")

    # Nhóm theo số message (kỳ vọng 6 nhóm x 20 theo cấu trúc session)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[len(r["body"]["messages"])].append(prefill[i])
    print(f"\n=== NHÓM THEO SỐ MESSAGE (6 nhóm x 20 = 20 session x 6 lượt) ===")
    for k in sorted(groups):
        v = groups[k]
        print(f"  {k:2d} msg: {len(v)} req | mean={stats.mean(v):.0f} "
              f"min={min(v)} max={max(v)}")

    # Ranh giới round theo IAT
    iat = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    rounds = [[0]]
    for i, gap in enumerate(iat):
        if gap > ROUND_GAP_MS:
            rounds.append([])
        rounds[-1].append(i + 1)
    print(f"\n=== ROUND (phát hiện qua IAT > {ROUND_GAP_MS}ms) ===")
    for ri, idxs in enumerate(rounds):
        ts0, ts1 = timestamps[idxs[0]], timestamps[idxs[-1]]
        nmsg = {len(rows[i]["body"]["messages"]) for i in idxs}
        print(f"  Round {ri+1}: {len(idxs)} req | {ts0}-{ts1}ms "
              f"(span {ts1-ts0}ms) | msg={nmsg}")

    # Prefix continuity: request thứ i của round N là prefix của request thứ i
    # của round N+1? (xác nhận 20 session nối dài, không phải 120 req rời rạc)
    mism = comp = 0
    for r in range(len(rounds) - 1):
        A = [rows[i] for i in rounds[r]]
        B = [rows[i] for i in rounds[r + 1]]
        for i in range(min(len(A), len(B))):
            ta = "".join(m["content"] for m in A[i]["body"]["messages"])
            tb = "".join(m["content"] for m in B[i]["body"]["messages"])
            comp += 1
            mism += not tb.startswith(ta)
    print(f"\n=== PREFIX CONTINUITY ===")
    print(f"{comp} cặp so sánh, {mism} không khớp "
          f"({'100% khớp' if mism == 0 else 'CÓ SAI LỆCH'})")

    # Ngân sách token: không cache vs cache hoàn hảo
    # round 1 = cold start (không né được); round 2-6 = chỉ phần token MỚI mỗi lượt
    round_totals = [sum(prefill[i] for i in idxs) for idxs in rounds]
    r1_cold = round_totals[0]
    # token mới mỗi round 2..6 = prefill round N - prefill round N-1 (theo từng session)
    new_tokens_r2_6 = 0
    for r in range(1, len(rounds)):
        for i in range(len(rounds[r])):
            cur = prefill[rounds[r][i]]
            prev = prefill[rounds[r - 1][i]]
            new_tokens_r2_6 += cur - prev
    perfect_cache = r1_cold + new_tokens_r2_6
    print(f"\n=== NGÂN SÁCH TOKEN PREFILL ===")
    print(f"KHÔNG cache (tính lại từ đầu mỗi req) = {sum(prefill):,}")
    print(f"Round 1 cold-start (không né được)     = {r1_cold:,}")
    print(f"Token MỚI round 2-6 (nếu cache hoàn hảo) = {new_tokens_r2_6:,}")
    print(f"Tổng nếu CACHE HOÀN HẢO                 = {perfect_cache:,}")
    print(f"→ cache hoàn hảo giảm compute {sum(prefill)/perfect_cache:.2f}x")


if __name__ == "__main__":
    main()
