#!/usr/bin/env python3
"""(b) BOC NHO frontend_prep — offline, KHONG server, KHONG GPU.

frontend_prep (arrival->queued) = tokenize + apply_chat_template + hash-prefix + handoff,
chay CPU frontend. Doan 242ms/prompt phinh theo prompt. Script nay do RIENG tung phan o
tung turn (prompt lon dan) -> biet cai nao chiem chinh + co ∝ prompt khong.

  PY probe_frontend_prep.py --model /root/model --trace /root/trace-round1.jsonl --reps 5
"""
import json, time, argparse, statistics as st
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/model")
    ap.add_argument("--trace", default="/root/trace-round1.jsonl")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--block", type=int, default=1072)  # block-align fp8-KV
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    # hash-prefix: proxy sha256-chained tren tung block (∝ prompt, xap xi chi phi hash cua vLLM)
    import hashlib
    def hash_chain(ids, block):
        h = b""
        for i in range(0, len(ids), block):
            h = hashlib.sha256(h + bytes(str(ids[i:i+block]), "utf-8")).digest()
    print("[hash] proxy: sha256-chained per-block (uoc luong O(prompt))")

    # doc trace -> messages per record, group theo turn_index (prompt lon dan)
    recs = []
    for line in open(a.trace):
        try:
            d = json.loads(line)
        except Exception:
            continue
        body = d.get("body", d)
        msgs = body.get("messages")
        if not msgs:
            continue
        # trace body khong co turn_index -> suy tu SO MESSAGE (turn0=2 msg, ... turn5=12 msg)
        recs.append((len(msgs), msgs))
    # 1 dai dien moi kich-thuoc-turn (msgs dai nhat)
    by_turn = {}
    for t, m in recs:
        if t not in by_turn or len(json.dumps(m)) > len(json.dumps(by_turn[t])):
            by_turn[t] = m

    print(f"\n{'turn':>4} {'n_tok':>7} {'template_ms':>11} {'tokenize_ms':>11} {'hash_ms':>8} {'TONG_ms':>8}")
    for t in sorted(by_turn):
        msgs = by_turn[t]
        # template-only (tokenize=False)
        def _tmpl():
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        try:
            text = _tmpl()
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            _tmpl = lambda: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        ntok = len(ids)

        def timed(fn, reps):
            xs = []
            for _ in range(reps):
                s = time.perf_counter(); fn(); xs.append((time.perf_counter()-s)*1000)
            return st.median(xs)
        t_tmpl = timed(_tmpl, a.reps)
        t_tok = timed(lambda: tok(text, add_special_tokens=False), a.reps)
        t_hash = timed(lambda: hash_chain(ids, a.block), a.reps)
        print(f"{t:>4} {ntok:>7} {t_tmpl:>11.1f} {t_tok:>11.1f} {t_hash:>8.1f} {t_tmpl+t_tok+t_hash:>8.1f}")

    print("\nDoc: cot nao ∝ n_tok manh nhat = thu pham chinh frontend_prep. "
          "Tren grader 3-core con nang hon (CPU cham + engine tranh core).")

if __name__ == "__main__":
    main()
