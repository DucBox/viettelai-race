#!/usr/bin/env python3
"""GPQA-Diamond accuracy harness — the independent accuracy gate (f(Δ)).

Fires one chat request per GPQA-Diamond question at the *running* vLLM server
(the same /v1/chat/completions the competition scores), parses the model's
letter choice, and reports accuracy + the competition gate f(Δ).

This is deliberately SEPARATE from scripts/08_ers_score.py:
  - 08 measures speed (ERS) from AIPerf artifacts and takes accuracy via --accuracy.
  - 09 (this) measures that accuracy by actually answering the 4-choice questions.

The gate (docs/VIETTEL AI RACE §1.3 accuracy gate):
    Baseline BF16 reference accuracy = 0.40 (measured on 100 fixed GPQA-Diamond Qs).
    Δ = 0.40 − your_accuracy
    f(Δ) = 1.0            if Δ ≤ 0.10   (accuracy ≥ 0.30  → no penalty)
         = (0.16−Δ)/0.06  if 0.10 < Δ < 0.16   (linear ramp)
         = 0.0            if Δ ≥ 0.16   (accuracy ≤ 0.24  → total wipeout, ×0)
Because the final Score multiplies by f(Δ), keeping accuracy ≥ 0.30 is the goal:
below 0.24 the whole run scores 0 no matter how fast it is.

FIRST JOB — validate the harness: run this against the untouched BF16 model and
confirm it lands near 0.40. If it reads ~0.25 (random) the harness is broken
(bad parse / wrong prompt), NOT the model. Only once BF16 ≈ 0.40 can you trust
the number while optimizing.

Note on "100 fixed questions": the organizer's gate uses a fixed 100-question
subset we don't have the exact membership of. This runs the full 198 by default
(a more stable estimate); use --limit / --offset to slice. A correct harness
should read ≈0.40 on any reasonable slice of BF16.

Depends on the Python STANDARD LIBRARY ONLY — runs inside the vLLM image as-is.

Usage:
    python scripts/09_gpqa_accuracy.py                     # full 198, localhost:8000
    python scripts/09_gpqa_accuracy.py --limit 20          # quick smoke on 20 Qs
    python scripts/09_gpqa_accuracy.py --concurrency 32 --max-tokens 2048
    python scripts/09_gpqa_accuracy.py --url http://host:8000 --model qwen3.5-2b
Outputs a summary + writes per-question results to artifacts/gpqa/<timestamp>/.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

# ── accuracy-gate constants (docs/VIETTEL AI RACE §1.3) ──────────────────────
BASELINE_ACC = 0.40
GATE_FULL, GATE_ZERO = 0.10, 0.16   # Δ thresholds
CHOICES = ("A", "B", "C", "D")

DEFAULT_SYSTEM = (
    "You are an expert answering a hard multiple-choice question. "
    "Choose the single best option. Respond with only the letter of the "
    "correct option (A, B, C, or D) and nothing else."
)


# ── data ─────────────────────────────────────────────────────────────────────
def load_questions(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── answer parsing ───────────────────────────────────────────────────────────
# Order matters: try the most explicit signals first, fall back to weaker ones.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{\s*([ABCD])\s*\}", re.IGNORECASE)
_PHRASE_RE = re.compile(
    r"(?:answer|correct option|choice|option)\s*(?:is|:|=)?\s*[\*`\"'(\[]*\s*([ABCD])\b",
    re.IGNORECASE,
)
_BOLD_RE = re.compile(r"\*\*\s*([ABCD])\s*\*\*")
_STANDALONE_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


def parse_choice(text):
    """Extract the chosen letter A–D from a raw completion. None if not found."""
    if not text:
        return None
    # Drop chain-of-thought so we score the FINAL answer, not reasoning mentions.
    body = _THINK_RE.sub(" ", text)
    # If an unclosed <think> remains (truncated), keep only what's after it.
    if "<think>" in body.lower() and "</think>" not in body.lower():
        body = re.split(r"<think>", body, flags=re.IGNORECASE)[-1]

    # Take the LAST match of each: reasoning discusses options early ("option A
    # is wrong ...") and states the real answer at the end ("... so the answer is D").
    for rx in (_BOXED_RE, _PHRASE_RE, _BOLD_RE):
        ms = list(rx.finditer(body))
        if ms:
            return ms[-1].group(1).upper()
    # Last standalone A–D wins (models often restate then conclude with the letter).
    hits = _STANDALONE_RE.findall(body)
    if hits:
        return hits[-1].upper()
    return None


# ── one request ──────────────────────────────────────────────────────────────
def ask(url, model, system, question, max_tokens, temperature, timeout):
    """Return (raw_text, error_str). error_str is None on success."""
    payload = {
        "model": model,
        "messages": (
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": question}]
        ),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]["content"], None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:200]}"
    except Exception as e:  # noqa: BLE001 — network/parse errors all count as failures
        return None, f"{type(e).__name__}: {e}"


# ── gate ─────────────────────────────────────────────────────────────────────
def f_delta(accuracy):
    delta = BASELINE_ACC - accuracy
    if delta <= GATE_FULL:
        f = 1.0
    elif delta >= GATE_ZERO:
        f = 0.0
    else:
        f = (GATE_ZERO - delta) / (GATE_ZERO - GATE_FULL)
    return delta, f


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="GPQA-Diamond accuracy gate harness.")
    ap.add_argument("--url", default=os.environ.get("URL", "http://localhost:8000"))
    ap.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "qwen3.5-2b"))
    ap.add_argument("--data", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "GPQA", "gpqa_diamond.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="use only the first N questions (after --offset)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=32, help="parallel in-flight requests (match --max-num-seqs)")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="cap on completion tokens; leave room if the model thinks before answering")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--system", default=DEFAULT_SYSTEM, help="system prompt ('' to disable)")
    ap.add_argument("--out", default=None, help="output dir (default artifacts/gpqa/<timestamp>)")
    args = ap.parse_args()

    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        sys.exit(f"data file not found: {data_path}")
    rows = load_questions(data_path)
    rows = rows[args.offset:]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("no questions selected")

    out_dir = args.out or os.path.join(
        "artifacts", "gpqa", dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"GPQA-Diamond accuracy gate")
    print(f"  server   : {args.url}  (model={args.model})")
    print(f"  questions: {len(rows)}  (offset={args.offset}, limit={args.limit})")
    print(f"  decode   : temp={args.temperature}  max_tokens={args.max_tokens}  concurrency={args.concurrency}")
    print(f"  out      : {out_dir}")
    print("  running ...", flush=True)

    results = [None] * len(rows)
    t0 = time.time()
    done = 0

    def work(idx):
        r = rows[idx]
        raw, err = ask(args.url, args.model, args.system, r["question"],
                       args.max_tokens, args.temperature, args.timeout)
        pred = parse_choice(raw) if raw is not None else None
        gold = str(r["answer"]).strip().upper()
        return idx, {
            "id": r.get("id", idx),
            "gold": gold,
            "pred": pred,
            "correct": (pred == gold),
            "error": err,
            "unparsed": (err is None and pred is None),
            "raw": raw,
        }

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, i) for i in range(len(rows))]
        for fut in cf.as_completed(futs):
            idx, res = fut.result()
            results[idx] = res
            done += 1
            if done % 20 == 0 or done == len(rows):
                print(f"    {done}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)

    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_err = sum(1 for r in results if r["error"])
    n_unparsed = sum(1 for r in results if r["unparsed"])
    accuracy = n_correct / n if n else 0.0
    delta, f = f_delta(accuracy)

    # persist
    with open(os.path.join(out_dir, "results.jsonl"), "w") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "url": args.url, "model": args.model, "n": n,
        "correct": n_correct, "accuracy": accuracy,
        "errors": n_err, "unparsed": n_unparsed,
        "baseline_acc": BASELINE_ACC, "delta": delta, "f_delta": f,
        "elapsed_s": time.time() - t0,
        "params": {"max_tokens": args.max_tokens, "temperature": args.temperature,
                   "concurrency": args.concurrency, "system": args.system},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # per-gold breakdown (spot a parse bias — e.g. always guessing 'A')
    print()
    print(f"accuracy = {n_correct}/{n} = {accuracy:.4f}")
    print(f"errors (request failed): {n_err}    unparsed (no letter found): {n_unparsed}")
    print(f"gold vs pred distribution:")
    print(f"  {'letter':>6} {'gold_n':>6} {'pred_n':>6} {'correct':>7}")
    for L in CHOICES:
        gold_n = sum(1 for r in results if r["gold"] == L)
        pred_n = sum(1 for r in results if r["pred"] == L)
        corr = sum(1 for r in results if r["gold"] == L and r["correct"])
        print(f"  {L:>6} {gold_n:>6} {pred_n:>6} {corr:>7}")
    print("-" * 48)
    if delta <= GATE_FULL:
        verdict = "PASS (Δ≤0.10, no penalty)"
    elif delta >= GATE_ZERO:
        verdict = "WIPEOUT (Δ≥0.16, f=0 → Score ×0)"
    else:
        verdict = "PARTIAL (linear penalty zone)"
    print(f"baseline={BASELINE_ACC:.2f}  Δ={delta:+.4f}  f(Δ)={f:.3f}   → {verdict}")
    print(f"wrote {out_dir}/summary.json + results.jsonl")


if __name__ == "__main__":
    main()
