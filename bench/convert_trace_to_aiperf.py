#!/usr/bin/env python3
"""Convert the competition trace (data/trace-round1.jsonl) into an AIPerf
mooncake_trace input file so AIPerf replays the REAL requests verbatim on a
fixed schedule — instead of the synthetic word-salad it generates itself.

Why this shape works
--------------------
AIPerf's ``mooncake_trace`` custom-dataset type accepts four per-line input
modes; two carry real content: ``messages`` (an OpenAI-style message array that
AIPerf still wraps with --model / --streaming) and ``payload`` (a fully verbatim
body that bypasses all formatting). We use ``messages`` because:
  - AIPerf applies --model (routes to SERVED_MODEL_NAME, not the trace's literal
    "Qwen3.5-2B") and --streaming (so TTFT/TPOT are measured normally), and
  - the trace's fixed sampling params travel with each record: max_tokens via
    ``output_length``, and temperature/seed via ``extra`` (shallow-merged into
    the request body at dispatch). The output file is thus self-contained.

Each of the 120 trace records already contains its FULL conversation history in
body.messages, so every line becomes one independent conversation fired at its
own ``timestamp_ms``. Records that share the same 20 original sessions still
share a long common prefix (same system prompt + repeated earlier turns), which
is exactly what exercises vLLM's prefix cache — we do NOT collapse them into
sessions; the fixed schedule replays all 120 as the organizer's harness would.

We deliberately do NOT set ignore_eos: the trace doesn't, so the model stops at
its natural EOS (or 200 tokens), which is the real scored behavior.

Usage:
    python3 bench/convert_trace_to_aiperf.py                 # full -> data/trace-round1.aiperf.jsonl
    python3 bench/convert_trace_to_aiperf.py --short          # quick pipeline dry-run on the 10-char preview
    python3 bench/convert_trace_to_aiperf.py IN.jsonl OUT.jsonl
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def convert_record(rec):
    """Map one trace line to one AIPerf mooncake_trace `messages` line."""
    body = rec.get("body") or {}
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"record {rec.get('request_id')} has no body.messages array")

    out = {
        "timestamp": rec["timestamp_ms"],   # ms; drives --fixed-schedule replay
        "messages": messages,               # sent as-is (full history per record)
        "output_length": body.get("max_tokens", 200),  # -> per-request max_tokens
    }
    # Fixed sampling params ride along per-record so the file is self-contained
    # (AIPerf shallow-merges `extra` into the request body at dispatch time).
    extra = {}
    if "temperature" in body:
        extra["temperature"] = body["temperature"]
    if "seed" in body:
        extra["seed"] = body["seed"]
    if extra:
        out["extra"] = extra
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile", nargs="?", help="input trace jsonl (default: data/trace-round1.jsonl)")
    ap.add_argument("outfile", nargs="?", help="output aiperf jsonl (default: data/trace-round1.aiperf.jsonl)")
    ap.add_argument("--short", action="store_true",
                    help="use data/trace-round1.short.jsonl (10-char preview) for a fast pipeline check")
    args = ap.parse_args()

    default_in = "data/trace-round1.short.jsonl" if args.short else "data/trace-round1.jsonl"
    infile = args.infile or os.path.join(REPO_ROOT, default_in)
    if args.outfile:
        outfile = args.outfile
    else:
        stem = "trace-round1.short" if args.short else "trace-round1"
        outfile = os.path.join(REPO_ROOT, "data", f"{stem}.aiperf.jsonl")

    if not os.path.exists(infile):
        sys.exit(f"input not found: {infile}")

    n = 0
    ts_first = ts_last = None
    with open(infile) as fin, open(outfile, "w") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out = convert_record(rec)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                sys.exit(f"{infile}:{lineno}: {e}")
            if ts_first is None:
                ts_first = out["timestamp"]
            ts_last = out["timestamp"]
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1

    print(f">> wrote {n} records -> {outfile}")
    print(f">> arrival window: {ts_first} .. {ts_last} ms "
          f"({(ts_last - ts_first) / 1000:.1f}s of traffic)")
    print(">> replay it with:  MODE=replay ./bench/run_aiperf_baseline.sh")
    if args.short:
        print(">> NOTE: --short uses 10-char-truncated content — pipeline smoke test only, "
              "not a real latency/throughput measurement.")


if __name__ == "__main__":
    main()
