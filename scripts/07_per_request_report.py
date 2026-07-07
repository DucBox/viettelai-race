#!/usr/bin/env python3
"""Turn AIPerf's per-request records into a per-user / per-turn table — the
console summary table only shows aggregates (avg/p50/p90/...); this reads the
raw files AIPerf already writes on every run and reconstructs the detail view.

Reads, from the most recent (or a given) artifacts/<run>/ directory:
  - profile_export.jsonl        one line per request: session_num, turn_index,
                                 TTFT, TPOT(ITL), tokens, latency, timestamps.
  - server_metrics_export.jsonl time-series of raw vLLM /metrics scrapes
                                 (only present if the run used
                                 --server-metrics-formats ... jsonl).

For each request, finds the closest-in-time server-metrics scrape to correlate
KV cache usage / running-request count / prefix-cache counters. IMPORTANT
CAVEAT: those are GLOBAL vLLM engine metrics (shared across all concurrent
requests), not per-request instrumentation — vLLM does not expose "did THIS
specific request hit the prefix cache". What you get here is "what was the
system's state around the time this request ran", which is still the most
useful lens available without deeper OTLP tracing.

Usage:
    ./.venv/bin/python scripts/07_per_request_report.py
    ./.venv/bin/python scripts/07_per_request_report.py artifacts/qwen3.5-2b-openai-chat-concurrency20
"""
import bisect
import glob
import json
import os
import sys


def find_run_dir(argv):
    if len(argv) > 1:
        return argv[1]
    candidates = [d for d in glob.glob("artifacts/*") if os.path.isdir(d)]
    if not candidates:
        sys.exit("No artifacts/* directory found. Run bench/run_aiperf_baseline.sh first.")
    return max(candidates, key=os.path.getmtime)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def metric_val(sample, name):
    entries = sample.get("metrics", {}).get(name)
    if not entries:
        return None
    v = entries[0].get("value")
    return v


def main():
    run_dir = find_run_dir(sys.argv)
    print(f"run: {run_dir}")

    requests = load_jsonl(os.path.join(run_dir, "profile_export.jsonl"))
    if not requests:
        sys.exit(f"No profile_export.jsonl records found in {run_dir}")

    server_path = os.path.join(run_dir, "server_metrics_export.jsonl")
    server_samples = load_jsonl(server_path)
    if not server_samples:
        print(f"(no {server_path} — re-run with --server-metrics-formats ... jsonl "
              f"to get KV-cache/prefix-cache correlation; showing client-side metrics only)")
    server_samples.sort(key=lambda s: s["timestamp_ns"])
    sample_times = [s["timestamp_ns"] for s in server_samples]

    # Running cumulative counters need a fixed reference point to compute
    # "activity during this window" deltas — use the very first scrape as t0.
    def nearest_sample(ts_ns):
        if not sample_times:
            return None
        i = bisect.bisect_right(sample_times, ts_ns) - 1
        i = max(0, min(i, len(server_samples) - 1))
        return server_samples[i]

    base = server_samples[0] if server_samples else None
    base_hits = metric_val(base, "vllm:prefix_cache_hits_total") if base else None
    base_queries = metric_val(base, "vllm:prefix_cache_queries_total") if base else None

    rows = []
    for r in requests:
        meta = r.get("metadata", {})
        metrics = r.get("metrics") or {}
        if r.get("error"):
            rows.append({
                "session": meta.get("session_num"), "turn": meta.get("turn_index"),
                "ttft": None, "tpot": None, "in_tok": None, "out_tok": None,
                "latency": None, "error": r["error"].get("type", "error"),
                "end_ns": meta.get("request_end_ns"),
            })
            continue

        def mv(name):
            e = metrics.get(name)
            return e["value"] if e else None

        end_ns = meta.get("request_end_ns")
        s = nearest_sample(end_ns) if end_ns else None
        kv_pct = metric_val(s, "vllm:kv_cache_usage_perc") if s else None
        running = metric_val(s, "vllm:num_requests_running") if s else None
        hits = metric_val(s, "vllm:prefix_cache_hits_total") if s else None
        queries = metric_val(s, "vllm:prefix_cache_queries_total") if s else None
        d_hits = (hits - base_hits) if (hits is not None and base_hits is not None) else None
        d_queries = (queries - base_queries) if (queries is not None and base_queries is not None) else None

        rows.append({
            "session": meta.get("session_num"), "turn": meta.get("turn_index"),
            "ttft": mv("time_to_first_token"), "tpot": mv("inter_token_latency"),
            "in_tok": mv("input_sequence_length"), "out_tok": mv("output_sequence_length"),
            "latency": mv("request_latency"),
            "kv_pct": kv_pct, "running": running,
            "cum_hits": d_hits, "cum_queries": d_queries,
            "error": None, "end_ns": end_ns,
        })

    # Group by session, sort by turn — this is the "user 1 turn 1, turn 2, ..." view.
    by_session = {}
    for row in rows:
        by_session.setdefault(row["session"], []).append(row)
    for session_rows in by_session.values():
        session_rows.sort(key=lambda x: (x["turn"] is None, x["turn"]))

    def fmt(v, spec="{:.1f}"):
        return spec.format(v) if isinstance(v, (int, float)) else "-"

    have_server = any(row.get("kv_pct") is not None for row in rows)
    header = f"{'user':>5} {'turn':>4} {'TTFT(ms)':>9} {'TPOT(ms)':>9} {'in_tok':>7} {'out_tok':>7} {'lat(ms)':>9}"
    if have_server:
        header += f" {'KV%@end':>8} {'running':>7} {'Δhits':>8} {'Δqueries':>9}"
    print(header)
    print("-" * len(header))

    for session in sorted(by_session, key=lambda s: (s is None, s)):
        for row in by_session[session]:
            if row["error"]:
                print(f"{fmt(row['session'],'{:>5}'):>5} {fmt(row['turn'],'{:>4}'):>4}  ERROR: {row['error']}")
                continue
            line = (f"{row['session']:>5} {row['turn']:>4} "
                    f"{fmt(row['ttft']):>9} {fmt(row['tpot']):>9} "
                    f"{fmt(row['in_tok'],'{:.0f}'):>7} {fmt(row['out_tok'],'{:.0f}'):>7} "
                    f"{fmt(row['latency']):>9}")
            if have_server:
                kv_pct = row.get("kv_pct")
                kv_str = f"{kv_pct*100:.1f}%" if kv_pct is not None else "-"
                line += (f" {kv_str:>8} {fmt(row.get('running'),'{:.0f}'):>7} "
                         f"{fmt(row.get('cum_hits'),'{:.0f}'):>8} {fmt(row.get('cum_queries'),'{:.0f}'):>9}")
            print(line)
        print()

    if have_server:
        print("Note: KV%/running/Δhits/Δqueries are the GLOBAL vLLM engine state at the")
        print("nearest scrape to each request's completion (scraped ~every 333ms) — not a")
        print("per-request cache-hit flag (vLLM doesn't expose that). Δhits/Δqueries are")
        print("cumulative since the run started, shared across all concurrent requests.")


if __name__ == "__main__":
    main()
