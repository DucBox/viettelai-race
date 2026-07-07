#!/usr/bin/env python3
"""Turn AIPerf's per-request records into a per-user / per-turn table — the
console summary only shows aggregates (avg/p50/p90/...); this reads the raw
files AIPerf writes on every run and reconstructs the detail view, plus dumps a
full per-request CSV with every metric.

Reads, from the most recent (or a given) artifacts/<run>/ directory:
  - profile_export.jsonl        one line per request: session_num, turn_index,
                                 TTFT, TPOT(ITL), tokens, latency, and absolute
                                 ns timestamps (credit_issued/start/ack/end).
  - server_metrics_export.jsonl time-series of raw vLLM /metrics scrapes
                                 (only present if the run used
                                 --server-metrics-formats ... jsonl).

TRACE-REPLAY MODE (MODE=replay on the real trace)
-------------------------------------------------
The competition trace is 20 users x 6 turns = 120 independent requests, laid out
in request_id order: request_id % 20 = user, request_id // 20 = turn. AIPerf
replays each line as its own conversation, so metadata.session_num == request_id
(file order, verified) and metadata.turn_index is always 0. We therefore derive
(user, turn) from session_num rather than turn_index. Pass --users N to change
the 20-user assumption; auto-detected when turn_index is uniformly 0.

For SESSIONS mode (synthetic multi-turn) turn_index carries the real turn, so we
use session_num as the user and turn_index as the turn (the original behaviour).

Timestamps (all shown as ms relative to the run's first credit_issued):
  - arr   = credit_issued_ns : when AIPerf released the request per the fixed
            schedule (≈ the trace's own timestamp_ms — the "arrival").
  - start = request_start_ns : when the request actually went on the wire.
  - end   = request_end_ns   : when the full response finished streaming.
  - queue = start - arr      : scheduling/dispatch delay before it went out.

CAVEAT on server columns: KV%/running/Δhits/Δqueries are GLOBAL vLLM engine
state at the nearest scrape to each request's completion (shared across all
concurrent requests) — vLLM does not expose a per-request cache-hit flag.

Usage:
    ./.venv/bin/python scripts/07_per_request_report.py
    ./.venv/bin/python scripts/07_per_request_report.py artifacts/<run>
    ./.venv/bin/python scripts/07_per_request_report.py --users 20 artifacts/<run>
"""
import bisect
import csv
import glob
import json
import os
import sys


def parse_args(argv):
    users = None
    run_dir = None
    args = argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--users", "-u"):
            users = int(args[i + 1]); i += 2; continue
        if a.startswith("--users="):
            users = int(a.split("=", 1)[1]); i += 1; continue
        run_dir = a; i += 1
    return run_dir, users


def find_run_dir(explicit):
    if explicit:
        return explicit
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
    """Read a value out of a server_metrics_export.jsonl scrape."""
    entries = sample.get("metrics", {}).get(name)
    if not entries:
        return None
    return entries[0].get("value")


def main():
    explicit_dir, users_cli = parse_args(sys.argv)
    run_dir = find_run_dir(explicit_dir)

    requests = load_jsonl(os.path.join(run_dir, "profile_export.jsonl"))
    if not requests:
        sys.exit(f"No profile_export.jsonl records found in {run_dir}")

    server_path = os.path.join(run_dir, "server_metrics_export.jsonl")
    server_samples = load_jsonl(server_path)
    if not server_samples:
        print(f"(no {server_path} — re-run with --server-metrics-formats ... jsonl "
              f"for KV-cache/prefix-cache correlation; showing client-side metrics only)")
    server_samples.sort(key=lambda s: s["timestamp_ns"])
    sample_times = [s["timestamp_ns"] for s in server_samples]

    def nearest_sample(ts_ns):
        if not sample_times or ts_ns is None:
            return None
        i = bisect.bisect_right(sample_times, ts_ns) - 1
        i = max(0, min(i, len(server_samples) - 1))
        return server_samples[i]

    base = server_samples[0] if server_samples else None
    base_hits = metric_val(base, "vllm:prefix_cache_hits_total") if base else None
    base_queries = metric_val(base, "vllm:prefix_cache_queries_total") if base else None

    # --- decide (user, turn) mapping ---------------------------------------
    metas = [r.get("metadata", {}) for r in requests]
    turn_idxs = {m.get("turn_index") for m in metas}
    is_replay = turn_idxs == {0} or turn_idxs == {0, None}
    n = len(requests)
    if users_cli:
        users = users_cli
    elif is_replay and n % 20 == 0:
        users = 20            # competition trace default: 20 concurrent users
    else:
        users = None          # sessions mode: session_num already IS the user

    def user_turn(meta):
        sn = meta.get("session_num")
        if is_replay and users:
            if sn is None:
                return None, None
            return sn % users, sn // users
        return sn, meta.get("turn_index")

    # run t0 = earliest credit_issued (fallback request_start) for relative ms
    starts = [m.get("credit_issued_ns") or m.get("request_start_ns")
              for m in metas if (m.get("credit_issued_ns") or m.get("request_start_ns"))]
    t0 = min(starts) if starts else 0

    def rel_ms(ns):
        return (ns - t0) / 1e6 if ns else None

    rows = []
    for r in requests:
        meta = r.get("metadata", {})
        metrics = r.get("metrics") or {}
        user, turn = user_turn(meta)
        arr = meta.get("credit_issued_ns")
        start = meta.get("request_start_ns")
        end = meta.get("request_end_ns")

        def mv(name):
            e = metrics.get(name)
            return e.get("value") if e else None

        def first(*names):
            for nm in names:
                v = mv(nm)
                if v is not None:
                    return v
            return None

        if r.get("error"):
            rows.append({"user": user, "turn": turn, "req": meta.get("session_num"),
                         "error": r["error"].get("type", "error"),
                         "arr_ms": rel_ms(arr), "start_ms": rel_ms(start), "end_ms": rel_ms(end)})
            continue

        s = nearest_sample(end)
        hits = metric_val(s, "vllm:prefix_cache_hits_total") if s else None
        queries = metric_val(s, "vllm:prefix_cache_queries_total") if s else None
        rows.append({
            "user": user, "turn": turn, "req": meta.get("session_num"),
            "arr_ms": rel_ms(arr), "start_ms": rel_ms(start), "end_ms": rel_ms(end),
            "queue_ms": (rel_ms(start) - rel_ms(arr)) if (arr and start) else None,
            "ttft": mv("time_to_first_token"),
            "tpot": mv("inter_token_latency"),
            # ISL from client-side tokenization when present, else the server's
            # reported usage.prompt_tokens (populated on a real vLLM run).
            "in_tok": first("input_sequence_length", "usage_prompt_tokens"),
            "out_tok": first("output_sequence_length", "output_token_count", "usage_completion_tokens"),
            "latency": mv("request_latency"),
            "gen_tps": mv("output_token_throughput_per_user"),
            # Per-request prefix-cache signal from vLLM usage
            # (prompt_tokens_details.cached_tokens) — the real "did THIS request
            # hit the cache" number, unlike the global scrape columns below.
            "cache_rd": mv("usage_prompt_cache_read_tokens"),
            "cache_miss": mv("usage_prompt_cache_miss_tokens"),
            "kv_pct": metric_val(s, "vllm:kv_cache_usage_perc") if s else None,
            "running": metric_val(s, "vllm:num_requests_running") if s else None,
            "d_hits": (hits - base_hits) if (hits is not None and base_hits is not None) else None,
            "d_queries": (queries - base_queries) if (queries is not None and base_queries is not None) else None,
            "error": None,
        })

    have_server = any(row.get("kv_pct") is not None for row in rows)
    have_cache = any(row.get("cache_rd") is not None for row in rows)
    label = "trace-replay" if (is_replay and users) else "sessions"
    print(f"run: {run_dir}")
    print(f"mode: {label}" + (f"  ({users} users)" if (is_replay and users) else "")
          + f"   requests: {n}   t0=credit_issued of first request (all ms are relative to it)")
    print()

    def fmt(v, spec="{:.1f}"):
        return spec.format(v) if isinstance(v, (int, float)) else "-"

    header = (f"{'user':>4} {'turn':>4} {'arr':>8} {'start':>8} {'end':>8} {'queue':>6} "
              f"{'TTFT':>7} {'TPOT':>6} {'in_tok':>6} {'out':>5} {'lat':>8}")
    if have_cache:
        header += f" {'cache_rd':>8} {'hit%':>5}"
    if have_server:
        header += f" {'KV%':>6} {'run':>4} {'Δhits':>7} {'Δqry':>7}"
    print(header)
    print("-" * len(header))

    # group by user, sort by turn (the "user 0: turn0, turn1, ..." view)
    by_user = {}
    for row in rows:
        by_user.setdefault(row["user"], []).append(row)
    for user in sorted(by_user, key=lambda u: (u is None, u)):
        for row in sorted(by_user[user], key=lambda x: (x["turn"] is None, x["turn"])):
            if row.get("error"):
                print(f"{fmt(row['user'],'{:>4}'):>4} {fmt(row['turn'],'{:>4}'):>4} "
                      f"{fmt(row['arr_ms']):>8} {fmt(row['start_ms']):>8} {fmt(row['end_ms']):>8}  "
                      f"ERROR: {row['error']}")
                continue
            line = (f"{row['user']:>4} {row['turn']:>4} "
                    f"{fmt(row['arr_ms']):>8} {fmt(row['start_ms']):>8} {fmt(row['end_ms']):>8} "
                    f"{fmt(row['queue_ms']):>6} {fmt(row['ttft']):>7} {fmt(row['tpot']):>6} "
                    f"{fmt(row['in_tok'],'{:.0f}'):>6} {fmt(row['out_tok'],'{:.0f}'):>5} "
                    f"{fmt(row['latency']):>8}")
            if have_cache:
                crd, itok = row.get("cache_rd"), row.get("in_tok")
                hit = f"{crd / itok * 100:.0f}%" if (crd is not None and itok) else "-"
                line += f" {fmt(crd,'{:.0f}'):>8} {hit:>5}"
            if have_server:
                kv = row.get("kv_pct")
                line += (f" {(f'{kv*100:.1f}%' if kv is not None else '-'):>6} "
                         f"{fmt(row.get('running'),'{:.0f}'):>4} "
                         f"{fmt(row.get('d_hits'),'{:.0f}'):>7} {fmt(row.get('d_queries'),'{:.0f}'):>7}")
            print(line)
        print()

    # --- full CSV dump: every column, one row per request ------------------
    csv_path = os.path.join(run_dir, "per_request_report.csv")
    cols = ["user", "turn", "req", "arr_ms", "start_ms", "end_ms", "queue_ms",
            "ttft", "tpot", "in_tok", "out_tok", "latency", "gen_tps",
            "cache_rd", "cache_miss", "kv_pct", "running", "d_hits", "d_queries", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in sorted(rows, key=lambda x: (x["user"] is None, x["user"],
                                               x["turn"] is None, x["turn"])):
            w.writerow({c: row.get(c) for c in cols})
    print(f">> full per-request CSV (all columns): {csv_path}")

    if have_server:
        print()
        print("Note: KV%/run/Δhits/Δqueries are GLOBAL vLLM engine state at the nearest scrape")
        print("to each request's end (scraped ~every 333ms), shared across all concurrent")
        print("requests — not a per-request cache-hit flag (vLLM doesn't expose that).")
    elif not server_samples:
        print("(in_tok is blank when the server doesn't return prompt-token usage; a real vLLM")
        print(" run populates it. Re-run with --server-metrics-formats ... jsonl for KV/prefix.)")


if __name__ == "__main__":
    main()
