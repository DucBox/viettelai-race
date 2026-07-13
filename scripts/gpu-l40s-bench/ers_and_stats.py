#!/usr/bin/env python3
"""Đọc merged full.json (1 rep) -> tính:
  1. ERS thật (đúng công thức BTC: F/C/gamma/W như 08_ers_score.py) — trọng tài A/B.
     Dùng số CLIENT-observed (ttft_ms + client_mean_chunk_gap_ms) = gần aiperf nhất,
     đồng thời in kèm ERS tính từ server_mean_tpot để đối chiếu.
  2. Overall percentiles (mean/median/p90/p95/p99/max) cho:
     queue / prefill / decode / ttft / tpot.
  3. Per-turn (0..5) breakdown: queue, prefill, ttft, tpot, num_cached_tokens.
  4. Prefix-cache: mean cached tokens tổng + theo turn.
  5. GPU/CPU trong lúc bench (từ samples.json): util/sm_clock/power/temp + throttle.
Ghi score_summary.json (machine-readable) + in bảng người đọc.
"""
import argparse
import json
import statistics as st

F_TTFT, C_TTFT = 100.0, 1500.0
F_TPOT, C_TPOT = 20.0, 45.0
GAMMA, W = 2.0, 0.5


def comp(x, F, C):
    if x is None:
        return 0.0
    v = (C - x) / (C - F)
    v = max(0.0, min(1.0, v))
    return v ** GAMMA


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = min(len(xs) - 1, int(p * len(xs)))
    return xs[k]


def stats_block(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": round(st.mean(xs), 1), "median": round(st.median(xs), 1),
            "p90": round(pct(xs, .90), 1), "p95": round(pct(xs, .95), 1),
            "p99": round(pct(xs, .99), 1), "max": round(max(xs), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--samples", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = json.load(open(args.full))
    n_total = len(rows)
    ok = [r for r in rows if r.get("ttft_ms") is not None]
    n_fail = n_total - len(ok)

    # ---- ERS (client-observed) ----
    def ers_over(rows, tpot_key):
        s = 0.0
        for r in rows:
            ttft = r.get("ttft_ms")
            tpot = r.get(tpot_key)
            if ttft is None:
                s += 0.0
                continue
            s += W * comp(ttft, F_TTFT, C_TTFT) + (1 - W) * comp(tpot, F_TPOT, C_TPOT)
        return s / len(rows) if rows else 0.0

    ers_client = ers_over(rows, "client_mean_chunk_gap_ms")
    ers_server = ers_over(rows, "server_mean_tpot_ms")

    # mean components (chỉ trên request ok, để thấy s_ttft vs s_tpot lệch đâu)
    s_ttft = st.mean([comp(r["ttft_ms"], F_TTFT, C_TTFT) for r in ok]) if ok else 0
    s_tpot_c = st.mean([comp(r.get("client_mean_chunk_gap_ms"), F_TPOT, C_TPOT) for r in ok]) if ok else 0

    # ---- overall percentiles ----
    overall = {
        "queue_ms": stats_block([r.get("server_queue_ms") for r in ok]),
        "prefill_ms": stats_block([r.get("server_prefill_ms") for r in ok]),
        "decode_ms": stats_block([r.get("server_decode_ms") for r in ok]),
        "ttft_ms": stats_block([r.get("ttft_ms") for r in ok]),
        "tpot_ms_client": stats_block([r.get("client_mean_chunk_gap_ms") for r in ok]),
        "tpot_ms_server": stats_block([r.get("server_mean_tpot_ms") for r in ok]),
    }

    # ---- per-turn ----
    per_turn = {}
    for t in range(6):
        g = [r for r in ok if r.get("turn_index") == t]
        if not g:
            continue
        per_turn[t] = {
            "n": len(g),
            "queue_ms": round(st.mean([r["server_queue_ms"] for r in g]), 1),
            "prefill_ms": round(st.mean([r["server_prefill_ms"] for r in g]), 1),
            "ttft_ms": round(st.mean([r["ttft_ms"] for r in g]), 1),
            "tpot_ms": round(st.mean([r["client_mean_chunk_gap_ms"] for r in g if r.get("client_mean_chunk_gap_ms")]), 1),
            "cached_tokens": round(st.mean([r.get("num_cached_tokens", 0) for r in g]), 0),
            "prompt_tokens": round(st.mean([r.get("num_prompt_tokens", 0) for r in g]), 0),
        }

    # ---- GPU/CPU trong bench ----
    sys_summary = {}
    if args.samples:
        try:
            sm = json.load(open(args.samples))
            def col(k):
                return [s[k] for s in sm if isinstance(s.get(k), (int, float))]
            util = col("gpu_util_pct") or col("utilization_gpu")
            clk = col("sm_clock_mhz") or col("clocks_sm")
            pw = col("power_draw_w")
            tp = col("cpu_throttled_pct_of_window")
            cu = col("cpu_usage_pct_of_window")
            sys_summary = {"n_samples": len(sm),
                           "gpu_util_mean": round(st.mean(util), 1) if util else None,
                           "sm_clock_mean": round(st.mean(clk), 1) if clk else None,
                           "sm_clock_min": min(clk) if clk else None,
                           "power_mean": round(st.mean(pw), 1) if pw else None,
                           "cpu_throttle_max": round(max(tp), 2) if tp else None,
                           "cpu_usage_mean": round(st.mean(cu), 1) if cu else None}
        except Exception as e:  # noqa
            sys_summary = {"error": str(e)}

    summary = {
        "n_total": n_total, "n_failed": n_fail,
        "ERS_client": round(ers_client, 4), "ERS_server_tpot": round(ers_server, 4),
        "score_client_x100": round(ers_client * 100, 2),
        "mean_s_ttft": round(s_ttft, 3), "mean_s_tpot_client": round(s_tpot_c, 3),
        "overall": overall, "per_turn": per_turn, "system": sys_summary,
    }
    json.dump(summary, open(args.out, "w"), indent=2)

    # ---- in bảng ----
    print(f"n={n_total} failed={n_fail}  ERS(client)={ers_client:.4f}  "
          f"score~{ers_client*100:.2f}  [s_ttft={s_ttft:.3f} s_tpot={s_tpot_c:.3f}]  "
          f"ERS(server_tpot)={ers_server:.4f}")
    print(f"{'metric':16s}{'mean':>9}{'median':>9}{'p90':>9}{'p95':>9}{'p99':>9}{'max':>9}")
    for k, b in overall.items():
        if b.get("n"):
            print(f"{k:16s}{b['mean']:>9}{b['median']:>9}{b['p90']:>9}{b['p95']:>9}{b['p99']:>9}{b['max']:>9}")
    print("per-turn:  turn  n   queue   prefill    ttft    tpot   cached  prompt")
    for t, b in per_turn.items():
        print(f"           {t:4d} {b['n']:3d} {b['queue_ms']:8.0f} {b['prefill_ms']:8.0f} "
              f"{b['ttft_ms']:8.0f} {b['tpot_ms']:7.1f} {b['cached_tokens']:8.0f} {b['prompt_tokens']:7.0f}")
    if sys_summary:
        print(f"system: {sys_summary}")


if __name__ == "__main__":
    main()
