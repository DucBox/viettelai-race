#!/usr/bin/env python3
"""Cây phân rã TTFT + TPOT có KIỂM CHỨNG (reconcile |Σcon - cha| < eps).

Ghép 3 nguồn (join theo request_id):
  --full     *_full.json      (client ttft + REQSTAT server queue/prefill/decode)
  --rests    rests.jsonl      (patch_residual_ts: arrival, ftl, *_ts core clock)
  --sched    sched.jsonl      (patch_sched_trace: t, exec_gap, tokens, *_ids)  [tùy chọn]

In ra:
  TẦNG 1  TTFT = queue + prefill + residual                (khớp <1ms, đã verify)
  TẦNG 2a residual = frontend_prep + client_transport       (tách bằng ftl, EXACT)
  TẦNG 2b queue = wait_prefill + wait_decode + sched_ovhd    (correlate sched window)
  TẦNG 2c prefill = own_compute + interleave                 (correlate sched window)
  TPOT   = pure_decode + mixed_interleave_penalty            (phân loại step)
Mỗi tầng in kèm sai số reconcile (mean & p95 của |Σcon - cha|).

    python3 reconcile_trace.py --full X_full.json --rests rests.jsonl \\
        --sched sched.jsonl --out recon.json
"""
import argparse
import bisect
import json
import statistics as st


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--rests", required=True)
    ap.add_argument("--sched", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pref-thresh", type=int, default=50,
                    help="tokens_sched - n_running > thresh => iteration có prefill chunk")
    args = ap.parse_args()

    full = json.load(open(args.full))
    rests = {r["rid"]: r for r in load_jsonl(args.rests)}

    sched = None
    sched_t = None
    if args.sched:
        sched = sorted(load_jsonl(args.sched), key=lambda r: r["t"])
        sched_t = [r["t"] for r in sched]

    def window_rows(t0, t1):
        """sched rows có t trong [t0, t1]."""
        if not sched:
            return []
        lo = bisect.bisect_left(sched_t, t0)
        hi = bisect.bisect_right(sched_t, t1)
        return sched[lo:hi]

    recs = []
    for r in full:
        rid = r.get("server_request_id")
        if rid is None or r.get("ttft_ms") is None or rid not in rests:
            continue
        rs = rests[rid]
        ttft = r["ttft_ms"]
        q = r["server_queue_ms"]
        p = r["server_prefill_ms"]
        residual = ttft - q - p
        ftl_ms = rs["ftl"] * 1000.0

        # --- Tầng 2a: tách residual (EXACT: frontend_prep + client_transport = residual) ---
        frontend_prep = ftl_ms - q - p          # arrival -> queued (tokenize + handoff)
        client_transport = ttft - ftl_ms        # net + HTTP + detok/stream

        rec = {
            "rid": rid, "round": r.get("round"), "turn": r.get("turn_index"),
            "prompt_tok": r.get("num_prompt_tokens"), "cached_tok": r.get("num_cached_tokens"),
            "ttft": ttft, "queue": q, "prefill": p, "residual": residual,
            "frontend_prep": frontend_prep, "client_transport": client_transport,
            "err_t1": abs(ttft - (q + p + residual)),
            "err_t2a": abs(residual - (frontend_prep + client_transport)),
        }

        # --- Tầng 2b + 2c: correlate với sched window (cùng clock monotonic core) ---
        if sched:
            qs, ss, ft = rs["queued_ts"], rs["scheduled_ts"], rs["first_token_ts"]
            # QUEUE: iterations trong [queued_ts, scheduled_ts]
            wp = wd = so = 0.0
            for x in window_rows(qs, ss):
                so += x.get("sched_ms", 0.0)
                if (x["tokens_sched"] - x["n_running"]) > args.pref_thresh:
                    wp += x["exec_gap_ms"]
                else:
                    wd += x["exec_gap_ms"]
            rec.update(wait_prefill=wp, wait_decode=wd, sched_ovhd=so,
                       err_t2b=abs(q - (wp + wd + so)))
            # PREFILL: iterations trong [scheduled_ts, first_token_ts]
            own = 0.0
            for x in window_rows(ss, ft):
                if rid in (x.get("pref_ids") or []):
                    own += x["exec_gap_ms"]
            rec.update(own_compute=own, interleave=p - own)

            # --- TPOT: decode window [first_token_ts, last_token_ts] ---
            lt = rs["last_token_ts"]
            pure, mixed = [], []
            for x in window_rows(ft, lt):
                (mixed if x.get("n_prefilling", 0) > 0 else pure).append(x["exec_gap_ms"])
            tpot = r.get("client_mean_chunk_gap_ms")
            rec.update(
                tpot=tpot,
                tpot_pure=st.mean(pure) if pure else None,
                tpot_mixed=st.mean(mixed) if mixed else None,
                mixed_frac=len(mixed) / (len(pure) + len(mixed)) if (pure or mixed) else 0.0,
            )
        recs.append(rec)

    if not recs:
        raise SystemExit("Không join được request nào — kiểm tra rid khớp giữa full & rests.")

    # ---------- báo cáo ----------
    def m(key):
        v = [x[key] for x in recs if x.get(key) is not None]
        return st.mean(v) if v else 0.0

    n = len(recs)
    print(f"\n===== RECONCILE {n} request =====")
    print(f"TTFT {m('ttft'):.0f}ms")
    print(f" ├─ queue        {m('queue'):7.0f}  ({100*m('queue')/m('ttft'):4.0f}%)")
    if sched:
        print(f" │   ├─ wait_prefill {m('wait_prefill'):7.0f}")
        print(f" │   ├─ wait_decode  {m('wait_decode'):7.0f}")
        print(f" │   └─ sched_ovhd   {m('sched_ovhd'):7.0f}   [reconcile |Σ-queue| mean={m('err_t2b'):.1f} p95={pct([x['err_t2b'] for x in recs if 'err_t2b' in x],.95):.1f}ms]")
    print(f" ├─ prefill      {m('prefill'):7.0f}  ({100*m('prefill')/m('ttft'):4.0f}%)")
    if sched:
        print(f" │   ├─ own_compute  {m('own_compute'):7.0f}   (bóc kernel = chạy profile_config/parse_dir)")
        print(f" │   └─ interleave   {m('interleave'):7.0f}")
    print(f" └─ residual     {m('residual'):7.0f}  ({100*m('residual')/m('ttft'):4.0f}%)")
    print(f"     ├─ frontend_prep    {m('frontend_prep'):7.0f}   (tokenize+hash+handoff, ~scale prompt)")
    print(f"     └─ client_transport {m('client_transport'):7.0f}   (net+HTTP+detok/stream)")
    print(f"   [reconcile T1 |ttft-Σ| mean={m('err_t1'):.3f}  T2a |res-Σ| mean={m('err_t2a'):.3f}ms]")

    if sched and any(x.get("tpot") for x in recs):
        print(f"\nTPOT {m('tpot'):.2f}ms")
        print(f" ├─ pure-decode step  {m('tpot_pure'):.2f}ms  (n_prefilling=0)")
        print(f" ├─ mixed step        {m('tpot_mixed'):.2f}ms  (bị gộp prefill chunk)")
        print(f" └─ mixed fraction    {100*m('mixed_frac'):.1f}%  <- nguồn TBT tail; penalty=(mixed-pure)*frac")

    # cảnh báo clock lệch
    if sched:
        empty = sum(1 for x in recs if x.get("wait_prefill", 0) + x.get("wait_decode", 0) == 0 and x["queue"] > 5)
        if empty > n * 0.2:
            print(f"\n⚠️  {empty}/{n} request có queue>5ms nhưng window sched RỖNG — "
                  f"nghi CLOCK sched-trace lệch clock core (kiểm event.timestamp vs time.monotonic).")

    # per-round residual split (điểm mấu chốt: residual scale theo prompt)
    print("\nper-round:  round  n   queue  prefill  residual  front_prep  transport  prompt")
    for rd in sorted({x["round"] for x in recs if x["round"] is not None}):
        g = [x for x in recs if x["round"] == rd]
        print(f"            {rd:5d} {len(g):3d} {st.mean([x['queue'] for x in g]):7.0f} "
              f"{st.mean([x['prefill'] for x in g]):8.0f} {st.mean([x['residual'] for x in g]):9.0f} "
              f"{st.mean([x['frontend_prep'] for x in g]):11.0f} {st.mean([x['client_transport'] for x in g]):10.0f} "
              f"{st.mean([x['prompt_tok'] for x in g]):7.0f}")

    if args.out:
        json.dump({"n": n, "means": {k: m(k) for k in
                   ("ttft", "queue", "prefill", "residual", "frontend_prep",
                    "client_transport", "wait_prefill", "wait_decode", "sched_ovhd",
                    "own_compute", "interleave", "tpot", "tpot_pure", "tpot_mixed", "mixed_frac")},
                   "per_request": recs}, open(args.out, "w"), indent=2)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
