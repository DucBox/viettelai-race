#!/usr/bin/env python3
"""Internal ERS scorer — reproduce the latency half of the competition score
from an AIPerf run.

The BTC formula (round 1, from docs/VIETTEL AI RACE.pdf §1.3):

    Score = 100 × ERS × f(Δ)

  ERS (Effective Request Score) = mean over ALL requests of a per-request score:
      per-request = 0                      if the request errored / returned 0 tokens
                  = w·s_ttft + (1-w)·s_tpot otherwise
      s = clamp((C - x) / (C - F), 0, 1) ** γ      (x = the measured TTFT or TPOT)

  Round-1 parameters:
      s_ttft : F=100 ms,  C=1500 ms        (wide 15× window)
      s_tpot : F=20 ms,   C=45 ms          (tight 2.25× window)
      γ = 2   (steep: each ms nearer the Floor is worth more)
      w = 0.5 (TTFT and TPOT weighted equally)
      TPOT here = the request's MEAN inter-token latency.

This script deliberately measures ONLY ERS (the latency factor). Accuracy /
f(Δ) is a separate, independent concern — measured with its own harness
(scripts/09_gpqa_accuracy.py or scripts/12_gpqa_lmeval.sh), run and reported
on its own. Combining the two into a final Score is a manual step, not
something this pipeline does automatically.

We read the SAME artifacts as the per-request report: time_to_first_token and
inter_token_latency per request from profile_export.jsonl. So after any
`MODE=replay` run you get ERS immediately — no leaderboard wait.

Usage:
    ./bench/.venv/bin/python scripts/08_ers_score.py
    ./bench/.venv/bin/python scripts/08_ers_score.py artifacts/<run>
Writes score_summary.json into the run dir (machine-readable rollup).
"""
import glob
import json
import os
import sys

# ── Round-1 scoring parameters (edit here if BTC changes them) ───────────────
F_TTFT, C_TTFT = 100.0, 1500.0      # ms
F_TPOT, C_TPOT = 20.0, 45.0         # ms
GAMMA = 2.0
W = 0.5                             # weight on s_ttft (1-W on s_tpot)
USERS_DEFAULT = 20


def parse_args(argv):
    run_dir, users = None, USERS_DEFAULT
    args = argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--users":
            users = int(args[i + 1]); i += 2; continue
        if a.startswith("--users="):
            users = int(a.split("=", 1)[1]); i += 1; continue
        run_dir = a; i += 1
    return run_dir, users


def find_run_dir(explicit):
    if explicit:
        return explicit
    cands = [d for d in glob.glob("artifacts/*") if os.path.isdir(d)]
    if not cands:
        sys.exit("No artifacts/* directory found. Run bench/run_aiperf_baseline.sh first.")
    return max(cands, key=os.path.getmtime)


def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def component(x, F, C):
    """clamp((C - x) / (C - F), 0, 1) ** GAMMA — 1.0 at/below Floor, 0 at/above Ceiling."""
    if x is None:
        return 0.0
    s = (C - x) / (C - F)
    s = 0.0 if s < 0 else (1.0 if s > 1 else s)
    return s ** GAMMA


def main():
    explicit, users = parse_args(sys.argv)
    run_dir = find_run_dir(explicit)
    requests = load_jsonl(os.path.join(run_dir, "profile_export.jsonl"))
    if not requests:
        sys.exit(f"No profile_export.jsonl records found in {run_dir}")

    rows = []
    for r in requests:
        meta = r.get("metadata", {})
        metrics = r.get("metrics") or {}
        sn = meta.get("session_num")

        def mv(name):
            e = metrics.get(name)
            return e.get("value") if e else None

        ttft = mv("time_to_first_token")
        tpot = mv("inter_token_latency")
        out_tok = mv("output_sequence_length") or mv("output_token_count")
        failed = bool(r.get("error")) or not out_tok  # error / timeout / 0 tokens → 0 pts

        if failed:
            rows.append({"sn": sn, "turn": (sn // users) if sn is not None else None,
                         "ttft": ttft, "tpot": tpot, "s_ttft": 0.0, "s_tpot": 0.0,
                         "score": 0.0, "failed": True})
            continue

        s_ttft = component(ttft, F_TTFT, C_TTFT)
        s_tpot = component(tpot, F_TPOT, C_TPOT)
        score = W * s_ttft + (1 - W) * s_tpot
        rows.append({"sn": sn, "turn": (sn // users) if sn is not None else None,
                     "ttft": ttft, "tpot": tpot, "s_ttft": s_ttft, "s_tpot": s_tpot,
                     "score": score, "failed": False})

    n = len(rows)
    n_failed = sum(1 for x in rows if x["failed"])
    ers = sum(x["score"] for x in rows) / n if n else 0.0
    mean_s_ttft = sum(x["s_ttft"] for x in rows) / n if n else 0.0
    mean_s_tpot = sum(x["s_tpot"] for x in rows) / n if n else 0.0

    # Machine-readable rollup next to the run — 11_multi_bench.sh reads this.
    with open(os.path.join(run_dir, "score_summary.json"), "w") as fh:
        json.dump({
            "run_dir": run_dir, "requests": n, "failed": n_failed,
            "ers": ers, "mean_s_ttft": mean_s_ttft, "mean_s_tpot": mean_s_tpot,
        }, fh, indent=2)

    print(f"run: {run_dir}")
    print(f"params (round 1): TTFT F={F_TTFT:.0f}/C={C_TTFT:.0f}ms  "
          f"TPOT F={F_TPOT:.0f}/C={C_TPOT:.0f}ms  gamma={GAMMA:.0f}  w={W}")
    print()
    print(f"requests: {n}   failed/empty (0 pts): {n_failed}")
    print(f"mean s_ttft = {mean_s_ttft:.3f}    mean s_tpot = {mean_s_tpot:.3f}")
    print("-" * 52)
    print(f"ERS = {ers:.4f}")

    # Per-turn breakdown — shows exactly where points are won/lost (cold-start
    # turn 0 vs cached later turns). Only meaningful for the trace replay.
    turns = sorted({x["turn"] for x in rows if x["turn"] is not None})
    if turns:
        print()
        print("per-turn (where the points come from):")
        hdr = (f"{'turn':>4} {'n':>3} {'ttft(ms)':>9} {'s_ttft':>7} "
               f"{'tpot(ms)':>9} {'s_tpot':>7} {'score':>6} {'fail':>4}")
        print(hdr)
        print("-" * len(hdr))
        for t in turns:
            grp = [x for x in rows if x["turn"] == t]
            g = len(grp)
            ok = [x for x in grp if not x["failed"]]
            avg = lambda key, src: (sum(x[key] for x in src) / len(src)) if src else 0.0
            print(f"{t:>4} {g:>3} "
                  f"{avg('ttft', ok):>9.1f} {avg('s_ttft', grp):>7.3f} "
                  f"{avg('tpot', ok):>9.1f} {avg('s_tpot', grp):>7.3f} "
                  f"{avg('score', grp):>6.3f} {sum(1 for x in grp if x['failed']):>4}")

    print()
    print("Note: ERS averages over ALL requests (failures count as 0). This is the")
    print("LATENCY factor only — accuracy/f(Δ) is measured separately (scripts/09 or")
    print("scripts/12) and is not combined into a Score here.")


if __name__ == "__main__":
    main()
