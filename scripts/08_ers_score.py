#!/usr/bin/env python3
"""Internal ERS scorer — reproduce the competition score from an AIPerf run.

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

  f(Δ) — accuracy gate, scored separately on 100 GPQA-Diamond questions:
      Δ = 0.40 - accuracy;  f=1 if Δ≤0.10, 0 if Δ≥0.16, linear in between.
  This script assumes f(Δ)=1 unless you pass --accuracy (the accuracy gate is a
  separate GPQA harness, not something AIPerf measures).

We read the SAME artifacts as the per-request report: time_to_first_token and
inter_token_latency per request from profile_export.jsonl. So after any
`MODE=replay` run you get the projected Score immediately — no leaderboard wait.

Usage:
    ./bench/.venv/bin/python scripts/08_ers_score.py
    ./bench/.venv/bin/python scripts/08_ers_score.py artifacts/<run>
    ./bench/.venv/bin/python scripts/08_ers_score.py --accuracy 0.36
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
BASELINE_ACC = 0.40                # BF16 reference accuracy
GATE_FULL, GATE_ZERO = 0.10, 0.16  # Δ thresholds for f(Δ)
USERS_DEFAULT = 20


def parse_args(argv):
    run_dir, accuracy, users = None, None, USERS_DEFAULT
    args = argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--accuracy":
            accuracy = float(args[i + 1]); i += 2; continue
        if a.startswith("--accuracy="):
            accuracy = float(a.split("=", 1)[1]); i += 1; continue
        if a == "--users":
            users = int(args[i + 1]); i += 2; continue
        if a.startswith("--users="):
            users = int(a.split("=", 1)[1]); i += 1; continue
        run_dir = a; i += 1
    return run_dir, accuracy, users


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


def accuracy_gate(accuracy):
    if accuracy is None:
        return 1.0, True  # assumed pass
    delta = BASELINE_ACC - accuracy
    if delta <= GATE_FULL:
        return 1.0, False
    if delta >= GATE_ZERO:
        return 0.0, False
    return (GATE_ZERO - delta) / (GATE_ZERO - GATE_FULL), False


def main():
    explicit, accuracy, users = parse_args(sys.argv)
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
    f_delta, assumed = accuracy_gate(accuracy)
    final = 100.0 * ers * f_delta

    print(f"run: {run_dir}")
    print(f"params (round 1): TTFT F={F_TTFT:.0f}/C={C_TTFT:.0f}ms  "
          f"TPOT F={F_TPOT:.0f}/C={C_TPOT:.0f}ms  gamma={GAMMA:.0f}  w={W}")
    print()
    print(f"requests: {n}   failed/empty (0 pts): {n_failed}")
    print(f"mean s_ttft = {mean_s_ttft:.3f}    mean s_tpot = {mean_s_tpot:.3f}")
    print(f"ERS = {ers:.4f}")
    if assumed:
        print(f"f(Δ) = {f_delta:.3f}   (ASSUMED — pass --accuracy <acc> to apply the GPQA gate)")
    else:
        delta = BASELINE_ACC - accuracy
        print(f"f(Δ) = {f_delta:.3f}   (accuracy={accuracy:.3f}, Δ={delta:.3f})")
    print("-" * 52)
    print(f"SCORE = 100 × ERS × f(Δ) = {final:.2f}")

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
    print("Note: ERS averages over ALL requests (failures count as 0). f(Δ) is the")
    print("independent accuracy gate (100 GPQA-Diamond Qs) — measure it separately and")
    print("pass --accuracy; a real Score needs both cheap latency AND accuracy ≥ ~0.30.")


if __name__ == "__main__":
    main()
