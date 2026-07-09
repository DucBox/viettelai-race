#!/usr/bin/env python3
"""Diagnose TTFT variance across REPEATS for one multi-bench experiment —
answers "is the noise concentrated at turn 0 (cold-start artifact, expected)
or spread across all turns (something else going on)?"

Reads every rep's per_request_report.csv under <multibench_dir>/<exp_name>/
rep*/run/, groups by (user, turn), and prints stddev/range of TTFT across
reps for each (user, turn) — plus a per-turn summary so you can see the
pattern at a glance.

Usage:
    python3 scripts/check_ttft_variance.py artifacts/multibench/<ts> baseline
"""
import csv
import glob
import os
import sys


def load_reps(multibench_dir, exp_name):
    reps = {}
    pattern = os.path.join(multibench_dir, exp_name, "rep*", "run", "per_request_report.csv")
    for path in sorted(glob.glob(pattern)):
        rep_num = path.split(os.sep)[-3].replace("rep", "")
        with open(path, newline="") as f:
            reps[rep_num] = list(csv.DictReader(f))
    return reps


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: check_ttft_variance.py <multibench_dir> <exp_name>")
    multibench_dir, exp_name = sys.argv[1], sys.argv[2]
    reps = load_reps(multibench_dir, exp_name)
    if not reps:
        sys.exit(f"no reps found under {multibench_dir}/{exp_name}/rep*/run/per_request_report.csv")
    print(f">> {len(reps)} reps found: {sorted(reps, key=int)}")

    # group ttft by (user, turn) across all reps
    by_key = {}
    for rep_num, rows in reps.items():
        for row in rows:
            key = (row["user"], row["turn"])
            try:
                ttft = float(row["ttft"])
            except (ValueError, KeyError):
                continue
            by_key.setdefault(key, []).append((rep_num, ttft))

    def stats(vals):
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        return mean, var ** 0.5, min(vals), max(vals)

    print()
    print(f"{'user':>4} {'turn':>4} {'n':>3} {'mean(ms)':>9} {'std(ms)':>8} {'min(ms)':>8} {'max(ms)':>8} {'max/min':>8}")
    print("-" * 60)
    rows_out = []
    for (user, turn), pairs in sorted(by_key.items(), key=lambda kv: (int(kv[0][1]), int(kv[0][0]))):
        vals = [v for _, v in pairs]
        mean, std, mn, mx = stats(vals)
        ratio = mx / mn if mn > 0 else float("inf")
        rows_out.append((int(turn), int(user), len(vals), mean, std, mn, mx, ratio))
    for turn, user, n, mean, std, mn, mx, ratio in rows_out:
        print(f"{user:>4} {turn:>4} {n:>3} {mean:>9.1f} {std:>8.1f} {mn:>8.1f} {mx:>8.1f} {ratio:>7.2f}x")

    # per-turn aggregate: is variance concentrated at turn 0?
    print()
    print(">> per-turn mean stddev (average noise level at that turn, across all users):")
    by_turn = {}
    for turn, user, n, mean, std, mn, mx, ratio in rows_out:
        by_turn.setdefault(turn, []).append(std)
    for turn in sorted(by_turn):
        stds = by_turn[turn]
        print(f"   turn {turn}: mean std = {sum(stds)/len(stds):.1f} ms  (across {len(stds)} users)")

    print()
    turn0_std = sum(by_turn.get(0, [0])) / max(len(by_turn.get(0, [1])), 1)
    other_stds = [s for t, ss in by_turn.items() if t != 0 for s in ss]
    other_std = sum(other_stds) / len(other_stds) if other_stds else 0.0
    if other_std > 0 and turn0_std > 3 * other_std:
        print(f"VERDICT: turn 0 noise ({turn0_std:.0f}ms) is {turn0_std/other_std:.1f}x the other turns'")
        print(f"({other_std:.0f}ms) — consistent with cold-start GPU ramp-up / JIT compile spikes on")
        print("the very first request after each restart. Expected, not a bug — trust turns 1+")
        print("for tight A/B comparisons, or exclude turn 0 from strict noise-floor arguments.")
    else:
        print(f"VERDICT: turn0 std ({turn0_std:.0f}ms) is NOT dramatically higher than other turns")
        print(f"({other_std:.0f}ms) — noise is spread across the whole run, not just cold-start.")
        print("Worth digging further (GPU thermal state, background load, network jitter, ...).")


if __name__ == "__main__":
    main()
