#!/usr/bin/env python3
"""Diagnose TTFT variance across REPEATS — per experiment, then a combined
comparison across all experiments in a multibench run. Answers "is the noise
concentrated at turn 0 (cold-start artifact, expected) or spread across all
turns (something else going on)?" for EACH config, so you can see if some
configs are noisier than others too.

Reads every rep's per_request_report.csv under <multibench_dir>/<exp>/rep*/
run/, groups by (user, turn), and computes stddev of TTFT across reps for
each (user, turn) — then rolls that up into a per-turn summary per exp, and
finally a combined table across all exps.

Usage:
    # all experiments found under the multibench run
    python3 scripts/check_ttft_variance.py artifacts/multibench/<ts>

    # just specific ones
    python3 scripts/check_ttft_variance.py artifacts/multibench/<ts> baseline fp8

    # full per-(user,turn) breakdown table for each exp, not just the summary
    python3 scripts/check_ttft_variance.py artifacts/multibench/<ts> --detail
"""
import argparse
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


def discover_exps(multibench_dir):
    pattern = os.path.join(multibench_dir, "*", "rep*", "run", "per_request_report.csv")
    names = set()
    for path in glob.glob(pattern):
        rel = os.path.relpath(path, multibench_dir)
        names.add(rel.split(os.sep)[0])
    return sorted(names)


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return mean, var ** 0.5, min(vals), max(vals)


def analyze_exp(multibench_dir, exp_name, detail=False):
    reps = load_reps(multibench_dir, exp_name)
    if not reps:
        print(f"!! {exp_name}: no reps found (need REPEATS>1 — rep*/run/per_request_report.csv), skipping")
        print()
        return None
    if len(reps) < 2:
        print(f"!! {exp_name}: only {len(reps)} rep found — need 2+ reps to measure variance, skipping")
        print()
        return None

    by_key = {}
    for rep_num, rows in reps.items():
        for row in rows:
            key = (row.get("user"), row.get("turn"))
            try:
                ttft = float(row["ttft"])
            except (ValueError, KeyError, TypeError):
                continue
            by_key.setdefault(key, []).append(ttft)

    rows_out = []
    for (user, turn), vals in by_key.items():
        if user is None or turn is None:
            continue
        mean, std, mn, mx = stats(vals)
        rows_out.append((int(turn), int(user), len(vals), mean, std, mn, mx))
    rows_out.sort()

    print(f"### {exp_name}  ({len(reps)} reps: {sorted(reps, key=int)})")
    if detail:
        hdr = f"{'user':>4} {'turn':>4} {'n':>3} {'mean(ms)':>9} {'std(ms)':>8} {'min(ms)':>8} {'max(ms)':>8}"
        print(hdr)
        for turn, user, n, mean, std, mn, mx in rows_out:
            print(f"{user:>4} {turn:>4} {n:>3} {mean:>9.1f} {std:>8.1f} {mn:>8.1f} {mx:>8.1f}")

    by_turn = {}
    for turn, user, n, mean, std, mn, mx in rows_out:
        by_turn.setdefault(turn, []).append(std)
    for turn in sorted(by_turn):
        stds = by_turn[turn]
        print(f"   turn {turn}: mean std = {sum(stds)/len(stds):.1f} ms  (across {len(stds)} users)")

    turn0_std = sum(by_turn.get(0, [0])) / max(len(by_turn.get(0, [1])), 1)
    other_stds = [s for t, ss in by_turn.items() if t != 0 for s in ss]
    other_std = sum(other_stds) / len(other_stds) if other_stds else 0.0
    ratio = (turn0_std / other_std) if other_std > 0 else float("inf")
    is_cold_start = other_std > 0 and turn0_std > 3 * other_std
    verdict = "cold-start (turn0 only)" if is_cold_start else "spread / dig further"

    print(f"   -> turn0 std={turn0_std:.0f}ms  other-turns std={other_std:.0f}ms  "
          f"ratio={ratio:.1f}x  VERDICT: {verdict}")
    print()
    return {"exp": exp_name, "n_reps": len(reps), "turn0_std": turn0_std,
            "other_std": other_std, "ratio": ratio, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("multibench_dir")
    ap.add_argument("exps", nargs="*", help="exp names to analyze (default: every exp found)")
    ap.add_argument("--detail", action="store_true", help="print full per-(user,turn) table for each exp")
    args = ap.parse_args()

    exp_names = args.exps or discover_exps(args.multibench_dir)
    if not exp_names:
        sys.exit(f"no experiments (with REPEATS>1) found under {args.multibench_dir}")

    print(f">> analyzing {len(exp_names)} experiment(s): {exp_names}")
    print()

    results = [r for r in (analyze_exp(args.multibench_dir, exp, detail=args.detail) for exp in exp_names) if r]

    if not results:
        sys.exit("no experiment had usable data (need REPEATS>1)")

    if len(results) > 1:
        print("=" * 78)
        print("COMBINED COMPARISON (sorted noisiest-turn0-first)")
        print("=" * 78)
        hdr = f"{'exp':<22} {'reps':>4} {'turn0_std':>10} {'other_std':>10} {'ratio':>7}  verdict"
        print(hdr)
        print("-" * len(hdr))
        for r in sorted(results, key=lambda r: (r["ratio"] == float("inf"), -r["ratio"] if r["ratio"] != float("inf") else 0)):
            print(f"{r['exp']:<22} {r['n_reps']:>4} {r['turn0_std']:>9.0f}ms {r['other_std']:>9.0f}ms "
                  f"{r['ratio']:>6.1f}x  {r['verdict']}")
        print()
        n_cold = sum(1 for r in results if "cold-start" in r["verdict"])
        print(f">> {n_cold}/{len(results)} experiments show the cold-start (turn0-only) pattern.")
        if n_cold < len(results):
            print("   The rest have noise spread beyond turn 0 — worth a closer look at those")
            print("   specifically (thermal state, background load, that config's own behavior).")


if __name__ == "__main__":
    main()
