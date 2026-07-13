#!/usr/bin/env python3
"""Thong ke so sanh A/B: baseline (FCFS) vs spf, tu cac file
{baseline,spf}_rep*_full.json trong 1 thu muc (mac dinh /root/ab).

Voi moi config, gop tat ca rep, tinh theo TUNG turn (0..5):
  - queue_ms / prefill_ms / ttft_ms / tpot_ms : median cua mean-moi-rep
  - va median tren toan bo request gop lai
In bang so sanh + delta% (spf so voi baseline). Am = spf tot hon.

Dung: python3 compare_ab.py [--dir /root/ab]
"""
import argparse
import glob
import json
import os
from statistics import mean, median

FIELDS = {
    "queue": "server_queue_ms",
    "prefill": "server_prefill_ms",
    "ttft": "ttft_ms",
    "tpot": "server_mean_tpot_ms",
}


def load_reps(d, cfg):
    reps = []
    for path in sorted(glob.glob(os.path.join(d, f"{cfg}_rep*_full.json"))):
        rows = json.load(open(path))
        ok = [r for r in rows if "server_queue_ms" in r and r.get("ttft_ms") is not None]
        reps.append((os.path.basename(path), ok))
    return reps


def per_turn_rep_means(reps, field):
    """Tra ve dict[turn] -> list cac mean-moi-rep."""
    out = {t: [] for t in range(6)}
    for _, rows in reps:
        for t in range(6):
            vals = [r[field] for r in rows if r.get("turn_index") == t]
            if vals:
                out[t].append(mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/root/ab")
    args = ap.parse_args()

    base = load_reps(args.dir, "baseline")
    spf = load_reps(args.dir, "spf")
    if not base or not spf:
        print(f"Thieu du lieu: baseline={len(base)} rep, spf={len(spf)} rep trong {args.dir}")
        return

    print(f"baseline: {len(base)} rep -> {[n for n,_ in base]}")
    print(f"spf     : {len(spf)} rep -> {[n for n,_ in spf]}")
    print(f"(moi rep {len(base[0][1])} req OK)\n")

    for name, field in FIELDS.items():
        b = per_turn_rep_means(base, field)
        s = per_turn_rep_means(spf, field)
        print(f"=== {name.upper()}_MS theo turn (median cua mean-moi-rep) ===")
        print(f"{'turn':>4} {'baseline':>10} {'spf':>10} {'delta%':>8}  {'(base reps)':>22}")
        for t in range(6):
            if not b[t] or not s[t]:
                continue
            mb, ms = median(b[t]), median(s[t])
            d = 100 * (ms - mb) / mb if mb else 0
            reps_str = ",".join(f"{x:.0f}" for x in b[t])
            print(f"{t:>4} {mb:>10.1f} {ms:>10.1f} {d:>7.1f}%  [{reps_str:>20}]")
        # overall: gop moi request tat ca rep
        allb = [r[field] for _, rows in base for r in rows]
        alls = [r[field] for _, rows in spf for r in rows]
        db = 100 * (median(alls) - median(allb)) / median(allb) if allb else 0
        print(f"{'ALL':>4} {median(allb):>10.1f} {median(alls):>10.1f} {db:>7.1f}%  (median toan bo request)")
        print()


if __name__ == "__main__":
    main()
