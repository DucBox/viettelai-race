#!/usr/bin/env bash
# Sweep several serve configurations through the LATENCY-ONLY e2e benchmark
# and produce ONE comparison table. Each EXPERIMENT row is a distinct vLLM
# serve config; the script restarts the server cold per row (10_bench_e2e.sh
# step 0) so every number is fair, then aggregates ERS across all rows.
# Accuracy/f(Δ) is NOT part of this pipeline — measure it separately
# (scripts/09_gpqa_accuracy.py or scripts/12_gpqa_lmeval.sh) if you need it.
#
#   ./scripts/11_multi_bench.sh
#
# EDIT THE EXPERIMENTS BLOCK BELOW — one row per config, tab-separated:
#     <name> <TAB> <EXTRA_VLLM_ARGS> <TAB> [extra env: KEY=VAL KEY=VAL ...]
#   - field 1  name              : short label for the row (folder + table row)
#   - field 2  EXTRA_VLLM_ARGS   : vLLM `serve` flags for this config (may be empty)
#   - field 3  extra env         : optional serve overrides (MODEL_DIR=, GPU_MEM_UTIL=, ...)
#   Lines starting with # and blank lines are ignored. Add/remove rows freely.
#
# NOTE: fields are separated by a TAB, not spaces (EXTRA_VLLM_ARGS itself has
# spaces). In most editors a literal tab works; here we spell rows with $'\t'.
#
# Each row's outputs land in artifacts/multibench/<ts>/<name>/ (or .../<name>/rep<N>/
# when REPEATS>1 — see below):
#   console.log            full terminal output (the 2 AIPerf tables + 07 table live here)
#   run/                   the AIPerf run dir: profile_export.jsonl, server_metrics_export.*,
#                          per_request_report.csv (07), score_summary.json (08, ERS only)
#   exp_meta.json          {name, extra_vllm_args, extra_env}
# Plus a top-level summary.csv (one row per rep) + printed comparison table.
#
# WHY REPEAT: a single cold-restart run is NOISY (GPU boost-clock ramp-up, OS/
# scheduling jitter, ~333ms server-metrics scrape granularity) — comparing two
# configs from ONE run each cannot tell a real effect from noise. Set REPEATS=3
# (or more) to run every row that many times; the summary then reports
# mean±stddev per config so you can see whether a difference is bigger than the
# noise floor. Repeats are INTERLEAVED (rep1: row1,row2,...  rep2: row1,row2,...)
# rather than blocked (row1×3, row2×3, ...) so any drift over the session
# (thermal, background load) hits every config equally instead of biasing
# whichever one happened to run first or last.
#
# Env: MODE (default replay), REPEATS (default 1), URL, SERVED_MODEL_NAME —
#      forwarded to every row so the comparison is apples-to-apples.
set -euo pipefail
cd "$(dirname "$0")/.."

# ── EXPERIMENTS ──────────────────────────────────────────────────────────────
# name  <TAB>  EXTRA_VLLM_ARGS  <TAB>  [extra env]
# Default rows use flags known to exist on vLLM v0.22.1. Add/remove freely.
# Every flag here is just appended to `vllm serve` verbatim by serve_up.sh, so
# anything `vllm serve --help` accepts is a valid row.
EXPERIMENTS=(
  "$(printf 'baseline\t\t')"                                        # untouched BF16 reference
  "$(printf 'fp8\t--quantization fp8\t')"                           # on-the-fly W8A8 FP8 weights
  "$(printf 'fp8_kv\t--kv-cache-dtype fp8\t')"                      # FP8 KV cache (weights stay BF16)
  "$(printf 'fp8_both\t--quantization fp8 --kv-cache-dtype fp8\t')" # FP8 weights + FP8 KV cache
  "$(printf 'chunked_prefill\t--enable-chunked-prefill\t')"         # split long prefills into chunks
  # "$(printf 'no_prefix\t--no-enable-prefix-caching\t')"           # measure without prefix cache
  # "$(printf 'more_seqs\t--max-num-seqs 64\tGPU_MEM_UTIL=0.95')"   # bigger batch + example extra env
)
# ─────────────────────────────────────────────────────────────────────────────

MODE="${MODE:-replay}"

# Hard-check serve/.env's ACTUAL EXTRA_VLLM_ARGS value (not just line presence —
# .env.example ships an empty `EXTRA_VLLM_ARGS=` placeholder, which must NOT
# trigger this). Sourced in an isolated subshell so it can't be polluted by (or
# pollute) this script's own env. serve_up.sh sources .env AFTER inheriting each
# row's EXTRA_VLLM_ARGS, and a plain assignment always wins — so a non-empty
# value here silently makes every row in the sweep serve the SAME config.
_env_extra_vllm_args="$(
  unset EXTRA_VLLM_ARGS
  [[ -f serve/.env ]] && { set -a; source serve/.env >/dev/null 2>&1; set +a; }
  printf '%s' "${EXTRA_VLLM_ARGS:-}"
)"
# trim surrounding whitespace so "  " also counts as empty
read -r _env_extra_vllm_args <<<"$_env_extra_vllm_args"
if [[ -n "$_env_extra_vllm_args" ]]; then
  echo "!! serve/.env sets EXTRA_VLLM_ARGS=\"$_env_extra_vllm_args\"" >&2
  echo "   This OVERRIDES every row's flags below (serve_up.sh sources .env AFTER" >&2
  echo "   inheriting each row's EXTRA_VLLM_ARGS; a plain assignment always wins," >&2
  echo "   it never merges). Every experiment would silently serve the SAME" >&2
  echo "   config — the sweep would be meaningless." >&2
  echo "   Fix: clear it in serve/.env (EXTRA_VLLM_ARGS=) and re-run." >&2
  exit 1
fi
echo ">> serve/.env EXTRA_VLLM_ARGS check: empty/absent — OK, per-row flags will apply"

REPEATS="${REPEATS:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="artifacts/multibench/$TS"
mkdir -p "$OUT"
echo ">> multi-bench $TS  ($MODE)  ->  $OUT"
echo ">> ${#EXPERIMENTS[@]} experiments  ×  REPEATS=$REPEATS"

n=0
total=$(( ${#EXPERIMENTS[@]} * REPEATS ))
# Outer loop = rep, inner loop = row -> INTERLEAVED order (rep1: all rows, then
# rep2: all rows, ...) so session-wide drift (thermal, background load) can't
# bias one config vs another the way a blocked order (row1×N, row2×N, ...) would.
for rep in $(seq 1 "$REPEATS"); do
for row in "${EXPERIMENTS[@]}"; do
  # split on tab into name / args / extra-env. NOT `IFS=$'\t' read` — bash
  # classifies tab as "IFS whitespace" and COALESCES consecutive delimiters
  # (like it does for spaces), silently dropping an empty middle field: a row
  # with EXTRA_VLLM_ARGS empty and extra-env set (e.g. `name\t\tGPU_MEM_UTIL=0.9`)
  # would shift the extra-env value into args instead, which then gets appended
  # to `vllm serve` as a bogus token -> "unrecognized arguments: GPU_MEM_UTIL=0.9".
  # awk with a single-char FS does an exact split, no coalescing.
  name="$(awk -F'\t' '{print $1}' <<<"$row")"
  args="$(awk -F'\t' '{print $2}' <<<"$row")"
  extra="$(awk -F'\t' '{print $3}' <<<"$row")"
  name="${name## }"; name="${name%% }"
  [[ -z "$name" || "$name" == \#* ]] && continue
  n=$((n + 1))
  if [[ "$REPEATS" -gt 1 ]]; then
    EXPDIR="$OUT/$name/rep$rep"
  else
    EXPDIR="$OUT/$name"
  fi
  mkdir -p "$EXPDIR"

  printf '\n\033[1m########## EXP %d/%d: %s (rep %d/%d) ##########\033[0m\n' \
    "$n" "$total" "$name" "$rep" "$REPEATS"
  echo ">> EXTRA_VLLM_ARGS = ${args:-<none>}"
  echo ">> extra env       = ${extra:-<none>}"

  # scope env to this e2e invocation; MODE forwarded from our env
  EXP_ENV=(EXTRA_VLLM_ARGS="$args" MODE="$MODE")
  # shellcheck disable=SC2206
  [[ -n "${extra// }" ]] && EXP_ENV+=($extra)

  # exp_meta.json (record the exact config for the aggregate)
  python3 - "$EXPDIR/exp_meta.json" "$name" "$args" "$extra" "$rep" <<'PY'
import json, sys
json.dump({"name": sys.argv[2], "extra_vllm_args": sys.argv[3],
           "extra_env": sys.argv[4], "rep": int(sys.argv[5])}, open(sys.argv[1], "w"), indent=2)
PY

  # run the full e2e (cold restart -> load -> 07 -> ERS); save all output.
  set +e
  env "${EXP_ENV[@]}" bash scripts/10_bench_e2e.sh 2>&1 | tee "$EXPDIR/console.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "!! EXP '$name' rep $rep failed (rc=$rc) — see $EXPDIR/console.log; continuing."
    echo "$rc" > "$EXPDIR/FAILED"
    continue
  fi

  # move this run's artifact dir under the exp folder (also keeps artifacts/ clean
  # so the next exp's newest-run detection can't pick a stale dir).
  RUN_DIR=""
  for d in $(ls -dt artifacts/*/ 2>/dev/null); do
    case "$d" in */multibench/*) continue;; esac
    [[ -f "${d}profile_export.jsonl" ]] && { RUN_DIR="${d%/}"; break; }
  done
  if [[ -n "$RUN_DIR" ]]; then
    rm -rf "$EXPDIR/run"
    mv "$RUN_DIR" "$EXPDIR/run"
    echo ">> saved run -> $EXPDIR/run"
  else
    echo "!! could not locate this exp's run dir (no profile_export.jsonl found)"
  fi
done
done

# ── aggregate ────────────────────────────────────────────────────────────────
# Reads BOTH layouts: flat <name>/exp_meta.json (REPEATS=1) and nested
# <name>/rep<N>/exp_meta.json (REPEATS>1) — finds every exp_meta.json under OUT.
printf '\n\033[1m========== MULTI-BENCH SUMMARY ==========\033[0m\n'
python3 - "$OUT" <<'PY'
import csv, glob, json, math, os, sys

out = sys.argv[1]
rows = []
for meta_path in sorted(glob.glob(os.path.join(out, "**", "exp_meta.json"), recursive=True)):
    exp = os.path.dirname(meta_path)
    meta = json.load(open(meta_path))
    r = {"name": meta["name"], "flags": meta["extra_vllm_args"] or "-",
         "rep": meta.get("rep", 1), "status": "ok",
         "ers": None, "s_ttft": None, "s_tpot": None, "kv_max": None}
    if os.path.exists(os.path.join(exp, "FAILED")):
        r["status"] = "FAILED"
    sc_p = os.path.join(exp, "run", "score_summary.json")
    smx_p = os.path.join(exp, "run", "server_metrics_export.jsonl")
    if os.path.exists(sc_p):
        sc = json.load(open(sc_p))
        r.update(ers=sc.get("ers"), s_ttft=sc.get("mean_s_ttft"),
                 s_tpot=sc.get("mean_s_tpot"))
    if os.path.exists(smx_p):
        # Peak (not mean) kv_cache_usage_perc across the FULL server scrape
        # time series (~every 333ms for the whole run) — NOT from
        # per_request_report.csv's kv_pct column, which only samples the
        # nearest scrape to each REQUEST'S END (~120 points here vs ~93+
        # continuous scrapes, but more importantly: the true peak can fall
        # between request completions, e.g. mid-burst while several long
        # prefills are in flight and none has finished yet — the per-request
        # column would silently never see that moment). This is the number
        # that actually says "how close did we get to evicting/preempting".
        kv_vals = []
        with open(smx_p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for entry in rec.get("metrics", {}).get("vllm:kv_cache_usage_perc", []):
                    v = entry.get("value")
                    if v is not None:
                        kv_vals.append(float(v))
        if kv_vals:
            r["kv_max"] = max(kv_vals)
    rows.append(r)

if not rows:
    print("(no experiments found)"); sys.exit(0)

def fmt(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"

def mean(xs):
    return sum(xs) / len(xs) if xs else None

def stdev(xs):
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

# raw per-rep CSV — nothing lost, every individual run is in here
csv_path = os.path.join(out, "summary.csv")
with open(csv_path, "w", newline="") as fh:
    wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wcsv.writeheader()
    for r in rows:
        wcsv.writerow(r)

# group by config name -> aggregate across reps
by_name = {}
for r in rows:
    by_name.setdefault(r["name"], []).append(r)

agg = []
for name, reps in by_name.items():
    ok = [r for r in reps if r["status"] == "ok"]
    n_fail = len(reps) - len(ok)
    entry = {"name": name, "flags": reps[0]["flags"], "n": len(reps), "n_fail": n_fail}
    for key in ("ers", "s_ttft", "s_tpot", "kv_max"):
        vals = [r[key] for r in ok if r[key] is not None]
        entry[key + "_mean"] = mean(vals)
        entry[key + "_std"] = stdev(vals)
    agg.append(entry)

multi_rep = any(a["n"] > 1 for a in agg)
hdr = ["exp", "n", "ERS", "s_ttft", "s_tpot", "KV%peak", "flags"]
w = [16, 3, 16, 9, 9, 12, 28]
line = "  ".join(h.ljust(wi) for h, wi in zip(hdr, w))
print(line); print("-" * len(line))

def cell(entry, key, d=3, pct=False):
    m, s = entry[key + "_mean"], entry[key + "_std"]
    if m is None:
        return "-"
    if pct:
        m, s = m * 100, (s * 100 if s is not None else None)
        d = 1
    if s is None:
        return f"{m:.{d}f}" + ("" if entry["n"] == 1 else " (n/a)")
    return f"{m:.{d}f}±{s:.{d}f}"

for a in sorted(agg, key=lambda a: (a["ers_mean"] is None, -(a["ers_mean"] or 0))):
    name_disp = a["name"] + (f" [{a['n_fail']} FAILED]" if a["n_fail"] else "")
    cells = [name_disp, str(a["n"]), cell(a, "ers", 4),
             cell(a, "s_ttft"), cell(a, "s_tpot"),
             cell(a, "kv_max", pct=True), a["flags"]]
    print("  ".join(str(c).ljust(wi) for c, wi in zip(cells, w)))

if multi_rep:
    print("\n(mean±stddev across reps; if two configs' ranges overlap, the difference")
    print(" is within noise — not a real effect. 'n/a' = only 1 successful rep, no")
    print(" variance estimate; re-run with more REPEATS before trusting that row.)")
else:
    print("\n(single rep per config — this is a noisy point estimate, not a fair A/B.")
    print(" Re-run with REPEATS=3+ before concluding one config beats another.)")

agg_csv = os.path.join(out, "summary_agg.csv")
with open(agg_csv, "w", newline="") as fh:
    wcsv = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
    wcsv.writeheader()
    for a in agg:
        wcsv.writerow(a)

print(f"\nwrote {csv_path} (raw, one row per rep)")
print(f"wrote {agg_csv} (aggregated mean±std per config)")
print(f"per-exp: {out}/<name>/{'rep<N>/' if multi_rep else ''}{{console.log, run/per_request_report.csv, run/score_summary.json}}")
PY
