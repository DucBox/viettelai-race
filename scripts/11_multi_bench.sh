#!/usr/bin/env bash
# Sweep several serve configurations through the full e2e benchmark and produce
# ONE comparison table. Each EXPERIMENT row is a distinct vLLM serve config; the
# script restarts the server cold per row (10_bench_e2e.sh step 0) so every
# number is fair, then aggregates accuracy + ERS + Score across all rows.
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
# Each row's outputs land in artifacts/multibench/<ts>/<name>/:
#   console.log            full terminal output (the 2 AIPerf tables + 07 table live here)
#   run/                   the AIPerf run dir: profile_export.jsonl, server_metrics_export.*,
#                          per_request_report.csv (07), gpqa/summary.json, score_summary.json
#   exp_meta.json          {name, extra_vllm_args, extra_env}
# Plus a top-level summary.csv + printed comparison table across all rows.
#
# Env: MODE (default replay), GPQA_LIMIT / GPQA_CONCURRENCY / GPQA_MAX_TOKENS,
#      URL, SERVED_MODEL_NAME — forwarded to every row so the comparison is
#      apples-to-apples.
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

TS="$(date +%Y%m%d_%H%M%S)"
OUT="artifacts/multibench/$TS"
mkdir -p "$OUT"
echo ">> multi-bench $TS  ($MODE)  ->  $OUT"
echo ">> ${#EXPERIMENTS[@]} experiments"

n=0
for row in "${EXPERIMENTS[@]}"; do
  # split on tab into name / args / extra-env
  IFS=$'\t' read -r name args extra <<<"$row"
  name="${name## }"; name="${name%% }"
  [[ -z "$name" || "$name" == \#* ]] && continue
  n=$((n + 1))
  EXPDIR="$OUT/$name"
  mkdir -p "$EXPDIR"

  printf '\n\033[1m########## EXP %d/%d: %s ##########\033[0m\n' \
    "$n" "${#EXPERIMENTS[@]}" "$name"
  echo ">> EXTRA_VLLM_ARGS = ${args:-<none>}"
  echo ">> extra env       = ${extra:-<none>}"

  # scope env to this e2e invocation; MODE + GPQA_* forwarded from our env
  EXP_ENV=(EXTRA_VLLM_ARGS="$args" MODE="$MODE")
  # shellcheck disable=SC2206
  [[ -n "${extra// }" ]] && EXP_ENV+=($extra)
  [[ -n "${GPQA_LIMIT:-}" ]]       && EXP_ENV+=(GPQA_LIMIT="$GPQA_LIMIT")
  [[ -n "${GPQA_CONCURRENCY:-}" ]] && EXP_ENV+=(GPQA_CONCURRENCY="$GPQA_CONCURRENCY")
  [[ -n "${GPQA_MAX_TOKENS:-}" ]]  && EXP_ENV+=(GPQA_MAX_TOKENS="$GPQA_MAX_TOKENS")

  # exp_meta.json (record the exact config for the aggregate)
  python3 - "$EXPDIR/exp_meta.json" "$name" "$args" "$extra" <<'PY'
import json, sys
json.dump({"name": sys.argv[2], "extra_vllm_args": sys.argv[3],
           "extra_env": sys.argv[4]}, open(sys.argv[1], "w"), indent=2)
PY

  # run the full e2e (cold restart -> load -> 07 -> GPQA -> ERS); save all output.
  set +e
  env "${EXP_ENV[@]}" bash scripts/10_bench_e2e.sh 2>&1 | tee "$EXPDIR/console.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "!! EXP '$name' failed (rc=$rc) — see $EXPDIR/console.log; continuing."
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

# ── aggregate ────────────────────────────────────────────────────────────────
printf '\n\033[1m========== MULTI-BENCH SUMMARY ==========\033[0m\n'
python3 - "$OUT" <<'PY'
import csv, glob, json, os, sys

out = sys.argv[1]
rows = []
for meta_path in sorted(glob.glob(os.path.join(out, "*", "exp_meta.json"))):
    exp = os.path.dirname(meta_path)
    meta = json.load(open(meta_path))
    r = {"name": meta["name"], "flags": meta["extra_vllm_args"] or "-",
         "status": "ok", "acc": None, "unparsed": None, "ers": None,
         "s_ttft": None, "s_tpot": None, "f": None, "score": None}
    if os.path.exists(os.path.join(exp, "FAILED")):
        r["status"] = "FAILED"
    sc_p = os.path.join(exp, "run", "score_summary.json")
    gp_p = os.path.join(exp, "run", "gpqa", "summary.json")
    if os.path.exists(sc_p):
        sc = json.load(open(sc_p))
        r.update(ers=sc.get("ers"), s_ttft=sc.get("mean_s_ttft"),
                 s_tpot=sc.get("mean_s_tpot"), f=sc.get("f_delta"),
                 score=sc.get("score"), acc=sc.get("accuracy"))
    if os.path.exists(gp_p):
        gp = json.load(open(gp_p))
        r["acc"] = gp.get("accuracy", r["acc"])
        r["unparsed"] = gp.get("unparsed")
    rows.append(r)

if not rows:
    print("(no experiments found)"); sys.exit(0)

def f(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"

hdr = ["exp", "acc", "unpars", "ERS", "s_ttft", "s_tpot", "f(Δ)", "SCORE", "flags"]
w = [12, 6, 6, 7, 7, 7, 6, 7, 30]
line = "  ".join(h.ljust(wi) for h, wi in zip(hdr, w))
print(line); print("-" * len(line))
# best score first for readability, failures last
for r in sorted(rows, key=lambda r: (r["score"] is None, -(r["score"] or 0))):
    cells = [r["name"], f(r["acc"]), ("-" if r["unparsed"] is None else str(r["unparsed"])),
             f(r["ers"], 4), f(r["s_ttft"]), f(r["s_tpot"]), f(r["f"]),
             f(r["score"], 2), (r["flags"] if r["status"] == "ok" else f"[{r['status']}] {r['flags']}")]
    print("  ".join(str(c).ljust(wi) for c, wi in zip(cells, w)))

csv_path = os.path.join(out, "summary.csv")
with open(csv_path, "w", newline="") as fh:
    wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wcsv.writeheader()
    for r in rows:
        wcsv.writerow(r)
print(f"\nwrote {csv_path}")
print(f"per-exp: {out}/<name>/{{console.log, run/per_request_report.csv, run/score_summary.json}}")
PY
