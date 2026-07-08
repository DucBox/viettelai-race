#!/usr/bin/env bash
# End-to-end benchmark: one command that drives load, then produces the SINGLE
# competition number (Score = 100 × ERS × f(Δ)) from the SAME running server.
#
#   ./scripts/10_bench_e2e.sh
#
# Pipeline (each step feeds the next):
#   0. scripts/serve_up.sh            fresh cold vLLM restart  (fair, repeatable)
#   1. bench/run_aiperf_baseline.sh   drive load  -> artifacts/<run>/profile_export.jsonl
#   2. scripts/07_per_request_report.py   per-request / per-turn detail table
#   3. scripts/09_gpqa_accuracy.py    198 GPQA-Diamond Qs -> accuracy  (the f(Δ) gate)
#   4. scripts/08_ers_score.py        ERS from step 1 + accuracy from step 3 -> Score
#   5. combined roll-up printed from the machine-readable summaries.
#
# WHY step 0: with --enable-prefix-caching ON, KV blocks from a previous run stay
# cached, so a 2nd back-to-back run gets prefix hits and reads artificially fast —
# not comparable to the cold first run the competition actually scores. Restarting
# the server each time makes every latency number cold and apples-to-apples.
# Steps 1 (latency) and 3 (accuracy) are INDEPENDENT measurements of the same
# server — together they are exactly the two factors the competition multiplies.
# (Accuracy is greedy/deterministic, so caching never changes it — only latency.)
#
# Env (all optional):
#   MODE=replay|trace|smoke   load profile for step 1 (default replay = real trace)
#   SKIP_SERVE=1              don't restart vLLM — measure the already-running server
#   RESET_CACHE=1             (only with SKIP_SERVE=1) POST /reset_prefix_cache first,
#                             a lightweight alternative to a full restart
#   SKIP_LOAD=1               reuse the most recent artifacts/<run>, don't re-run AIPerf
#                             (implies SKIP_SERVE — never restarts when reusing)
#   SKIP_ACC=1                skip GPQA (step 3/4 assume f(Δ)=1) — latency-only check
#   URL, SERVED_MODEL_NAME    target server (default localhost:8000 / qwen3.5-2b)
#   GPQA_LIMIT, GPQA_CONCURRENCY, GPQA_MAX_TOKENS, GPQA_ARGS   forwarded to step 3
set -euo pipefail
cd "$(dirname "$0")/.."

# Preserve EVERY caller-provided env var (e.g. 11_multi_bench.sh's per-row
# `env EXTRA_VLLM_ARGS=... GPU_MEM_UTIL=... bash scripts/10_bench_e2e.sh`, or
# any var a user exports before running this directly) across the .env source
# below. serve/.env ships REAL non-empty defaults for GPU_MEM_UTIL, MAX_NUM_SEQS,
# MODEL_DIR, etc. (only EXTRA_VLLM_ARGS ships empty) — sourcing it is a plain
# assignment that unconditionally overwrites ANY of these the caller set,
# regardless of variable name. Snapshotting every exported var beforehand and
# re-declaring them after restores exactly what the caller passed in, while
# untouched vars still pick up .env's value normally (they were never in the
# pre-snapshot, so the restore doesn't touch them).
_pre_env_declare="$(declare -p $(compgen -e) 2>/dev/null || true)"
if [[ -f serve/.env ]]; then set -a; source serve/.env; set +a; fi
eval "$_pre_env_declare"

# Activate the bench venv so `aiperf` is on PATH; its python runs the scorers too
# (07/08/09 are stdlib-only, so any python3 works — we just reuse this one).
if [[ -f bench/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source bench/.venv/bin/activate
fi
PY="$(command -v python3)"

MODE="${MODE:-replay}"
URL="${URL:-http://localhost:8000}"
SECTION() { printf '\n\033[1m========== %s ==========\033[0m\n' "$1"; }

# ── 0. fresh server (cold, fair, repeatable) ─────────────────────────────────
# Reusing artifacts (SKIP_LOAD) never touches the server, so never restart then.
if [[ "${SKIP_LOAD:-0}" == "1" ]]; then SKIP_SERVE=1; fi
if [[ "${SKIP_SERVE:-0}" == "1" ]]; then
  SECTION "0/5  SERVE  (SKIP_SERVE — using the already-running server)"
  if [[ "${RESET_CACHE:-0}" == "1" ]]; then
    echo ">> POST $URL/reset_prefix_cache (drop cached KV blocks for a cold-ish run)"
    curl -fsS -X POST "$URL/reset_prefix_cache" >/dev/null 2>&1 \
      && echo ">> prefix cache reset OK" \
      || echo "!! reset_prefix_cache not available on this server — ignoring"
  fi
else
  SECTION "0/5  SERVE  (fresh cold restart via serve_up.sh)"
  ./scripts/serve_up.sh
fi

# ── 1. drive load ────────────────────────────────────────────────────────────
if [[ "${SKIP_LOAD:-0}" == "1" ]]; then
  SECTION "1/5  LOAD  (SKIP_LOAD=1 — reusing latest artifacts/<run>)"
else
  SECTION "1/5  LOAD  (AIPerf, MODE=$MODE)"
  MODE="$MODE" ./bench/run_aiperf_baseline.sh
fi

# newest artifacts/<run> that is a real AIPerf run (has profile_export.jsonl).
RUN_DIR=""
for d in $(ls -dt artifacts/*/ 2>/dev/null); do
  if [[ -f "${d}profile_export.jsonl" ]]; then RUN_DIR="${d%/}"; break; fi
done
[[ -n "$RUN_DIR" ]] || { echo "!! no artifacts/<run>/profile_export.jsonl found — did AIPerf run?"; exit 1; }
echo ">> run dir: $RUN_DIR"

# ── 2. per-request detail ────────────────────────────────────────────────────
SECTION "2/5  PER-REQUEST REPORT (07)"
"$PY" scripts/07_per_request_report.py "$RUN_DIR" || echo "!! step 2 failed (non-fatal, continuing)"

# ── 3. accuracy gate ─────────────────────────────────────────────────────────
ACC_ARGS=(--url "${URL:-http://localhost:8000}" --model "${SERVED_MODEL_NAME:-qwen3.5-2b}"
          --out "$RUN_DIR/gpqa")
[[ -n "${GPQA_LIMIT:-}" ]]       && ACC_ARGS+=(--limit "$GPQA_LIMIT")
[[ -n "${GPQA_CONCURRENCY:-}" ]] && ACC_ARGS+=(--concurrency "$GPQA_CONCURRENCY")
[[ -n "${GPQA_MAX_TOKENS:-}" ]]  && ACC_ARGS+=(--max-tokens "$GPQA_MAX_TOKENS")
# shellcheck disable=SC2206
[[ -n "${GPQA_ARGS:-}" ]]        && ACC_ARGS+=(${GPQA_ARGS})

ACC_FILE=""
if [[ "${SKIP_ACC:-0}" == "1" ]]; then
  SECTION "3/5  ACCURACY  (SKIP_ACC=1 — f(Δ) assumed 1.0)"
else
  SECTION "3/5  ACCURACY GATE — GPQA-Diamond (09)"
  "$PY" scripts/09_gpqa_accuracy.py "${ACC_ARGS[@]}"
  ACC_FILE="$RUN_DIR/gpqa/summary.json"
fi

# ── 4. ERS + gate -> Score ───────────────────────────────────────────────────
SECTION "4/5  SCORE (08)  ERS × f(Δ)"
SCORE_ARGS=("$RUN_DIR")
[[ -n "$ACC_FILE" ]] && SCORE_ARGS+=(--accuracy-file "$ACC_FILE")
"$PY" scripts/08_ers_score.py "${SCORE_ARGS[@]}"

# ── 5. combined roll-up ──────────────────────────────────────────────────────
SECTION "5/5  E2E SUMMARY"
"$PY" - "$RUN_DIR/score_summary.json" "$ACC_FILE" <<'PY'
import json, sys
score = json.load(open(sys.argv[1]))
acc = json.load(open(sys.argv[2])) if len(sys.argv) > 2 and sys.argv[2] else None
print(f"run              : {score['run_dir']}")
print(f"requests         : {score['requests']}  (failed/empty: {score['failed']})")
print(f"ERS              : {score['ers']:.4f}   (s_ttft={score['mean_s_ttft']:.3f}  s_tpot={score['mean_s_tpot']:.3f})")
if acc:
    print(f"GPQA accuracy    : {acc['correct']}/{acc['n']} = {acc['accuracy']:.4f}"
          f"   (errors={acc['errors']} unparsed={acc['unparsed']})")
if score.get("accuracy") is None:
    print(f"f(Δ)             : {score['f_delta']:.3f}   (ASSUMED — no accuracy measured)")
else:
    print(f"f(Δ)             : {score['f_delta']:.3f}   (Δ={score['delta']:+.4f} vs baseline 0.40)")
print("-" * 52)
print(f"SCORE = 100 × ERS × f(Δ) = {score['score']:.2f}")
PY
