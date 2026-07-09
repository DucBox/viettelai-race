#!/usr/bin/env bash
# End-to-end LATENCY benchmark: one command that drives load against a fresh,
# cold-restarted server and produces ERS (the Effective Request Score) from
# it. Accuracy/f(Δ) is measured separately (scripts/09_gpqa_accuracy.py or
# scripts/12_gpqa_lmeval.sh) — this pipeline is latency-only, on purpose.
#
#   ./scripts/10_bench_e2e.sh
#
# Pipeline (each step feeds the next):
#   0. scripts/serve_up.sh            fresh cold vLLM restart  (fair, repeatable)
#   1. bench/run_aiperf_baseline.sh   drive load  -> artifacts/<run>/profile_export.jsonl
#   2. scripts/07_per_request_report.py   per-request / per-turn detail table
#   3. scripts/08_ers_score.py        ERS from step 1
#
# WHY step 0: with --enable-prefix-caching ON, KV blocks from a previous run stay
# cached, so a 2nd back-to-back run gets prefix hits and reads artificially fast —
# not comparable to the cold first run the competition actually scores. Restarting
# the server each time makes every latency number cold and apples-to-apples.
#
# Env (all optional):
#   MODE=replay|trace|smoke   load profile for step 1 (default replay = real trace)
#   SKIP_SERVE=1              don't restart vLLM — measure the already-running server
#   RESET_CACHE=1             (only with SKIP_SERVE=1) POST /reset_prefix_cache first,
#                             a lightweight alternative to a full restart
#   SKIP_LOAD=1               reuse the most recent artifacts/<run>, don't re-run AIPerf
#                             (implies SKIP_SERVE — never restarts when reusing)
#   STOP_AFTER=1              stop the server (scripts/stop_server.sh) after ERS is
#                             computed — only if THIS run started it fresh in step 0
#                             (never touches a server left up via SKIP_SERVE). Use this
#                             so the GPU actually goes idle between runs instead of a
#                             server sitting there until the NEXT run's serve_up.sh
#                             kills it — e.g. to let scripts/13's separate engine run
#                             without a KILL_SERVER=1 fight. 11_multi_bench.sh sets
#                             this for every row by default.
#   URL, SERVED_MODEL_NAME    target server (default localhost:8000 / qwen3.5-2b)
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
# (07/08 are stdlib-only, so any python3 works — we just reuse this one).
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
WE_STARTED_SERVER=0
if [[ "${SKIP_SERVE:-0}" == "1" ]]; then
  SECTION "0/3  SERVE  (SKIP_SERVE — using the already-running server)"
  if [[ "${RESET_CACHE:-0}" == "1" ]]; then
    echo ">> POST $URL/reset_prefix_cache (drop cached KV blocks for a cold-ish run)"
    curl -fsS -X POST "$URL/reset_prefix_cache" >/dev/null 2>&1 \
      && echo ">> prefix cache reset OK" \
      || echo "!! reset_prefix_cache not available on this server — ignoring"
  fi
else
  SECTION "0/3  SERVE  (fresh cold restart via serve_up.sh)"
  ./scripts/serve_up.sh
  WE_STARTED_SERVER=1
fi

# ── 1. drive load ────────────────────────────────────────────────────────────
if [[ "${SKIP_LOAD:-0}" == "1" ]]; then
  SECTION "1/3  LOAD  (SKIP_LOAD=1 — reusing latest artifacts/<run>)"
else
  SECTION "1/3  LOAD  (AIPerf, MODE=$MODE)"
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
SECTION "2/3  PER-REQUEST REPORT (07)"
"$PY" scripts/07_per_request_report.py "$RUN_DIR" || echo "!! step 2 failed (non-fatal, continuing)"

# ── 3. ERS ───────────────────────────────────────────────────────────────────
SECTION "3/3  ERS (08)"
"$PY" scripts/08_ers_score.py "$RUN_DIR"

# ── stop (optional) ──────────────────────────────────────────────────────────
# Only stop a server THIS run started — never touch one left up via SKIP_SERVE,
# that's the caller's to manage.
if [[ "${STOP_AFTER:-0}" == "1" && "$WE_STARTED_SERVER" == "1" ]]; then
  SECTION "STOP  (STOP_AFTER=1 — freeing the GPU until the next run)"
  ./scripts/stop_server.sh
fi
