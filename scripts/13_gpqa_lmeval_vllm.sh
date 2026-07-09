#!/usr/bin/env bash
# GPQA-Diamond accuracy via lm-evaluation-harness, SEPARATE vLLM ENGINE mode
# (--model vllm) — NOT the HTTP server scripts/12 hits.
#
# DIFFERENT FROM scripts/12_gpqa_lmeval.sh: this does NOT talk to the running
# `vllm serve` process at all. lm-eval loads the model into ITS OWN process
# via vLLM's Python `LLM` class (offline batch inference) — using EXACTLY the
# flags you set in the VLLM_MODEL_ARGS block below. Use this when you want an
# accuracy number for a config WITHOUT going through serve_up.sh / .env /
# EXTRA_VLLM_ARGS at all — you configure the vLLM engine flags directly, here,
# by hand.
#
# IMPORTANT — do NOT run this while a live `vllm serve` is up on the same GPU:
# two separate engines both trying to load the weights = GPU memory conflict /
# likely OOM. The script detects a live server on :8000 and either warns +
# pauses 3s (default) or, with KILL_SERVER=1, stops it first (reuses
# serve_up.sh's own stop logic: PIDFILE for native, `docker rm -f` /
# `compose stop` for docker) before proceeding.
#
# WHY THIS IS A SEPARATE SCRIPT FROM 12, NOT A FLAG: scripts/12 is intentionally
# coupled to "whatever serve_up.sh actually started" — that's the whole point
# of it (measure the literal artifact that gets scored, no config-sync burden).
# This script is the opposite: an intentionally DECOUPLED, hand-configured
# engine, for a quick accuracy read on a config before/without committing it to
# serve_up.sh. Keep the flags below in sync with serve_up.sh's EXTRA_VLLM_ARGS
# by hand if you want the numbers to mean the same thing as scripts/12's.
#
#   ./bench/install_lmeval.sh   # once (shared venv with script 12)
#   source bench/.venv-lmeval/bin/activate
#   ./scripts/13_gpqa_lmeval_vllm.sh            # full run
#   ./scripts/13_gpqa_lmeval_vllm.sh --limit 20 # quick smoke
#
# SCORING METHOD — this is the LOG-LIKELIHOOD / forced-choice variant, NOT
# generation+regex: default task is gpqa_diamond_local_zeroshot
# (output_type: multiple_choice — mirrors lm-eval's official
# gpqa_diamond_zeroshot: no generation, no CoT, just compares the
# log-probability of the literal continuations "A"/"B"/"C"/"D" and picks the
# highest). Script 12 stays the generation+CoT+regex variant
# (gpqa_diamond_local_cot_zeroshot) against the real HTTP server — run both as
# a cross-check, they can legitimately disagree. Data is still our local
# data/GPQA/gpqa_diamond.parquet (not the gated Idavidrein/gpqa Hub dataset),
# fully offline. Set TASK=gpqa_diamond_local_cot_zeroshot to run the CoT
# variant through this script's separate-engine backend instead.
#
# ── EDIT THIS — the exact vLLM engine config to test ─────────────────────────
# Comma-separated key=value pairs, passed verbatim as `--model_args` to
# `lm_eval --model vllm`. Add/remove flags freely — anything vLLM's Python
# `LLM(...)` constructor accepts is valid here (quantization, kv_cache_dtype,
# gpu_memory_utilization, max_model_len, tensor_parallel_size, dtype, ...).
# `pretrained` below defaults to the same local model dir serve_up.sh uses.
VLLM_MODEL_ARGS="${VLLM_MODEL_ARGS:-pretrained=__MODEL_DIR_ABS__,dtype=auto,gpu_memory_utilization=0.8,tensor_parallel_size=1}"
# ──────────────────────────────────────────────────────────────────────────────
#
# Env (all optional):
#   GPU_ID            which physical GPU to pin this engine to via
#                     CUDA_VISIBLE_DEVICES (default 0 — same convention as
#                     serve_up.sh; on a multi-GPU host, point this at an IDLE
#                     card if serve_up.sh's server is using GPU_ID=0)
#   TASK              lm-eval task name (default gpqa_diamond_local_zeroshot,
#                     the log-likelihood variant)
#   GPQA_PARQUET      local data file (default data/GPQA/gpqa_diamond.parquet)
#   BATCH_SIZE        vLLM offline batch size (default auto)
#   MAX_GEN_TOKS      generation budget — only applies if TASK is a
#                     generate_until task (e.g. the _cot_zeroshot variant);
#                     ignored for multiple_choice tasks, nothing to generate
#                     (default 16384)
#   SYSTEM_INSTRUCTION  forces the model to sign off in the exact phrase our
#                     regex looks for (generate_until tasks only — see comment
#                     below for why this is NOT optional in practice). Empty
#                     to disable.
#   APPLY_CHAT_TEMPLATE  1 (default) formats the prompt through the model's
#                     chat template, matching what /v1/chat/completions does
#                     in script 12 (fair comparison). Set 0 to test raw
#                     completion-mode prompting instead (no chat formatting).
#   KILL_SERVER       0 (default) warn+pause if a server is up on :8000; 1 stop
#                     it first (native PIDFILE / docker rm / compose stop)
#   OUT               output dir (default artifacts/gpqa_lmeval_vllm/<ts>)
#   LMEVAL_EXTRA_ARGS extra args appended verbatim to `lm_eval`
set -euo pipefail
cd "$(dirname "$0")/.."

# Preserve caller-provided env vars across the .env source — consistent with
# every other script in this repo.
_pre_env_declare="$(declare -p $(compgen -e) 2>/dev/null || true)"
if [[ -f serve/.env ]]; then set -a; source serve/.env; set +a; fi
eval "$_pre_env_declare"

MODEL_DIR="${MODEL_DIR:-./models/qwen3.5-2b}"
case "$MODEL_DIR" in
  /*|serve/*) MODEL_DIR_ABS="$(cd "$MODEL_DIR" 2>/dev/null && pwd)" || MODEL_DIR_ABS="$MODEL_DIR" ;;
  *) MODEL_DIR_ABS="$(cd "serve/${MODEL_DIR#./}" 2>/dev/null && pwd)" || MODEL_DIR_ABS="$MODEL_DIR" ;;
esac
VLLM_MODEL_ARGS="${VLLM_MODEL_ARGS/__MODEL_DIR_ABS__/$MODEL_DIR_ABS}"

GPU_ID="${GPU_ID:-0}"
TASK="${TASK:-gpqa_diamond_local_zeroshot}"
GPQA_PARQUET="${GPQA_PARQUET:-data/GPQA/gpqa_diamond.parquet}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-16384}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"
# Empirically necessary for the CoT (generate_until) task: with NO system
# instruction, this model concludes in its own natural style ("The correct
# option is **B**." / "**Answer:** C.") — markdown-bold, no parens, never the
# literal phrase "The answer is". Neither filter ever matches that -> near-
# zero measured accuracy unrelated to the model's real capability (verified:
# full-length, non-truncated, correctly-reasoned completions, just the wrong
# sign-off phrasing). Only applied for generate_until tasks (see OUTPUT_TYPE
# check below) — meaningless for multiple_choice/log-likelihood. Set
# SYSTEM_INSTRUCTION= (empty) to disable and reproduce the raw failure mode.
SYSTEM_INSTRUCTION="${SYSTEM_INSTRUCTION:-You are an expert scientist. Reason through the problem step by step, then end your response with exactly this sentence on its own line: The answer is (X) — replacing X with the correct letter.}"
OUT="${OUT:-artifacts/gpqa_lmeval_vllm/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

KILL_SERVER="${KILL_SERVER:-0}"
if curl -fsS "http://localhost:8000/health" >/dev/null 2>&1; then
  if [[ "$KILL_SERVER" != "1" ]]; then
    echo "!! WARNING: something is answering http://localhost:8000/health right now."
    echo "   If that's a live vllm serve on this GPU, this script will try to load"
    echo "   a SECOND copy of the model alongside it — likely OOM."
    echo "   Set KILL_SERVER=1 to have this script stop it first, or Ctrl-C now."
    sleep 3
  else
    echo ">> KILL_SERVER=1 — a server is up on :8000, stopping it first ..."
    ./scripts/stop_server.sh
    if curl -fsS "http://localhost:8000/health" >/dev/null 2>&1; then
      echo "!! Server still responding after stop attempt — aborting to avoid GPU conflict." >&2
      exit 1
    fi
  fi
fi

# Generalized: match by naming convention "<task>.yaml.template" so either
# local task (the log-likelihood _zeroshot default, or _cot_zeroshot via
# TASK= override) renders correctly — not hardcoded to one task name.
TASK_DIR="bench/lmeval_tasks/gpqa_diamond_local"
TEMPLATE="$TASK_DIR/$TASK.yaml.template"
OUTPUT_TYPE=""
if [[ -f "$TEMPLATE" ]]; then
  if [[ ! -f "$GPQA_PARQUET" ]]; then
    echo "!! GPQA_PARQUET not found: $GPQA_PARQUET" >&2
    exit 1
  fi
  GPQA_PARQUET_ABS="$(cd "$(dirname "$GPQA_PARQUET")" && pwd)/$(basename "$GPQA_PARQUET")"
  sed "s|__GPQA_PARQUET_PATH__|$GPQA_PARQUET_ABS|" "$TEMPLATE" > "$TASK_DIR/$TASK.yaml"
  OUTPUT_TYPE="$(grep -m1 '^output_type:' "$TASK_DIR/$TASK.yaml" | awk '{print $2}')"
  INCLUDE_ARGS=(--include_path "$TASK_DIR")
else
  INCLUDE_ARGS=()
fi

# multiple_choice (log-likelihood) tasks don't generate anything —
# --gen_kwargs/max_gen_toks/system_instruction would be meaningless there, so
# only pass them for generate_until tasks (the CoT variant, if TASK= override
# points there).
GEN_ARGS=()
SYS_ARGS=()
if [[ "$OUTPUT_TYPE" != "multiple_choice" ]]; then
  GEN_ARGS=(--gen_kwargs "temperature=0,max_gen_toks=${MAX_GEN_TOKS}")
  [[ -n "$SYSTEM_INSTRUCTION" ]] && SYS_ARGS=(--system_instruction "$SYSTEM_INSTRUCTION")
fi

if [[ -f bench/.venv-lmeval/bin/activate ]]; then
  # shellcheck disable=SC1091
  source bench/.venv-lmeval/bin/activate
elif ! command -v lm_eval >/dev/null 2>&1; then
  echo "!! lm_eval not found and bench/.venv-lmeval doesn't exist." >&2
  echo "   Run ./bench/install_lmeval.sh first." >&2
  exit 1
fi

CHAT_ARGS=()
[[ "$APPLY_CHAT_TEMPLATE" == "1" ]] && CHAT_ARGS+=(--apply_chat_template)

echo ">> lm-eval GPQA  task=$TASK (output_type=${OUTPUT_TYPE:-unknown})  backend=vllm (separate engine, NOT the HTTP server)"
echo ">> GPU_ID=$GPU_ID (CUDA_VISIBLE_DEVICES)"
echo ">> model_args: $VLLM_MODEL_ARGS"
echo ">> apply_chat_template=$APPLY_CHAT_TEMPLATE"
if [[ ${#SYS_ARGS[@]} -gt 0 ]]; then
  echo ">> system_instruction: $SYSTEM_INSTRUCTION"
else
  echo ">> system_instruction: <none>"
fi
echo ">> out: $OUT"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  lm_eval \
  --model vllm \
  --model_args "$VLLM_MODEL_ARGS" \
  --tasks "$TASK" \
  "${INCLUDE_ARGS[@]}" \
  "${GEN_ARGS[@]}" \
  "${SYS_ARGS[@]}" \
  --batch_size "$BATCH_SIZE" \
  "${CHAT_ARGS[@]}" \
  --output_path "$OUT" \
  --log_samples \
  ${LMEVAL_EXTRA_ARGS:-} \
  "$@"

echo
echo ">> results: $OUT  (accuracy in the printed table above + results JSON)"
echo ">> per-sample generations (for auditing parse/truncation issues): $OUT/**/*.jsonl"
