#!/usr/bin/env bash
# GPQA-Diamond accuracy via lm-evaluation-harness (EleutherAI), instead of the
# hand-rolled scripts/09_gpqa_accuracy.py — standardized prompt template +
# answer-extraction logic instead of our own regex/system-prompt.
#
#   ./bench/install_lmeval.sh              # once
#   ./scripts/12_gpqa_lmeval.sh            # full run
#   ./scripts/12_gpqa_lmeval.sh --limit 20 # quick smoke (few examples)
#
# DATA SOURCE: by default this uses a CUSTOM local task
# (gpqa_diamond_local_cot_zeroshot, defined in
# bench/lmeval_tasks/gpqa_diamond_local/) that reads data/GPQA/gpqa_diamond.
# parquet — the same file scripts/09 uses — NOT lm-eval's built-in
# gpqa_diamond_cot_zeroshot task, whose dataset (Idavidrein/gpqa on HF Hub) is
# GATED (needs an HF account that accepted its terms + a token) and needs raw
# Question/Correct Answer/Incorrect Answer N columns our local parquet doesn't
# have. The prompt/answer-extraction-regex/generation_kwargs/metric in the
# local task are copied verbatim from lm-eval's own official task definition
# (EleutherAI/lm-evaluation-harness, lm_eval/tasks/gpqa/cot_zeroshot/) — same
# standardized scoring pipeline, just pointed at data we already have, fully
# offline, no HF account needed. KNOWN LIMITATION vs the official task: it
# does NOT get lm-eval's random answer-choice reshuffling (the official
# task's process_docs() defeats position bias by reshuffling A/B/C/D on every
# run; our source data has a fixed lettering already baked into the question
# text, same limitation scripts/09 already has). Set TASK=gpqa_diamond_cot_zeroshot
# to use the official gated task instead, IF you've pre-cached it (see git
# history of this file for the HF-cache-transfer offline recipe).
#
# CRITICAL — this hits the ALREADY-RUNNING vllm serve over HTTP
# (--model local-chat-completions -> $URL/v1/chat/completions), the SAME
# server AIPerf/ERS just measured. Do NOT use `--model vllm` — that makes
# lm-eval load weights into its OWN separate process via the vLLM Python API,
# completely decoupled from serve_up.sh's config (quantization, kv-cache-dtype,
# gpu-mem-util, ...). That would silently test a DIFFERENT engine than the one
# actually serving — exactly the config-drift class of bug this repo spent a
# lot of effort eliminating elsewhere (see git log: serve_up.sh / 10_bench_e2e
# / run_aiperf_baseline / 01_check_model env-clobber fixes).
#
# UNVERIFIED ON A LIVE SERVER — lm-eval's exact CLI flag names can vary by
# version. Before trusting this in an automated sweep, run it standalone once
# and check the printed accuracy + a few --log_samples records for
# garbage/truncated generations.
#
# Env (all optional):
#   URL, SERVED_MODEL_NAME   target server (default localhost:8000 / qwen3.5-2b)
#   TASK                     lm-eval task name (default gpqa_diamond_local_cot_zeroshot)
#   GPQA_PARQUET             local data file (default data/GPQA/gpqa_diamond.parquet)
#   NUM_CONCURRENT           parallel in-flight requests (default 32)
#   MAX_RETRIES              per-request retry budget (default 3)
#   MAX_GEN_TOKS             generation budget — CoT needs room to think before
#                            answering, same lesson as scripts/09 (default 2048)
#   OUT                      output dir (default artifacts/gpqa_lmeval/<ts>)
#   LMEVAL_EXTRA_ARGS        extra args appended verbatim to `lm_eval`
set -euo pipefail
cd "$(dirname "$0")/.."

# Preserve caller-provided env vars across the .env source — consistent with
# every other script in this repo (serve_up.sh / 10_bench_e2e.sh / etc.).
_pre_env_declare="$(declare -p $(compgen -e) 2>/dev/null || true)"
if [[ -f serve/.env ]]; then set -a; source serve/.env; set +a; fi
eval "$_pre_env_declare"

URL="${URL:-http://localhost:8000}"
MODEL="${SERVED_MODEL_NAME:-qwen3.5-2b}"
TASK="${TASK:-gpqa_diamond_local_cot_zeroshot}"
GPQA_PARQUET="${GPQA_PARQUET:-data/GPQA/gpqa_diamond.parquet}"
NUM_CONCURRENT="${NUM_CONCURRENT:-32}"
MAX_RETRIES="${MAX_RETRIES:-3}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-2048}"
OUT="${OUT:-artifacts/gpqa_lmeval/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

TASK_DIR="bench/lmeval_tasks/gpqa_diamond_local"
if [[ "$TASK" == "gpqa_diamond_local_cot_zeroshot" ]]; then
  if [[ ! -f "$GPQA_PARQUET" ]]; then
    echo "!! GPQA_PARQUET not found: $GPQA_PARQUET" >&2
    exit 1
  fi
  GPQA_PARQUET_ABS="$(cd "$(dirname "$GPQA_PARQUET")" && pwd)/$(basename "$GPQA_PARQUET")"
  # Render the template -> a real task YAML with the absolute local path
  # substituted in (gitignored; regenerated every run, cheap and avoids any
  # path-portability issue across machines/checkouts).
  sed "s|__GPQA_PARQUET_PATH__|$GPQA_PARQUET_ABS|" \
    "$TASK_DIR/gpqa_diamond_local_cot_zeroshot.yaml.template" \
    > "$TASK_DIR/gpqa_diamond_local_cot_zeroshot.yaml"
  INCLUDE_ARGS=(--include_path "$TASK_DIR")
else
  INCLUDE_ARGS=()
fi

if [[ -f bench/.venv-lmeval/bin/activate ]]; then
  # shellcheck disable=SC1091
  source bench/.venv-lmeval/bin/activate
elif ! command -v lm_eval >/dev/null 2>&1; then
  echo "!! lm_eval not found and bench/.venv-lmeval doesn't exist." >&2
  echo "   Run ./bench/install_lmeval.sh first." >&2
  exit 1
fi

echo ">> lm-eval GPQA  task=$TASK  server=$URL  model=$MODEL"
echo ">> out: $OUT"

# HF_HUB_OFFLINE=1 is harmless (and correct) even for the local task — it
# doesn't touch the Hub at all; this just guarantees no accidental network
# call is attempted if TASK is switched to the official gated task.
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  lm_eval \
  --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${URL}/v1/chat/completions,num_concurrent=${NUM_CONCURRENT},max_retries=${MAX_RETRIES},tokenized_requests=False" \
  --tasks "$TASK" \
  "${INCLUDE_ARGS[@]}" \
  --gen_kwargs "temperature=0,max_gen_toks=${MAX_GEN_TOKS}" \
  --output_path "$OUT" \
  --log_samples \
  ${LMEVAL_EXTRA_ARGS:-} \
  "$@"

echo
echo ">> results: $OUT  (accuracy in the printed table above + results JSON)"
echo ">> per-sample generations (for auditing parse/truncation issues): $OUT/**/*.jsonl"
