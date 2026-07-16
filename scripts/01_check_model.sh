#!/usr/bin/env bash
# Verify the model directory is complete — this repo does NOT download models.
# Model weights must already be placed at serve/models/<name>/ (e.g. via scp,
# rsync, or a manual `hf download` done elsewhere). This script only checks.
#
#   ./scripts/01_check_model.sh                       # checks serve/models/lfm2.5-1.2b
#   MODEL_DIR=serve/models/other ./scripts/01_check_model.sh
#
# Exit code 0 = complete, 1 = missing files (lists exactly what's missing).
set -euo pipefail
cd "$(dirname "$0")/.."

# Preserve a caller-provided MODEL_DIR (e.g. serve_up.sh forwarding a
# 11_multi_bench.sh row override) across the .env source below — same class of
# bug fixed in serve_up.sh / 10_bench_e2e.sh / run_aiperf_baseline.sh. Without
# this, a row testing weights at a different MODEL_DIR would have this script
# silently check serve/.env's own default directory instead — reporting the
# WRONG directory's completeness.
_pre_env_declare="$(declare -p $(compgen -e) 2>/dev/null || true)"
if [[ -f serve/.env ]]; then set -a; source serve/.env; set +a; fi
eval "$_pre_env_declare"
MODEL_DIR="${MODEL_DIR:-serve/models/lfm2.5-1.2b}"
# .env's MODEL_DIR is relative to serve/ (docker-compose context) — normalize
# so this script works whether it's given as "./models/x", "serve/models/x",
# or an absolute path (left untouched).
case "$MODEL_DIR" in
  /*) : ;;                                   # absolute path — use as-is
  serve/*) : ;;                              # already repo-root-relative
  ./*) MODEL_DIR="serve/${MODEL_DIR#./}" ;;  # "./models/x" -> "serve/models/x"
  *) MODEL_DIR="serve/$MODEL_DIR" ;;
esac

echo ">> Checking model directory: $MODEL_DIR"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "!! MISSING: directory does not exist."
  echo "   Place the model at: $MODEL_DIR/"
  echo "   Expected layout: config.json, tokenizer files, and *.safetensors"
  echo "   (see serve/models/README.md for the exact file list this model needs)."
  exit 1
fi

missing=()
warn=()

# Hard requirements: without these, vLLM cannot even load the config/tokenizer.
required=(
  config.json
  tokenizer.json
  tokenizer_config.json
)
for f in "${required[@]}"; do
  [[ -s "$MODEL_DIR/$f" ]] || missing+=("$f")
done

# Weights: either a single model.safetensors, or a sharded set described by
# model.safetensors.index.json — verify every shard the index points to.
if [[ -s "$MODEL_DIR/model.safetensors.index.json" ]]; then
  while IFS= read -r shard; do
    [[ -s "$MODEL_DIR/$shard" ]] || missing+=("$shard (referenced by model.safetensors.index.json)")
  done < <(python3 -c "
import json
d = json.load(open('$MODEL_DIR/model.safetensors.index.json'))
print('\n'.join(sorted(set(d['weight_map'].values()))))
")
elif [[ -s "$MODEL_DIR/model.safetensors" ]]; then
  : # single-shard checkpoint, present
else
  missing+=("model.safetensors or model.safetensors.index.json + shards")
fi

# Soft requirements: nice to have (chat formatting, multimodal preprocessing),
# but their absence doesn't block a text-only vLLM serve.
# LFM2.5 is text-only (no multimodal preprocessor); tokenizer.json is self-contained
# (no separate vocab.json/merges.txt). These are the nice-to-haves for chat + sampling.
optional=(chat_template.jinja generation_config.json special_tokens_map.json LICENSE)
for f in "${optional[@]}"; do
  [[ -s "$MODEL_DIR/$f" ]] || warn+=("$f")
done

if ((${#missing[@]})); then
  echo "!! MODEL INCOMPLETE — missing ${#missing[@]} required file(s):"
  for f in "${missing[@]}"; do echo "     - $f"; done
  echo "   Fix: copy the missing files into $MODEL_DIR/ and re-run this check."
  exit 1
fi

if ((${#warn[@]})); then
  echo ">> Present (required files all OK). Missing optional file(s):"
  for f in "${warn[@]}"; do echo "     - $f (non-fatal)"; done
fi

size=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
nfiles=$(find "$MODEL_DIR" -type f | wc -l | tr -d ' ')
echo ">> OK: $MODEL_DIR is complete ($nfiles files, $size)."
