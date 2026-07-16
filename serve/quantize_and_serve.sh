#!/bin/bash
# Container entrypoint: quantize the BF16 model BTC mounts at /model into an
# offline FP8 checkpoint, then serve THAT with vLLM (no on-the-fly --quantization).
#
# Env knobs (all optional):
#   QUANT_SRC   source BF16 dir (BTC mount)         default /model
#   QUANT_DST   output FP8 dir (writable layer)     default /model-fp8
#   QUANT_MODE  calib = static W8A8 (frozen per-tensor activation scales; the
#               only variant that can actually be FASTER than the current
#               on-the-fly dynamic fp8, because it drops per-token activation
#               scaling at decode). dynamic = data-free FP8_DYNAMIC (== what
#               --quantization=fp8 already does; NO perf gain, kept only as a
#               fast-startup fallback).                default calib
#   NUM_CALIB   calib samples (static only); lower = faster startup, less
#               representative. 0 = all 198 GPQA-diamond.   default 0
set -euo pipefail

SRC="${QUANT_SRC:-/model}"
DST="${QUANT_DST:-/model-fp8}"
MODE="${QUANT_MODE:-calib}"
NUM_CALIB="${NUM_CALIB:-0}"
IGNORE_MODE="${QUANT_IGNORE:-default}"   # default | match-online | none  (14_quantize_fp8.py --ignore-preset)
CALIB_DATA=/opt/quantize/data/GPQA/gpqa_diamond.parquet

echo "[entrypoint] $(date -u +%FT%TZ) quantize START  src=$SRC dst=$DST mode=$MODE"
t0=$(date +%s)

if [ -f "$DST/config.json" ]; then
  echo "[entrypoint] $DST already quantized — skipping (re-run: rm -rf $DST)"
else
  if [ ! -f "$SRC/config.json" ]; then
    echo "[entrypoint] FATAL: no $SRC/config.json — is the BF16 model mounted at $SRC?" >&2
    exit 1
  fi
  QARGS=(--model-dir "$SRC" --out "$DST" --skip-baseline-generate)
  if [ "$MODE" = "calib" ]; then
    QARGS+=(--calib --calib-set gpqa --calib-data "$CALIB_DATA")
    [ "$NUM_CALIB" != "0" ] && QARGS+=(--num-calib-samples "$NUM_CALIB")
  fi
  QARGS+=(--ignore-preset "$IGNORE_MODE")
  # </dev/null: 14_quantize_fp8.py prompts input() only if output is degenerate;
  # with no stdin that raises EOFError and aborts — correct (never serve garbage).
  /opt/qvenv/bin/python /opt/quantize/14_quantize_fp8.py "${QARGS[@]}" </dev/null
fi

t1=$(date +%s)
echo "[entrypoint] quantize DONE in $((t1-t0))s — launching vLLM on $DST"

# Serve with the image's SYSTEM python + vLLM (not the venv). Args come from the
# compose `command:` (which points --model at $DST and omits --quantization=fp8).
exec python3 -m vllm.entrypoints.openai.api_server "$@"
