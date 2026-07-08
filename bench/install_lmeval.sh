#!/usr/bin/env bash
# Install lm-evaluation-harness into its OWN venv (separate from bench/.venv,
# which is aiperf's — lm-eval pulls in transformers/torch even for the
# API-only backend used here, and pinning conflicts with aiperf's deps are not
# worth risking).
#
#   ./bench/install_lmeval.sh
#   source bench/.venv-lmeval/bin/activate
#
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv-lmeval
# shellcheck disable=SC1091
source .venv-lmeval/bin/activate
pip install -U pip

# [api] pulls in the HTTP-client extras (requests/openai client, tokenizer
# utils) needed for --model local-chat-completions — we're hitting the
# ALREADY-RUNNING vllm serve endpoint over HTTP, not loading weights into
# this process (that would be --model vllm, which spins up a SEPARATE engine
# decoupled from serve_up.sh's config — not what we want here).
pip install "lm-eval[api]"

echo ""
echo ">> lm-eval installed. Activate before running:"
echo "     source bench/.venv-lmeval/bin/activate"
lm_eval --version || true
