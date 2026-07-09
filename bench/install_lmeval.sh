#!/usr/bin/env bash
# Install lm-evaluation-harness into its OWN venv (separate from bench/.venv,
# which is aiperf's — pinning conflicts between the two are not worth risking).
# Shared by both scripts/12 (HTTP backend) and scripts/13 (separate vLLM
# engine backend) — installs both extras they need, see below.
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

# [api]  pulls in the HTTP-client extras (requests/openai client, tokenizer
#        utils) needed for --model local-chat-completions (scripts/12 — hits
#        the ALREADY-RUNNING vllm serve endpoint over HTTP).
# [vllm] pulls in the vllm Python package itself, needed for --model vllm
#        (scripts/13 — a SEPARATE, hand-configured engine loaded in-process,
#        decoupled from serve_up.sh's config on purpose).
# Both extras in one venv so 12 and 13 share bench/.venv-lmeval, as documented
# in scripts/13's header. This is a genuinely heavy install (vllm drags in
# torch/transformers/ray/...) — expected, not a mistake.
pip install "lm-eval[api,vllm]"

echo ""
echo ">> lm-eval installed. Activate before running:"
echo "     source bench/.venv-lmeval/bin/activate"
lm_eval --version || true
