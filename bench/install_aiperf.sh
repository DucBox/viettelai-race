#!/usr/bin/env bash
# Install AIPerf into a local venv, from the vendored clone at ../aiperf when
# present (single source of truth — same code as ../aiperf/docs reference),
# falling back to the PyPI release otherwise.
#
#   ./bench/install_aiperf.sh
#   source bench/.venv/bin/activate
#
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip

# On Linux aarch64 you may first need: sudo apt install build-essential
if [[ -f ../aiperf/pyproject.toml ]]; then
  echo ">> installing AIPerf from local clone ../aiperf (editable)"
  pip install -e ../aiperf
else
  echo ">> ../aiperf clone not found — installing AIPerf from PyPI instead"
  pip install aiperf
fi

echo ""
echo ">> AIPerf installed. Activate the venv before running benchmarks:"
echo "     source bench/.venv/bin/activate"
aiperf --version || true
