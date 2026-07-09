#!/usr/bin/env bash
# Stop whatever vLLM server serve_up.sh started — native PIDFILE, docker
# container, or docker-compose service, whichever applies. Idempotent: safe
# to call even if nothing is running (just reports "nothing to stop").
#
#   ./scripts/stop_server.sh
#
# Factored out of serve_up.sh's own restart logic and scripts/13's
# KILL_SERVER path so there's exactly ONE place that knows how to tear a
# server down, instead of three copies drifting apart.
set -euo pipefail
cd "$(dirname "$0")/.."

PIDFILE="serve/.vllm.pid"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  OLD_PID="$(cat "$PIDFILE")"
  echo ">> Stopping native vllm serve (pid $OLD_PID) ..."
  kill "$OLD_PID" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$OLD_PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo ">> Still alive after 30s — SIGKILL ..."
    kill -9 "$OLD_PID" 2>/dev/null || true
    sleep 2
  fi
  rm -f "$PIDFILE"
  echo ">> Stopped."
elif docker ps -q --filter name=vllm-qwen35 2>/dev/null | grep -q .; then
  echo ">> Stopping docker container vllm-qwen35 ..."
  docker rm -f vllm-qwen35 >/dev/null 2>&1 || true
  echo ">> Stopped."
elif [[ -f serve/docker-compose.yml ]] && (cd serve && docker compose ps -q vllm 2>/dev/null | grep -q .); then
  echo ">> Stopping docker compose service 'vllm' ..."
  ( cd serve && docker compose stop vllm ) || true
  echo ">> Stopped."
else
  echo ">> No running vllm serve detected (native PIDFILE / docker container / compose service) — nothing to stop."
fi
