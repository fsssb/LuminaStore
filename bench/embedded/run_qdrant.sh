#!/usr/bin/env bash
# Start/stop the Qdrant docker container used by the embedded-scenario bench.
set -euo pipefail

NAME="lumina-qdrant-bench"
IMAGE="qdrant/qdrant"

start() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" -p 6333:6333 -p 6334:6334 "$IMAGE" >/dev/null
  echo "qdrant container '$NAME' started (waiting for readiness)..."
  for _ in $(seq 1 30); do
    if curl -sf http://localhost:6333/healthz >/dev/null 2>&1; then
      echo "qdrant ready."
      exit 0
    fi
    sleep 1
  done
  echo "qdrant did not become ready in 30s" >&2
  exit 1
}

stop() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "qdrant container '$NAME' stopped."
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  *) echo "usage: $0 [start|stop]"; exit 1 ;;
esac
