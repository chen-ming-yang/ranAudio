#!/usr/bin/env bash
# Restart Kimi demo and open via Cursor Ports (see kimi/OPEN.md).
set -euo pipefail
set +o noclobber

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[restart] stop old demo…"
pkill -f 'scripts/demo_web.py' 2>/dev/null || true
fuser -k 7861/tcp 7862/tcp 2>/dev/null || true
sleep 2

echo "[restart] start…"
nohup setsid bash scripts/run_demo.sh >> demo.log 2>&1 < /dev/null &
echo "[restart] pid≈$!"

bash "$ROOT/scripts/open_in_cursor.sh"
